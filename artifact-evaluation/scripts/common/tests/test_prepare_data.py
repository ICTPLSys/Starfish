import contextlib
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_data as data


@contextlib.contextmanager
def server(payload):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            offset = int(self.headers.get("Range", "bytes=0-")[6:].split("-")[0])
            self.send_response(206 if offset else 200)
            if offset:
                self.send_header("Content-Range",
                                 f"bytes {offset}-{len(payload)-1}/{len(payload)}")
            self.send_header("Content-Length", str(len(payload) - offset))
            self.end_headers()
            self.wfile.write(payload[offset:])

        def log_message(self, *_):
            pass
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/data"
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


class DataPreparationTests(unittest.TestCase):
    def test_source_cache_verifies_bytes_and_records_actual_retrieval(self):
        payload = b"already downloaded original source\n"
        spec = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            source = cache / "input.gz"
            source.write_bytes(payload)
            target = root / "downloads/input.gz"
            url = "https://example.invalid/original/input.gz"
            with patch.dict(data.os.environ, {"AE_SOURCE_CACHE": str(cache)}):
                data.download(url, target, spec)
            self.assertEqual(target.read_bytes(), payload)
            record = json.loads(target.with_name("input.gz.download.json").read_text())
            self.assertEqual(record["url"], url)
            self.assertEqual(record["retrieved_from"], source.as_uri())

    def test_bfs_download_prepare_register_and_reuse(self):
        raw = b"# fixture\n" * 4 + b"1 2\n" * 124
        payload = gzip.compress(raw, mtime=0)
        with tempfile.TemporaryDirectory() as directory, server(payload) as url:
            root = Path(directory)
            source = {"url": url, "size": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest(),
                      "lines": 128, "uncompressed_size": len(raw)}
            free = data.shutil._ntuple_diskusage(1 << 40, 0, 1 << 40)
            args = ["--datasets", "bfs", "--data-dir", str(root / "inputs"),
                    "--site", str(root / "site.json")]
            with patch.dict(data.SOURCES, {"bfs": source}), \
                 patch.object(data.shutil, "disk_usage", return_value=free):
                self.assertEqual(data.main(args), 0)
                graph = root / "inputs/bfs"
                self.assertEqual(b"".join((graph / f"graph.{i}").read_bytes()
                                         for i in range(32)), raw)
                site = json.loads((root / "site.json").read_text())
                self.assertEqual(site["inputs"]["bfs"], str(graph / "graph"))
                with patch.object(data, "run", side_effect=AssertionError("unexpected download")):
                    self.assertEqual(data.main(args), 0)
                (graph / "manifest.json").write_text("{}")
                with self.assertRaises(RuntimeError):
                    data.main(args)

    def test_root_readme_shell_commands_exist_and_parse(self):
        readme = data.ROOT.parent / "README.md"
        commands = re.findall(r"\bbash (scripts/[\w./-]+\.sh)", readme.read_text())
        self.assertTrue(commands)
        for script in commands:
            target = data.ROOT / script
            self.assertTrue(target.is_file(), f"README names a missing script: {script}")
            data.run(["bash", "-n", target])

    def test_real_http_resume_and_receipt(self):
        payload = b"fixture input data\n" * 100
        expected = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        with tempfile.TemporaryDirectory() as directory, server(payload) as url:
            target = Path(directory) / "data"
            target.with_name("data.part").write_bytes(payload[:41])
            data.download(url, target, expected)
            self.assertEqual(target.read_bytes(), payload)
            with patch.object(data, "run", side_effect=AssertionError("unexpected network")):
                data.download(url, target, expected)
            target.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "unverified"):
                data.download(url, target, expected)

    def test_wrong_checksum_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory, server(b"abc") as url:
            target = Path(directory) / "data"
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                data.download(url, target, {"size": 3, "sha256": "0" * 64})
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name("data.download.json").exists())

    def test_git_blob_checksum(self):
        payload = b'{"dim":4096}\n'
        expected = {"size": len(payload),
                    "git_blob": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()}
        with tempfile.TemporaryDirectory() as directory, server(payload) as url:
            target = Path(directory) / "params.json"
            data.download(url, target, expected)
            self.assertEqual(target.read_bytes(), payload)

    def test_dry_run_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "inputs"
            self.assertEqual(data.main(["--data-dir", str(target), "--dry-run"]), 0)
            self.assertFalse(target.exists())

    def test_missing_authorization_precedes_download(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(data.os.environ, {}, clear=True):
            target = Path(directory) / "inputs"
            with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
                data.main(["--data-dir", str(target)])
            self.assertFalse(target.exists())

    def test_site_registration_preserves_server_and_user_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory) / "site.json"
            site.write_text(json.dumps({"memory_host": "my-memory",
                             "inputs": {"llama": "/existing/model",
                                        "bfs": "/path/to/graph"}}))
            data.register_inputs(site, {"llama": "/new/model", "bfs": "/new/graph"})
            actual = json.loads(site.read_text())
            self.assertEqual(actual["memory_host"], "my-memory")
            self.assertEqual(actual["inputs"], {"llama": "/existing/model", "bfs": "/new/graph"})


if __name__ == "__main__":
    unittest.main()
