"""Stage the compute build in an owned memory-service directory."""

from pathlib import Path
import re
import shlex
import subprocess
import tarfile


def pack_source(runtime, archive):
    with tarfile.open(archive, "w:gz") as bundle:
        for relative in ("include", "src/rdma/server.cpp", "src/rdma/exchange_msg.cpp"):
            bundle.add(runtime / relative, arcname=relative)


def build_command(directory):
    root = shlex.quote(directory)
    return (f"cd {root}; mkdir source; tar -xzf server-source.tar.gz -C source; "
            "command -v g++ >/dev/null || "
            "{ echo 'install g++ and libibverbs-dev on the memory host' >&2; exit 2; }; "
            "g++ -std=gnu++2a -O3 -DNDEBUG "
            "-DFARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION=1 -pthread "
            "-Isource/include source/src/rdma/server.cpp "
            "source/src/rdma/exchange_msg.cpp -libverbs -o server.native "
            "> build.log 2>&1 || { cat build.log >&2; exit 2; }; "
            "mv server.native server")


def dependencies(binary):
    result = subprocess.run(["ldd", str(binary)], text=True, capture_output=True)
    if result.returncode or "not found" in result.stdout + result.stderr:
        raise ValueError(f"unresolved local server dependencies: {binary}: "
                         + result.stdout + result.stderr)
    private = []
    for name, path in re.findall(r"^\s*(\S+) => (/\S+) ", result.stdout, re.MULTILINE):
        # System libraries must be installed on the memory host, not copied
        # across distributions with an incompatible loader or verbs provider.
        if name.startswith(("libfibre.", "liberrnoname.")):
            private.append((Path(path).resolve(), name))
        elif not path.startswith(("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")):
            raise ValueError(f"unsupported private server dependency: {name}: {path}")
    return private


def check_script(binary, library_path=None):
    prefix = ("env LD_LIBRARY_PATH=" + shlex.quote(library_path) + " "
              if library_path else "")
    return ("test -x " + shlex.quote(binary) + "; "
            "deps=$(" + prefix + "ldd " + shlex.quote(binary) + " 2>&1) || "
            '{ printf "%s\\n" "$deps" >&2; exit 2; }; '
            'if printf "%s\\n" "$deps" | grep -q "not found"; then '
            'printf "%s\\n" "$deps" >&2; exit 2; fi')


def stage(host, directory, binary, private, *, ssh, scp, source_archive=None):
    # The caller has just created this unique run directory. Never replace a
    # pre-existing executable, including one from another incomplete run.
    destination = directory + "/server"
    ssh(host, "test ! -e " + shlex.quote(destination))
    scp(str(binary), host + ":" + destination)
    for source, name in private:
        scp(str(source), host + ":" + directory + "/" + name)
    rebuilt = False
    try:
        ssh(host, "chmod 700 " + shlex.quote(destination) + "; "
            + check_script(destination, directory))
    except (RuntimeError, subprocess.SubprocessError):
        if source_archive is None:
            raise
        # Older memory-node glibc cannot load a compute-node executable.
        # Compile the same two-source server target natively; no runtime
        # algorithm or operating-system library is replaced.
        scp(str(source_archive), host + ":" + directory + "/server-source.tar.gz")
        ssh(host, build_command(directory))
        ssh(host, check_script(destination, directory))
        rebuilt = True
    return {"binary": destination, "rebuilt_on_memory_host": rebuilt}
