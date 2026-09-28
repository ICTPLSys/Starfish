# Vendored artifact-evaluation dependencies

This directory contains a source export used by the Linux artifact-evaluation
setup scripts. The export is not a Git submodule and contains no Git metadata
or build products.

- libfibre: https://github.com/Crazylqx/libfibre
  commit `885d74dfb6746966a911ff6c821cc4a88fda3b91`, license files preserved in `libfibre/`.
- errnoname submodule: https://github.com/mentalisttraceur/errnoname
  commit `610a51162641f21fcac2ba812bc1996da11525f1`, exported under `libfibre/src/errnoname/`.
- The optional upstream `apps/picohttpparser` submodule at commit
  `066d2b1e9ab820703db0837a7255d92d30f0c9f5` is intentionally omitted because AE builds `src/all` only.
- `libfibre.vendor.json` records every exported tracked regular file with
  SHA-256, upstream Git blob SHA-1, and mode. Generated objects and shared
  libraries belong under ignored `artifact-evaluation/deps/`.
