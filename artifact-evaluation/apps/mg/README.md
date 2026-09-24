MG is the far-memory multigrid application based on NPB; run `mg <client.config>`.

Build with `-DFARLIB_BUILD_MG=ON`. Defaults are 1024³, five iterations and
24 fibres; `mg.input` overrides grid/iteration settings and
`-DMG_UTHREAD_COUNT=N` selects fibres at build time. The five-iteration workload
is NPB Class U, not an official Class D verification run.
