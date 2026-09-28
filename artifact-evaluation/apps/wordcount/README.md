# Wordcount

All runtimes compile `wordcount_far.cpp`: `wordcount_far <config> <dataset> [threads] [max_bytes] [top_k]`.

`FARLIB_WORDCOUNT_STARFISH` enables Starfish-specific recomputation hooks
(default ON for Starfish, OFF for NonFT/Hydra). The
`FARLIB_WORDCOUNT_RECOMPUTABLE`/`FARLIB_WORDCOUNT_RECIPES` environment flags
select the Starfish mode; NonFT/Hydra builds ignore them. Recipe mode requires
`FARLIB_WORDCOUNT_KEEP_LOCAL_FAR_MAPS=1`; workload and map settings come from
the selected configuration.

`FARLIB_BUILD_WORDCOUNT_RECOVERY=ON` additionally builds `wordcount_recovery`
from that same source, with Starfish fault gates and verification enabled.
Normal `wordcount_far` binaries exclude those test sections.
