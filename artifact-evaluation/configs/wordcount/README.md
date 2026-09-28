# Wordcount validation configuration

`recomputable_ec.config` is the six-endpoint EC 4+2 configuration used by the
small recomputable/recipe correctness runs.  It keeps backup and Resident
placement disabled. Run the `wordcount_recovery` validation binary, with its
application mode selected by environment variables
(`FARLIB_WORDCOUNT_RECOMPUTABLE` or `FARLIB_WORDCOUNT_RECIPES`) so the same
configuration can exercise the protected and recomputable paths.

Use this configuration for correctness tests, not as a performance baseline.
