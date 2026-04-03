## Plan: Reduce OpenAlex 404 CLI Noise

Recommendation: treat OpenAlex 404 as expected-not-found events for interactive output, aggregate them into end-of-run stats, and keep an opt-in diagnostic mode for per-request visibility. This preserves debuggability while removing high-frequency clutter from normal runs.

**Steps**
1. Define failure-severity policy and output behavior in the HTTP wrapper in [citation_ingestion.py](citation_ingestion.py#L367): classify permanent failures by HTTP code and map each to an output tier (silent, summary-only, immediate warning).
2. Add structured failure counters in [citation_ingestion.py](citation_ingestion.py#L332) for per-code and per-provider breakdown (for example: total_404, total_403, total_5xx) so the CLI can report concise summaries instead of per-event lines.
3. Update request-error output in [citation_ingestion.py](citation_ingestion.py#L367): suppress per-request verbose printing for 404 by default, keep immediate output for higher-signal failures (401, 403, repeated 5xx), and retain full detail in logger debug entries.
4. Add an explicit diagnostics switch in [cli.py](cli.py#L113) and pass it into ingestion setup in [cli.py](cli.py#L311): allow users to opt in to full per-request 404 logging when troubleshooting.
5. Add periodic failure summary messaging where ingestion results are reported in [cli.py](cli.py): emit rolling counts every N requests/time window (for example every 50 requests or 15 seconds) with total failures and per-code counts, plus a final end-of-run summary for completeness.
6. Ensure metadata output already carrying fetch stats in [citation_ingestion.py](citation_ingestion.py#L1695) is expanded to include per-code counts so downstream tooling and tests can assert behavior without parsing logs.
7. Add or update tests for noise-control behavior:
- unit coverage for HTTP wrapper classification and counter updates in [tests/test_citation_ingestion.py](tests/test_citation_ingestion.py)
- CLI behavior checks for default summary-only versus diagnostics mode in [tests/test_cli.py](tests/test_cli.py)
- regression checks that non-404 failures remain visible in [tests/test_exception_handling.py](tests/test_exception_handling.py)
8. Verify interactively with a smoke run against a known-not-found-heavy seed and confirm output remains readable while summary counts still reflect actual misses.

**Relevant files**
- [citation_ingestion.py](citation_ingestion.py) — adjust _safe_get failure handling, counters, and exported fetch stats.
- [cli.py](cli.py) — add diagnostics flag and concise end-of-run reporting path.
- [tests/test_citation_ingestion.py](tests/test_citation_ingestion.py) — validate HTTP code classification and stat accounting.
- [tests/test_cli.py](tests/test_cli.py) — validate user-facing output in default and diagnostics modes.
- [tests/test_exception_handling.py](tests/test_exception_handling.py) — confirm high-severity failures remain surfaced.

**Verification**
1. Run targeted tests for ingestion and CLI output behavior: pytest tests/test_citation_ingestion.py tests/test_cli.py tests/test_exception_handling.py.
2. Run a smoke CLI ingestion with default output and confirm repeated OpenAlex 404 lines are absent while an aggregate failure summary appears.
3. Run the same command with diagnostics enabled and confirm per-request 404 lines are visible for troubleshooting.
4. Validate produced metadata includes per-code failure stats in the fetch_stats section.

**Decisions**
- Included scope: output behavior, stats model, CLI flag, and tests.
- Excluded scope: changing OpenAlex query semantics or replacing provider-level retrieval strategy.
- Assumption: many OpenAlex 404s are expected due to stale/unresolvable IDs and should not be treated as immediately actionable runtime errors.

**Further Considerations**
1. Diagnostics naming: Option A use --report-404s for narrow control, Option B use --debug-http for broader transport diagnostics. Recommendation: Option A for least surprise.
2. Logging level policy: Option A demote 404 body logs to debug only, Option B keep warning logs but throttle duplicates. Decision: Option A, with response bodies shown only when debug-http is enabled.
3. Summary placement: Option A single end-of-run line, Option B include rolling periodic counts during long runs. Decision: Option B (periodic counts) plus a final end-of-run summary.
