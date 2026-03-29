## Plan: Ingestion Progress Messaging

Add always-visible ingestion progress updates so users can see that online ingestion is advancing through providers, L1 seeds, and major post-fetch phases. Keep the existing `--verbose` mode for lower-level diagnostics like retry countdowns and provider-specific detail, and introduce a small set of standard progress messages for normal runs.

**Steps**
1. Define the user-visible progress contract in `/Users/adyn1/Documents/ADIT/citation_ingestion.py`: identify the major milestones that should always print during online ingestion, and separate them from the existing verbose-only messages. This blocks the remaining steps.
2. Add a dedicated progress-output helper alongside the current verbose helpers in `/Users/adyn1/Documents/ADIT/citation_ingestion.py`. The helper should emit permanent stderr lines for normal progress, while `_vprint()` remains reserved for detailed tracing and retry internals.
3. Update `ingest_from_internet()` in `/Users/adyn1/Documents/ADIT/citation_ingestion.py` to report top-level milestones: ingestion start, provider `i/n`, cache hit vs live fetch, dedup/materialization start, metadata assembly, cache write, and final completion summary. This depends on step 2.
4. Update `_fetch_provider_graph()` in `/Users/adyn1/Documents/ADIT/citation_ingestion.py` to report provider-level progress: current seed `i/n`, cited-by traversal completion per L1, L2 candidate count after filtering, and L3 hydration progress when `depth=l2l3`. Keep page-by-page and retry chatter in verbose-only paths. This depends on step 2 and can run in parallel with step 3 once the helper contract is defined.
5. Decide whether cache short-circuiting should emit a normal progress line in `_load_cached_result()` or immediately after cache lookup in `ingest_from_internet()`. Recommended: emit it from `ingest_from_internet()` so user-visible flow stays centralized. This depends on step 3.
6. Update CLI-facing expectations in `/Users/adyn1/Documents/ADIT/cli.py` only if needed to avoid duplicate completion messages or to align wording between ingestion progress and the existing final CLI summary. This depends on steps 3 and 4.
7. Extend `/Users/adyn1/Documents/ADIT/tests/test_citation_ingestion.py` and `/Users/adyn1/Documents/ADIT/tests/test_cli.py` to capture stderr/stdout and assert that key progress markers appear for live ingestion, cache hits, and completion. This depends on steps 3 through 6.
8. Update `/Users/adyn1/Documents/ADIT/README.md` if the behavior of `--verbose` changes materially, clarifying that standard progress is always shown and verbose mode adds more detailed diagnostics. This depends on the final messaging shape.

**Relevant files**
- `/Users/adyn1/Documents/ADIT/citation_ingestion.py` — add normal progress helper; adjust `ingest_from_internet()` and `_fetch_provider_graph()`; preserve `_vprint()` for detailed tracing.
- `/Users/adyn1/Documents/ADIT/cli.py` — confirm the ingestion progress output and final CLI summary are complementary rather than redundant.
- `/Users/adyn1/Documents/ADIT/tests/test_citation_ingestion.py` — add assertions for emitted progress lines during ingestion and cache reuse.
- `/Users/adyn1/Documents/ADIT/tests/test_cli.py` — verify CLI output remains coherent when ingestion now emits progress automatically.
- `/Users/adyn1/Documents/ADIT/README.md` — document the new default progress behavior if necessary.

**Verification**
1. Run the targeted ingestion tests in `/Users/adyn1/Documents/ADIT/tests/test_citation_ingestion.py` and `/Users/adyn1/Documents/ADIT/tests/test_cli.py`, including cases that exercise live fetch flow and cache-hit flow.
2. Run the full test suite to catch output-capture regressions in integration paths.
3. Run a real CLI command such as `python cli.py --config examples/sdt.config.yml` and confirm the terminal shows clear progress transitions without excessive noise.
4. Confirm that detailed retry countdowns and similar low-level diagnostics still require `--verbose`, while high-level progress is visible by default.

**Decisions**
- Progress should be visible by default during online ingestion.
- `--verbose` should remain the switch for detailed, high-frequency diagnostics.
- Default messaging should be milestone-style, not per-page/per-request chatter.
- Scope is limited to online ingestion visibility; no progress bar, concurrency changes, or logging-framework migration.

**Further Considerations**
1. Recommended granularity is provider/seed/phase milestones rather than page-level pagination, so large ingestions stay readable.
2. Recommended output stream is stderr for progress, preserving stdout for the final CLI summary and any redirected outputs.
3. Recommended cache behavior is an explicit cache-hit message so fast returns are distinguishable from silent reuse.
