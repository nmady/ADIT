## Plan: Exhaustive L1 Citer Coverage

Guarantee practical completeness by switching L2 discovery from keyword search to per-L1 cited-by traversal with full pagination, strict provider reconciliation, and completeness audits. Keep current merge/dedupe pipeline, but make retrieval exhaustive and measurable.

**Steps**
1. Define completeness contract and constraints.
   - Coverage target: include every paper that cites each L1 and is exposed by provider APIs at crawl time.
   - Explicitly exclude provider-internal records not returned by APIs, transient outages after retry budget exhaustion, and permanently rate-limited windows.
2. Replace keyword-driven L2 discovery with identifier-driven cited-by crawling.
   - OpenAlex: query by `referenced_works` against each normalized L1 OpenAlex ID and paginate with cursor until exhaustion.
   - Semantic Scholar: use paper-specific citations endpoint for each L1 (with pagination) instead of global search query.
   - Crossref: keep as metadata enrichment only unless a reliable cited-by API path is implemented.
3. Build a robust L1 identifier resolution layer before crawl.
   - For each L1 seed, resolve and store provider IDs (DOI/OpenAlex/S2) via deterministic lookup.
   - Persist provider-id mapping to avoid repeated resolution and reduce misses from ID format drift.
4. Implement exhaustive pagination and continuation state.
   - Iterate all pages per L1/provider (cursor/offset/token).
   - Track continuation checkpoints so interrupted runs can resume without restarting full crawl.
   - Remove hard truncation behavior as default (`max_l2`) or move it behind an explicit non-exhaustive mode flag.
5. Harden networking and provider quota behavior.
   - Add retry policy for timeout/429/5xx with exponential backoff + jitter.
   - Respect provider-specific pacing; centralize throttle controls.
   - Record terminal failures with reason codes per L1/provider/page.
6. Add per-L1 completeness auditing in metadata.
   - Emit fetched citer counts per L1/provider.
   - Emit expected counts when provider exposes them (e.g., cited-by count fields).
   - Emit completion status (`complete`, `partial`, `failed`) plus residual gap estimate.
7. Tighten filtering rules to avoid false negatives.
   - Remove title/theory-name gate from primary L2 inclusion path.
   - Include any paper returned by cited-by traversal as valid L2.
8. Extend tests for recall-critical behavior.
   - Unit tests for multi-page traversal, continuation resume, retry on transient failures, and per-L1 completeness accounting.
   - Integration-style mocked provider tests asserting no page loss and deterministic dedupe of cross-provider duplicates.
9. Add operational monitoring and runbook.
   - Surface crawl summary (pages fetched, retries, unresolved IDs, partial L1s).
   - Add alert thresholds for non-zero partial/failed L1 counts.

**Relevant files**
- `citation_ingestion.py` — provider crawl strategy, pagination, retries, metadata auditing.
- `cli.py` — exhaustive mode flags and crawl-summary output.
- `tests/test_citation_ingestion.py` — paging/retry/completeness tests.
- `README.md` — completeness contract, limits, and operational guidance.

**Verification**
1. Mocked pagination test: each provider returns >2 pages; assert all expected citer IDs appear in final `citation_data`.
2. Retry test: inject 429/timeout for intermediate pages; assert eventual success and no missing pages when retries succeed.
3. Partial-failure test: force unrecoverable page failure; assert metadata marks affected L1/provider as `partial` with accurate gap counters.
4. Determinism test: same crawl inputs with cache refresh disabled/enabled produce identical normalized edge sets.
5. Manual smoke run in online mode for one theory and compare fetched per-L1 counts vs provider-reported counts.

**Decisions**
- Primary L2 recall must be ID-based cited-by traversal, not keyword search.
- `max_l2` should not silently cap exhaustive runs; retain only as explicit sampling mode.
- Crossref remains non-authoritative for cited-by edges unless a reliable path is implemented.

**Further Considerations**
1. Introduce an `--exhaustive` flag defaulting to true in online mode, with `--sample` for capped retrieval.
2. Add persistent checkpoint storage per run to recover long crawls after interruption.
3. Consider adding one more citation source with robust cited-by APIs to reduce single-provider blind spots.
