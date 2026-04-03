## Plan: Internet Citation Ingestion Pipeline

Build a provider-agnostic citation retrieval pipeline that pulls from OpenAlex, Semantic Scholar, and Crossref, normalizes and deduplicates results into the existing `citation_data`/`papers_data` schema, caches progress locally for resumability, and preserves offline local-file mode. Default behavior uses configurable depth with L2+L3 enabled, while tests remain deterministic by mocking providers and using fixtures only.

**Steps**
1. Define scope contract and canonical data model.
   - Specify exactly what "theory inputs" mean: `theory_name`, `l1_papers`.
   - Formalize canonical output schemas already used by ADIT:
     - `citation_data`: `{citing_paper_id: [cited_paper_ids]}`
     - `papers_data`: `{paper_id: {title, abstract, keywords, citations, year}}`
   - Add explicit ID normalization rules (DOI-prefixed IDs, OpenAlex IDs, Semantic Scholar IDs) and provenance metadata for traceability. *blocks all later steps*
2. Introduce citation provider abstraction and orchestration layer.
   - Add a provider interface with methods for: search/resolve paper IDs, fetch citing papers, fetch references, fetch metadata.
   - Add a provider registry + capability flags (for example supports DOI lookup, supports reference expansion, supports citation counts) so new adapters can be plugged in without changing orchestrator logic.
   - Implement source adapters for OpenAlex, Semantic Scholar, and Crossref.
   - Implement an orchestrator that calls all enabled providers, merges responses, and emits canonical structures. *depends on 1*
3. Implement robust deduplication and entity resolution.
   - Build merge keys in order: DOI (preferred), normalized title+year fallback, provider-native IDs as last resort.
   - Collapse duplicates across sources while preserving per-source provenance and citation counts.
   - Prevent edge duplication in graph generation (same citing->cited edge from multiple sources counted once). *depends on 2*
4. Add depth-controlled graph expansion logic.
   - Implement `depth` parameter with default `L2+L3` and option to choose alternate depths.
   - L2 retrieval: papers citing any L1 paper.
   - L3 retrieval: references of L2 papers not already in L1/L2.
   - Add limits/guards: max papers per level, per-source timeout, retry/backoff. *depends on 2*
5. Add local cache/checkpoint system for resumable retrieval.
   - Cache raw provider responses and normalized intermediate artifacts keyed by query + provider + timestamp/version.
   - Persist checkpoints after each provider batch and each level (L2/L3) completion.
   - On rerun, reuse cache and skip completed work unless `--refresh` is requested. *parallel with step 4 after step 2*
6. Integrate retrieval mode into CLI/config while preserving offline mode.
   - Extend CLI/config with source selection, depth, cache dir, refresh toggle, and provider enable/disable flags.
   - Keep existing file-based path for `citation_data`/`papers_data` as offline fallback and deterministic replay mode.
   - Add input validation and explicit error messages for network failures and empty results. *depends on 3, 4, 5*
7. Thread optional key constructs into retrieval relevance pipeline.
   - Use `key_constructs` to refine search queries and post-filter noisy candidates.
   - Store construct-match indicators in normalized metadata so downstream feature extraction can use them.
   - Keep construct filtering optional and configurable to avoid over-pruning. *depends on 2*
8. Expand tests with deterministic fixtures only.
   - Unit tests for each adapter mapping function and ID normalization logic.
   - Unit tests for dedupe precedence and edge merge correctness.
   - Integration tests for orchestrator using mocked HTTP responses + cached fixtures (no live API in tests).
   - CLI tests for online-mode option parsing and offline fallback behavior. *depends on 6 and 7*
9. Verification and acceptance gates.
   - Validate generated ecosystem shape (L1/L2/L3 counts and edge integrity) against known fixture scenarios.
   - Validate resumability by interrupting a run mid-way and confirming successful resume without duplicate output.
   - Validate reproducibility by re-running from cache and comparing deterministic canonical outputs. *depends on 8*

**Relevant files**
- `/Users/adyn1/Documents/ADIT/adit.py` — reuse `build_ecosystem(citation_data)` contract and existing L1/L2/L3 semantics.
- `/Users/adyn1/Documents/ADIT/cli.py` — extend argument/config resolution for source selection, depth, cache, refresh, and offline fallback.
- `/Users/adyn1/Documents/ADIT/tests/test_cli.py` — add coverage for new retrieval flags and fallback mode selection.
- `/Users/adyn1/Documents/ADIT/tests/test_build_ecosystem.py` — keep graph-level expectations stable with canonical `citation_data` produced by orchestrator fixtures.
- `/Users/adyn1/Documents/ADIT/tests/conftest.py` — add normalized multi-source fixtures and mock provider responses.

**Verification**
1. Run adapter and normalization unit tests to confirm schema mapping and ID normalization behavior.
2. Run deduplication tests for cross-source duplicate collapse and stable edge outputs.
3. Run orchestrator integration tests with mocked provider payloads for `depth=L2` and `depth=L2+L3`.
4. Run CLI tests for online mode, cache reuse, refresh behavior, and offline local-file fallback.
5. Run end-to-end fixture test that generates `citation_data`/`papers_data` and passes them into existing `ADIT.build_ecosystem` + feature extraction path.

**Decisions**
- Source strategy: aggregate all three sources (OpenAlex + Semantic Scholar + Crossref), then deduplicate.
- Default depth: configurable depth with `L2+L3` as default.
- Operational constraints included: no API keys in v1, maintain offline local JSON fallback, cache/checkpoint by default.
- Included scope: ingestion/orchestration/caching/dedup/testing for citation ecosystem inputs.
- Excluded scope in v1: adding paid/proprietary connectors (for example Dimensions, Web of Science), live-network tests in CI, and replacing current ADIT graph semantics.
- Future-ready design requirement: architecture must allow adding new providers (including Dimensions/Web of Science) as drop-in adapters with no orchestrator rewrite.

**Further Considerations**
1. Recommended additional constraint: set a default per-run paper cap (for example, max L2 and L3 papers) to control runtime and API load; allow override in CLI/config.
2. Recommended additional constraint: add a run manifest file documenting provider versions, query parameters, and cache keys for reproducibility/auditability.
3. Recommended additional constraint: define a minimum metadata completeness threshold (for example title+year required) before admitting papers into canonical outputs.
4. Recommended future-source readiness task: define and document an adapter onboarding checklist (auth model, quota/rate-limit strategy, field mapping, confidence scoring, and legal terms constraints) for upcoming connectors such as Dimensions and Web of Science.
