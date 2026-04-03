## Plan: Transient Checkpoint Retry On Resume

Persist provider-scoped transient failure records in checkpoints, keep a compact coordinator summary for visibility, and replay eligible failures on resume using backoff + Retry-After semantics. This keeps retries precise, resumable, and rate-limit aware without storing sensitive request data.

**Current status**
1. Implemented: schema/state extensions for transient failure persistence in provider/global checkpoints.
2. Implemented: transient failure capture from retry-exhausted request paths.
3. Implemented: provider-level transient replay during resume for wave-1 and L2->L3 paths.
4. Implemented: metadata/checkpoint stats counters and transient summary exposure.
5. Implemented: CLI/config knobs for transient retry max attempts and max age.
6. Implemented: unit + integration coverage for checkpoint resume and transient replay scenarios.
7. Remaining: optional hardening and observability follow-ups listed below.

**Steps**
1. Phase 1: Extend checkpoint schema for transient retry state. *completed*
2. Bump provider/coordinator/global checkpoint schema versions and preserve backward-compatible loads for prior versions. *completed*
3. Add provider-level `transient_failures` payload and coordinator/global compact summary fields. *completed*
4. Define canonical transient failure record shape and serializer/deserializer helpers. *completed*
5. Phase 2: Capture transient failures at execution boundaries. *completed*
6. Capture retry-exhausted transient failures with stable operation identity, error metadata, and minimal replay state. *completed*
7. Ensure atomic persistence after state mutations using existing checkpoint write path. *completed*
8. Add pruning policy for stale/terminal transient entries and track in checkpoint stats. *completed*
9. Phase 3: Replay transient failures on resume. *completed (initial)*
10. During checkpoint restore, compute retry eligibility from max(Retry-After, exponential backoff schedule). *completed*
11. Replay eligible entries first, remove on success, increment/reschedule on transient failure, prune on terminal failure. *completed*
12. Guard against duplicate replay per provider scope in sequential/parallel execution paths. *completed (initial)*
13. Phase 4: Metadata and observability. *completed (initial)*
14. Extend metadata/checkpoint stats with transient totals (queued, retried, succeeded-on-resume, exhausted, pruned). *completed*
15. Surface compact transient summary in final metadata for run-level visibility. *completed*
16. Phase 5: Tests and regression checks. *completed (initial)*
17. Add/extend unit tests for checkpoint compatibility, transient serialization, scheduling, dedupe, and pruning behavior. *completed (core cases)*
18. Add integration tests verifying checkpoint persistence and resumed retry convergence to expected outputs. *completed (core cases)*
19. Run targeted and integration test subsets to confirm no regression in checkpoint/resume behavior. *completed*

**Remaining follow-ups (optional next phase)**
1. Record active transient retry policy values (`transient_retry_max_attempts`, `transient_retry_max_age_seconds`) in final metadata/checkpoint stats for auditability.
2. Add a cross-provider parallel resume canary to stress concurrent transient replay behavior.
3. Add optional checkpoint compaction/rotation policy for long-running exhaustive jobs.

**Relevant files**
- `/Users/adyn1/Documents/ADIT/citation_ingestion.py` - schema constants, checkpoint read/write, transient capture/replay helpers, provider workers, metadata assembly.
- `/Users/adyn1/Documents/ADIT/cli.py` - CLI/config option resolution and ingestion parameter forwarding.
- `/Users/adyn1/Documents/ADIT/tests/test_citation_ingestion.py` - unit and resume-flow tests for transient checkpointing/replay.
- `/Users/adyn1/Documents/ADIT/tests/test_cli.py` - option forwarding and validation tests for transient retry knobs.
- `/Users/adyn1/Documents/ADIT/tests/test_integration.py` - checkpoint resume integration coverage.

**Verification**
1. Run transient-focused citation ingestion tests.
2. Run CLI option validation/forwarding tests.
3. Run checkpoint resume integration tests.
4. (Optional) Run full ingestion suite before release cut.

**Decisions**
- Included: provider-level detailed transient failure persistence plus coordinator/global compact summaries.
- Included transient classes: HTTP 429, HTTP 5xx, and network/timeout failures; permanent HTTP 400/401/403/404 remains non-retryable.
- Included retry policy: backoff on resume honoring Retry-After with capped attempts and age pruning.
- Included record detail: operation identity + minimal replay cursor/state + retry bookkeeping; no API keys or auth headers persisted.
- Excluded: replaying arbitrary raw HTTP payloads from checkpoint files.
- Excluded: changing core provider retrieval semantics for successful paths.
