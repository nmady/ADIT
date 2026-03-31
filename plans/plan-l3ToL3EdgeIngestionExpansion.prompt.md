## Plan: L3-to-L3 Edge Ingestion Expansion

Implement two expansions: (a) enable Crossref as a DOI-gated supplemental source in FIRST-pass L2→L3 reference discovery, and (b) add a SECOND-pass traversal that scans references of L3 papers, retaining only L3→L3 edges (no L4 materialization). Crossref participates in both passes. Keep output schema unchanged while adding metadata counters for observability, and reuse existing provider_l3_state checkpointing with phase markers.

**Steps**
1. Phase A: Contract lock from approved decisions.
2. Scope: keep only L3->L3 edges (no L4 persistence).
3. Budget policy: separate budget for second pass, leaving existing L2->L3 budget semantics unchanged.
4. Providers in rollout: OpenAlex, Semantic Scholar, CORE, and Crossref (supplemental only).
5. Output: existing citation_data graph plus new metadata counters.
6. Resume strategy: extend provider_l3_state with phase markers and second-pass cursor.
7. Phase B: Provider API and capability extension. depends on 1.
8. Add optional provider method for outgoing references from L3 parents with resume/progress callback compatibility.
9. Add capability flag for second-pass outgoing support, default false/no-op for backward compatibility.
10. Phase C: Provider implementations. depends on 8-9.
11. Implement outgoing reference retrieval for OpenAlex, Semantic Scholar, and CORE using existing normalization paths.
12. Implement Crossref DOI-based outgoing reference retrieval as best-effort supplemental source.
13. Ensure identifiers are normalized and unresolvable references are dropped.
14. Phase D: Orchestration second pass. depends on 11-13.
15. Build L3 membership set after existing L2->L3 pass.
16. Traverse each L3 parent via provider outgoing-reference methods.
17. Retain only edges where target is already in L3 membership set.
18. Merge retained L3->L3 edges into existing citation_data graph.
19. Phase E: Checkpoint/resume extension. depends on 15-18.
20. Reuse provider_l3_state with explicit phase markers (l2_to_l3, l3_to_l3).
21. Persist second-pass cursor (l3_parent_next_index) and relevant counters.
22. Apply existing stale-state and fail-open guards to second-pass resume payload.
23. Phase F: Metadata counters (output choice 4B). depends on 15-22.
24. Add counters such as l3_to_l3_edges_added, l3_to_l3_parent_scanned_count, and l3_to_l3_resumed_providers.
25. Surface counters in metadata.checkpoint_stats/provider_stats without changing top-level output schema.
26. Phase G: Testing. depends on 11-25.
27. Add provider unit tests for second-pass outgoing references and normalization.
28. Add orchestrator tests proving L3-only retention and explicit non-persistence of L4 nodes/edges.
29. Add crash-resume equivalence tests for second pass across OpenAlex/Semantic/CORE.
30. Add Crossref supplemental tests confirming additive behavior and non-authoritative coverage assumptions.
31. Add malformed second-pass resume-state tests with fail-open behavior.
31a. Add explicit acceptance test: Crossref references without DOI must not create new L3 nodes; DOI-backed Crossref references may create nodes with crossref provenance.
32. Phase H: Documentation. depends on 24-31.
33. Update README to define L3->L3 ingestion semantics and no-L4-materialization behavior.
34. Document Crossref as supplemental outgoing source and expected incompleteness caveat.
35. Document new metadata counters.

**Relevant files**
- /Users/adyn1/Documents/ADIT/citation_ingestion.py — provider abstraction, provider implementations, orchestration second pass, checkpoint state extension, metadata counters.
- /Users/adyn1/Documents/ADIT/tests/test_citation_ingestion.py — provider and orchestrator tests for retention, resume, and crash equivalence.
- /Users/adyn1/Documents/ADIT/tests/test_integration.py — integration coverage for second-pass crash-resume compatibility.
- /Users/adyn1/Documents/ADIT/README.md — behavior, provider caveats, and metadata documentation.

**Verification**
1. /Users/adyn1/Documents/ADIT/.venv/bin/python -m pytest tests/test_citation_ingestion.py -k "l3_to_l3 or second_pass" -q.
2. /Users/adyn1/Documents/ADIT/.venv/bin/python -m pytest tests/test_citation_ingestion.py -k "resume and l3" -q.
3. /Users/adyn1/Documents/ADIT/.venv/bin/python -m pytest tests/test_citation_ingestion.py -k "crossref and outgoing" -q.
4. /Users/adyn1/Documents/ADIT/.venv/bin/python -m pytest tests/test_integration.py -k "checkpoint" -q.
5. /Users/adyn1/Documents/ADIT/.venv/bin/python -m pytest tests/test_citation_ingestion.py -q.
6. /Users/adyn1/Documents/ADIT/.venv/bin/python -m pytest tests/test_cli.py -q.

**Decisions**
- 1A: retain only L3->L3 edges; do not persist L4 frontier.
- 2A: use a separate budget policy for second pass.
- 3: rollout providers are OpenAlex + Semantic Scholar + CORE + Crossref.
- 4B: keep same citation_data graph and add metadata counters.
- 5A: reuse provider_l3_state with phase markers/cursor for second-pass resume.
- Crossref L3 node creation policy: supplemental-only, DOI-gated for node creation, provenance-tagged, and not used as a completeness authority.

**Further Considerations**
1. If second-pass runtime is high, add optional toggle to disable L3->L3 augmentation for speed-sensitive runs.
2. If Crossref DOI sparsity limits value, consider provider weighting in second pass for request efficiency.
3. If state complexity grows, split second-pass state into dedicated block in a future schema bump.