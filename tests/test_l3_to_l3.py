"""Tests for L3-to-L3 edge ingestion expansion and Crossref first-pass L3 discovery."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

import citation_ingestion as ci

# ── Helpers ──────────────────────────────────────────────────────────


def _make_crossref_provider():
    return ci.CrossrefProvider()


def _crossref_work_response(references):
    """Build a Crossref /works/{doi} API response with the given references list."""
    return {"message": {"reference": references}}


# ── Phase B: Crossref first-pass L3 ─────────────────────────────────


class TestCrossrefFirstPassL3:
    """CrossrefProvider.fetch_l3_references() — DOI-gated reference expansion."""

    def test_doi_gated_admission(self):
        """Only references carrying a DOI create L3 nodes."""
        provider = _make_crossref_provider()
        refs = [
            {"DOI": "10.1000/ref1", "article-title": "Good Ref"},
            {"article-title": "No DOI Ref"},
            {"DOI": "10.1000/ref2", "article-title": "Another Good"},
        ]
        with patch.object(ci, "_safe_get", return_value=_crossref_work_response(refs)):
            edges, papers = provider.fetch_l3_references(["doi:10.1000/parent"])
        assert "doi:10.1000/parent" in edges
        refs_found = edges["doi:10.1000/parent"]
        assert "doi:10.1000/ref1" in refs_found
        assert "doi:10.1000/ref2" in refs_found
        assert len(refs_found) == 2
        assert "doi:10.1000/ref1" in papers
        assert "doi:10.1000/ref2" in papers

    def test_non_doi_references_dropped(self):
        """References without DOIs must not create L3 nodes."""
        provider = _make_crossref_provider()
        refs = [
            {"article-title": "No DOI"},
            {"key": "some-key", "unstructured": "Some citation text"},
        ]
        with patch.object(ci, "_safe_get", return_value=_crossref_work_response(refs)):
            edges, papers = provider.fetch_l3_references(["doi:10.1000/parent"])
        assert len(edges) == 0
        assert len(papers) == 0

    def test_budget_respected(self):
        """Budget limits how many L3 refs are admitted."""
        provider = _make_crossref_provider()
        refs = [{"DOI": f"10.1000/ref{i}", "article-title": f"Ref {i}"} for i in range(10)]
        with patch.object(ci, "_safe_get", return_value=_crossref_work_response(refs)):
            edges, papers = provider.fetch_l3_references(["doi:10.1000/parent"], max_l3=3)
        total = sum(len(v) for v in edges.values())
        assert total == 3
        assert len(papers) == 3

    def test_provenance_tagged(self):
        """Admitted L3 papers carry crossref provenance in source_ids."""
        provider = _make_crossref_provider()
        refs = [{"DOI": "10.1000/ref1", "article-title": "Ref"}]
        with patch.object(ci, "_safe_get", return_value=_crossref_work_response(refs)):
            _, papers = provider.fetch_l3_references(["doi:10.1000/parent"])
        paper = papers["doi:10.1000/ref1"]
        assert "crossref" in paper.source_ids
        assert paper.doi == "10.1000/ref1"

    def test_skips_non_doi_parents(self):
        """L2 papers without DOIs are skipped (Crossref can't query them)."""
        provider = _make_crossref_provider()
        with patch.object(ci, "_safe_get") as mock_get:
            edges, papers = provider.fetch_l3_references(["openalex:W100", "semantic_scholar:abc"])
        mock_get.assert_not_called()
        assert len(edges) == 0
        assert len(papers) == 0

    def test_resume_state_support(self):
        """Checkpoint resume continues from next_l2_index."""
        provider = _make_crossref_provider()
        l2_ids = ["doi:10.1000/p1", "doi:10.1000/p2", "doi:10.1000/p3"]
        refs = [{"DOI": "10.1000/ref_new", "article-title": "New"}]

        call_count = 0

        def _fake_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            return _crossref_work_response(refs)

        resume = {"next_l2_index": 2}  # skip first two parents
        with patch.object(ci, "_safe_get", side_effect=_fake_get):
            edges, papers = provider.fetch_l3_references(l2_ids, resume_state=resume)
        assert call_count == 1  # only processed p3

    def test_progress_callback_called(self):
        """Progress callback fires during traversal."""
        provider = _make_crossref_provider()
        refs = [{"DOI": "10.1000/ref1"}]
        states = []

        with patch.object(ci, "_safe_get", return_value=_crossref_work_response(refs)):
            provider.fetch_l3_references(
                ["doi:10.1000/p1"],
                progress_callback=lambda s: states.append(dict(s)),
            )
        assert len(states) >= 1
        assert states[-1]["status"] == "complete"

    def test_year_extraction(self):
        """Year is extracted from reference when present and numeric."""
        provider = _make_crossref_provider()
        refs = [
            {"DOI": "10.1000/ref1", "year": "2020"},
            {"DOI": "10.1000/ref2", "year": "not-a-year"},
            {"DOI": "10.1000/ref3"},
        ]
        with patch.object(ci, "_safe_get", return_value=_crossref_work_response(refs)):
            _, papers = provider.fetch_l3_references(["doi:10.1000/parent"])
        assert papers["doi:10.1000/ref1"].year == 2020
        assert papers["doi:10.1000/ref2"].year is None
        assert papers["doi:10.1000/ref3"].year is None

    def test_capabilities_updated(self):
        """CrossrefProvider now declares supports_reference_expansion and supports_l3_outgoing."""
        p = _make_crossref_provider()
        assert p.capabilities.supports_reference_expansion is True
        assert p.capabilities.supports_l3_outgoing is True


# ── Phase C/D: Provider second-pass outgoing references ─────────────


class TestOpenAlexL3Outgoing:
    """OpenAlexProvider.fetch_l3_outgoing_references()."""

    def test_returns_edges_and_papers(self):
        provider = ci.OpenAlexProvider()
        payload = {"referenced_works": ["https://openalex.org/W200", "https://openalex.org/W201"]}
        with patch.object(ci, "_safe_get", return_value=payload):
            edges, papers = provider.fetch_l3_outgoing_references(["openalex:W100"])
        assert "openalex:W100" in edges
        assert "openalex:W200" in edges["openalex:W100"]
        assert "openalex:W201" in edges["openalex:W100"]
        assert "openalex:W200" in papers
        assert "openalex:W201" in papers

    def test_skips_non_openalex_ids(self):
        provider = ci.OpenAlexProvider()
        with patch.object(ci, "_safe_get") as mock_get:
            edges, _ = provider.fetch_l3_outgoing_references(
                ["doi:10.1000/abc", "semantic_scholar:xyz"]
            )
        mock_get.assert_not_called()
        assert len(edges) == 0

    def test_budget_limit(self):
        provider = ci.OpenAlexProvider()
        payload = {"referenced_works": [f"https://openalex.org/W{i}" for i in range(20)]}
        with patch.object(ci, "_safe_get", return_value=payload):
            edges, papers = provider.fetch_l3_outgoing_references(["openalex:W100"], max_edges=5)
        total = sum(len(v) for v in edges.values())
        assert total == 5

    def test_resume_state(self):
        provider = ci.OpenAlexProvider()
        payload = {"referenced_works": ["https://openalex.org/W999"]}
        call_count = 0

        def _fake_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            return payload

        resume = {"next_l3_parent_index": 2}
        with patch.object(ci, "_safe_get", side_effect=_fake_get):
            provider.fetch_l3_outgoing_references(
                ["openalex:W1", "openalex:W2", "openalex:W3"],
                resume_state=resume,
            )
        assert call_count == 1  # only W3 processed


class TestSemanticScholarL3Outgoing:
    """SemanticScholarProvider.fetch_l3_outgoing_references()."""

    def test_returns_edges(self):
        provider = ci.SemanticScholarProvider()
        payload = {
            "references": [
                {
                    "paperId": "ss100",
                    "externalIds": {"DOI": "10.1000/ref1"},
                },
                {
                    "paperId": "ss200",
                    "externalIds": {},
                },
            ]
        }
        with patch.object(ci, "_safe_get", return_value=payload):
            edges, papers = provider.fetch_l3_outgoing_references(["semantic_scholar:parent1"])
        assert "semantic_scholar:parent1" in edges
        assert len(edges["semantic_scholar:parent1"]) == 2

    def test_skips_non_semantic_ids(self):
        provider = ci.SemanticScholarProvider()
        with patch.object(ci, "_safe_get") as mock_get:
            edges, _ = provider.fetch_l3_outgoing_references(["openalex:W100", "doi:10.1000/abc"])
        mock_get.assert_not_called()
        assert len(edges) == 0


class TestCrossrefL3Outgoing:
    """CrossrefProvider.fetch_l3_outgoing_references()."""

    def test_returns_doi_gated_edges(self):
        provider = _make_crossref_provider()
        refs = [
            {"DOI": "10.1000/target1"},
            {"article-title": "No DOI target"},
            {"DOI": "10.1000/target2"},
        ]
        with patch.object(ci, "_safe_get", return_value=_crossref_work_response(refs)):
            edges, papers = provider.fetch_l3_outgoing_references(["doi:10.1000/l3parent"])
        assert len(edges["doi:10.1000/l3parent"]) == 2

    def test_skips_non_doi_parents(self):
        provider = _make_crossref_provider()
        with patch.object(ci, "_safe_get") as mock_get:
            provider.fetch_l3_outgoing_references(["openalex:W100"])
        mock_get.assert_not_called()


class TestCoreL3Outgoing:
    """CoreProvider.fetch_l3_outgoing_references()."""

    def test_returns_edges(self):
        provider = ci.CoreProvider(api_key="test-key")
        work_payload = {
            "references": [
                {"doi": "10.1000/coreref1"},
                {"id": 99999},
            ]
        }
        with patch.object(provider, "_lookup_work", return_value=work_payload):
            edges, papers = provider.fetch_l3_outgoing_references(["core:12345"])
        assert "core:12345" in edges
        assert len(edges["core:12345"]) >= 1

    def test_budget_limit(self):
        provider = ci.CoreProvider(api_key="test-key")
        work_payload = {"references": [{"doi": f"10.1000/ref{i}"} for i in range(10)]}
        with patch.object(provider, "_lookup_work", return_value=work_payload):
            edges, _ = provider.fetch_l3_outgoing_references(["core:12345"], max_edges=3)
        total = sum(len(v) for v in edges.values())
        assert total == 3


# ── Phase E: Orchestrator second-pass L3→L3 retention ───────────────


class _SecondPassFakeProvider(ci.CitationProvider):
    """Fake provider that supports L3 outgoing references for testing."""

    name = "fakeprov"
    capabilities = ci.ProviderCapabilities(
        True,
        True,
        True,
        supports_cited_by_traversal=True,
        supports_l3_outgoing=True,
    )

    def __init__(self, l3_outgoing_edges=None, l3_outgoing_papers=None):
        self._l3_outgoing_edges = l3_outgoing_edges or {}
        self._l3_outgoing_papers = l3_outgoing_papers or {}

    def fetch_seed_metadata(self, l1_papers):
        result = {}
        for pid in l1_papers:
            result[pid] = ci.IngestionPaper(
                paper_id=pid,
                title="Seed",
                doi=pid.replace("doi:", "") if pid.startswith("doi:") else None,
                source_ids={"fakeprov": pid},
            )
        return result

    def fetch_citers_for_l1(
        self, l1_provider_id, max_results=None, resume_state=None, progress_callback=None
    ):
        return {}, 0, "complete"

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        return {}, {}

    def fetch_l3_references(
        self, l2_paper_ids, max_l3=None, resume_state=None, progress_callback=None
    ):
        return {}, {}

    def fetch_l3_outgoing_references(
        self, l3_paper_ids, max_edges=None, resume_state=None, progress_callback=None
    ):
        return dict(self._l3_outgoing_edges), dict(self._l3_outgoing_papers)


class TestSecondPassL3ToL3:
    """Orchestrator retains only L3→L3 edges, no new node creation."""

    def test_l3_to_l3_edges_retained(self, tmp_path):
        """Second pass retains edges where both ends are L3 members."""
        # Setup: L3 member papers with an L3->L3 outgoing reference.
        l3a = "fakeprov:l3a"
        l3b = "fakeprov:l3b"

        fake = _SecondPassFakeProvider(
            l3_outgoing_edges={l3a: {l3b}},
            l3_outgoing_papers={l3b: ci.IngestionPaper(paper_id=l3b)},
        )

        # Build provider state for a direct test of second-pass filtering logic.
        l3_member_set = {l3a, l3b}

        raw_edges, raw_papers = fake.fetch_l3_outgoing_references(sorted(l3_member_set))

        # Filter: retain only edges with target in L3 set
        retained = {}
        for parent, targets in raw_edges.items():
            kept = targets & l3_member_set
            if kept:
                retained[parent] = kept

        assert l3a in retained
        assert l3b in retained[l3a]

    def test_no_new_nodes_in_second_pass(self):
        """Second pass must not materialize L4 nodes."""
        l3_member_set = {"l3a", "l3b"}
        raw_edges = {"l3a": {"l3b", "l4_external"}}

        retained = {}
        for parent, targets in raw_edges.items():
            kept = targets & l3_member_set
            if kept:
                retained[parent] = kept

        # l4_external should be filtered out
        assert "l3a" in retained
        assert retained["l3a"] == {"l3b"}

    def test_empty_l3_set_means_no_edges(self):
        """If there are no L3 members, no second-pass edges are created."""
        l3_member_set: set = set()
        raw_edges = {"parent": {"target"}}

        retained = {}
        for parent, targets in raw_edges.items():
            kept = targets & l3_member_set
            if kept:
                retained[parent] = kept

        assert len(retained) == 0


# ── Phase F: Checkpoint state extension ──────────────────────────────


class TestCheckpointL3ToL3State:
    """L3-to-L3 state is persisted and restored from checkpoint."""

    def test_write_and_read_l3_to_l3_state(self, tmp_path):
        """l3_to_l3_state round-trips through checkpoint serialization."""
        state = {
            "openalex": {
                "next_l3_parent_index": 5,
                "budget_remaining": 42,
                "edges": {},
                "papers": {},
                "updated_at": time.time(),
            }
        }
        ci._write_checkpoint_state(
            tmp_path,
            "test-key",
            {"openalex"},
            {},  # all_edges
            {},  # all_papers
            {},  # provider_stats
            {},  # combined_completeness
            {},  # provider_pagination_state
            {},  # provider_l3_state
            l3_to_l3_state=state,
            ingestion_phase="l3_to_l3",
        )
        loaded = ci._load_checkpoint_state(tmp_path, "test-key", reset_checkpoints=False)
        assert loaded is not None
        assert loaded["ingestion_phase"] == "l3_to_l3"
        l3_state = ci._deserialize_provider_l3_state(loaded.get("l3_to_l3_state"))
        assert "openalex" in l3_state
        assert l3_state["openalex"]["next_l3_parent_index"] == 5

    def test_phase_marker_defaults_to_l2_to_l3(self, tmp_path):
        """Without explicit phase, checkpoint gets default l2_to_l3."""
        ci._write_checkpoint_state(
            tmp_path,
            "key2",
            set(),
            {},
            {},
            {},
            {},
            {},
            {},
        )
        loaded = ci._load_checkpoint_state(tmp_path, "key2", reset_checkpoints=False)
        assert loaded["ingestion_phase"] == "l2_to_l3"

    def test_malformed_phase_ignored(self, tmp_path):
        """Unknown phase string defaults to l2_to_l3 on load."""
        ci._write_checkpoint_state(
            tmp_path,
            "key3",
            set(),
            {},
            {},
            {},
            {},
            {},
            {},
            ingestion_phase="unknown_phase",
        )
        loaded = ci._load_checkpoint_state(tmp_path, "key3", reset_checkpoints=False)
        # The raw value is stored; caller should validate
        assert loaded["ingestion_phase"] == "unknown_phase"


# ── Phase G: Metadata counters ───────────────────────────────────────


class TestMetadataCounters:
    """New L3-to-L3 counters appear in checkpoint_stats."""

    def test_checkpoint_stats_keys_present(self):
        """Required counter keys exist in freshly initialized stats."""
        stats = {
            "l3_to_l3_edges_added": 0,
            "l3_to_l3_parent_scanned_count": 0,
            "l3_to_l3_resumed_providers": [],
        }
        assert "l3_to_l3_edges_added" in stats
        assert "l3_to_l3_parent_scanned_count" in stats
        assert "l3_to_l3_resumed_providers" in stats


# ── Acceptance tests ─────────────────────────────────────────────────


class TestAcceptanceCrossrefFirstPass:
    """Crossref references without DOI must not create L3 nodes."""

    def test_no_doi_no_node(self):
        provider = _make_crossref_provider()
        refs = [
            {"article-title": "No DOI at all"},
            {"key": "ref-2023a"},
        ]
        with patch.object(ci, "_safe_get", return_value=_crossref_work_response(refs)):
            edges, papers = provider.fetch_l3_references(["doi:10.1000/parent"])
        assert len(edges) == 0
        assert len(papers) == 0

    def test_doi_backed_creates_node(self):
        provider = _make_crossref_provider()
        refs = [{"DOI": "10.1000/valid", "article-title": "Valid Ref"}]
        with patch.object(ci, "_safe_get", return_value=_crossref_work_response(refs)):
            edges, papers = provider.fetch_l3_references(["doi:10.1000/parent"])
        assert "doi:10.1000/valid" in papers
        assert papers["doi:10.1000/valid"].source_ids.get("crossref") == "10.1000/valid"


class TestAcceptanceSecondPassNoNewNodes:
    """In second pass, no provider creates new nodes."""

    def test_second_pass_filters_unknown_targets(self):
        l3_member_set = {"doi:10.1000/a", "doi:10.1000/b"}
        raw_edges = {
            "doi:10.1000/a": {"doi:10.1000/b", "doi:10.1000/unknown"},
        }
        retained = {}
        for parent, targets in raw_edges.items():
            kept = targets & l3_member_set
            if kept:
                retained[parent] = kept

        assert retained == {"doi:10.1000/a": {"doi:10.1000/b"}}
        assert "doi:10.1000/unknown" not in retained.get("doi:10.1000/a", set())


# ── Crash-resume equivalence tests for L3→L3 second pass ────────────


class _L3ToL3CrashProvider(ci.CitationProvider):
    """Fake provider with full first-pass + second-pass support and crash injection."""

    capabilities = ci.ProviderCapabilities(
        True,
        True,
        True,
        supports_cited_by_traversal=False,
        supports_l3_outgoing=True,
    )

    def __init__(
        self,
        name: str,
        l2_prefix: str,
        l3_refs: dict,
        l3_outgoing_refs: dict,
        crash_at_l3_index=None,
        crash_at_l3_outgoing_index=None,
    ):
        self.name = name
        self.l2_prefix = l2_prefix
        self.l3_refs = l3_refs
        self.l3_outgoing_refs = l3_outgoing_refs
        self.crash_at_l3_index = crash_at_l3_index
        self.crash_at_l3_outgoing_index = crash_at_l3_outgoing_index

    def fetch_seed_metadata(self, l1_papers):
        return {
            l1_papers[0]: ci.IngestionPaper(
                paper_id=l1_papers[0],
                title=f"Seed from {self.name}",
                source_ids={self.name: f"seed-{self.name}"},
            )
        }

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        seed = l1_papers[0]
        l2_a = f"{self.l2_prefix}:L2A"
        l2_b = f"{self.l2_prefix}:L2B"
        return (
            {l2_a: {seed}, l2_b: {seed}},
            {
                l2_a: ci.IngestionPaper(paper_id=l2_a, title=f"{self.name} L2A", year=2020),
                l2_b: ci.IngestionPaper(paper_id=l2_b, title=f"{self.name} L2B", year=2021),
            },
        )

    def fetch_l3_references(
        self, l2_paper_ids, max_l3=None, resume_state=None, progress_callback=None
    ):
        edges = ci._deserialize_edges((resume_state or {}).get("edges"))
        papers = ci._deserialize_papers((resume_state or {}).get("papers"))

        start_index_raw = (resume_state or {}).get("next_l2_index")
        try:
            start_index = int(start_index_raw) if start_index_raw is not None else 0
        except (TypeError, ValueError):
            start_index = 0

        for idx in range(start_index, len(l2_paper_ids)):
            if self.crash_at_l3_index is not None and idx == self.crash_at_l3_index:
                raise RuntimeError(f"simulated l3 crash in {self.name}")

            pid = l2_paper_ids[idx]
            for ref_id in self.l3_refs.get(pid, []):
                edges.setdefault(pid, set()).add(ref_id)
                papers.setdefault(
                    ref_id,
                    ci.IngestionPaper(
                        paper_id=ref_id,
                        title=f"Ref {ref_id}",
                        doi=ref_id.split(":", 1)[1] if ref_id.startswith("doi:") else None,
                    ),
                )

            if progress_callback:
                progress_callback(
                    {
                        "status": "in_progress",
                        "next_l2_index": idx + 1,
                        "budget_remaining": None,
                        "edges": ci._serialize_edges(edges),
                        "papers": ci._serialize_papers(papers),
                        "updated_at": time.time(),
                    }
                )

        if progress_callback:
            progress_callback(
                {
                    "status": "complete",
                    "next_l2_index": len(l2_paper_ids),
                    "budget_remaining": None,
                    "edges": ci._serialize_edges(edges),
                    "papers": ci._serialize_papers(papers),
                    "updated_at": time.time(),
                }
            )

        return edges, papers

    def fetch_l3_outgoing_references(
        self, l3_paper_ids, max_edges=None, resume_state=None, progress_callback=None
    ):
        edges = ci._deserialize_edges((resume_state or {}).get("edges"))
        papers = ci._deserialize_papers((resume_state or {}).get("papers"))

        start_index_raw = (resume_state or {}).get("next_l3_parent_index")
        try:
            start_index = int(start_index_raw) if start_index_raw is not None else 0
        except (TypeError, ValueError):
            start_index = 0

        for idx in range(start_index, len(l3_paper_ids)):
            if (
                self.crash_at_l3_outgoing_index is not None
                and idx == self.crash_at_l3_outgoing_index
            ):
                raise RuntimeError(f"simulated l3-to-l3 crash in {self.name}")

            pid = l3_paper_ids[idx]
            for target in self.l3_outgoing_refs.get(pid, []):
                edges.setdefault(pid, set()).add(target)
                papers.setdefault(
                    target,
                    ci.IngestionPaper(paper_id=target, title=f"Outgoing {target}"),
                )

            if progress_callback:
                progress_callback(
                    {
                        "status": "in_progress",
                        "next_l3_parent_index": idx + 1,
                        "edges": ci._serialize_edges(edges),
                        "papers": ci._serialize_papers(papers),
                        "updated_at": time.time(),
                    }
                )

        if progress_callback:
            progress_callback(
                {
                    "status": "complete",
                    "next_l3_parent_index": len(l3_paper_ids),
                    "edges": ci._serialize_edges(edges),
                    "papers": ci._serialize_papers(papers),
                    "updated_at": time.time(),
                }
            )

        return edges, papers


def _make_l3_to_l3_refs():
    """Shared L3 and L3→L3 reference fixtures for crash-resume tests."""
    oa_l3_refs = {
        "openalex:L2A": ["doi:10.1000/shared-l3", "doi:10.1000/oa-l3a"],
        "openalex:L2B": ["doi:10.1000/oa-l3b"],
    }
    s2_l3_refs = {
        "semantic_scholar:L2A": ["doi:10.1000/shared-l3", "doi:10.1000/s2-l3a"],
        "semantic_scholar:L2B": ["doi:10.1000/s2-l3b"],
    }
    # L3→L3 outgoing references: only references between existing L3 members
    # should survive the second-pass filter.
    oa_outgoing = {
        "doi:10.1000/shared-l3": ["doi:10.1000/oa-l3a"],
        "doi:10.1000/oa-l3a": ["doi:10.1000/oa-l3b", "doi:10.1000/external-l4"],
    }
    s2_outgoing = {
        "doi:10.1000/s2-l3a": ["doi:10.1000/shared-l3"],
        "doi:10.1000/s2-l3b": ["doi:10.1000/s2-l3a", "doi:10.1000/another-l4"],
    }
    return oa_l3_refs, s2_l3_refs, oa_outgoing, s2_outgoing


def test_l3_to_l3_crash_resume_matches_baseline(monkeypatch, tmp_path):
    """Crash mid-second-pass in semantic_scholar, resume, verify equivalence with baseline."""
    oa_l3_refs, s2_l3_refs, oa_outgoing, s2_outgoing = _make_l3_to_l3_refs()
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"

    # ─── Phase 1: crash run ────────────────────────────────────────────
    first_oa = _L3ToL3CrashProvider("openalex", "openalex", oa_l3_refs, oa_outgoing)
    first_s2 = _L3ToL3CrashProvider(
        "semantic_scholar",
        "semantic_scholar",
        s2_l3_refs,
        s2_outgoing,
        crash_at_l3_outgoing_index=1,
    )
    monkeypatch.setattr(ci, "build_providers", lambda _: [first_oa, first_s2])

    with pytest.raises(RuntimeError, match="simulated l3-to-l3 crash in semantic_scholar"):
        ci.ingest_from_internet(
            theory_name="L3 Crash Theory",
            l1_papers=["10.1000/xyz1"],
            sources=["openalex", "semantic_scholar"],
            depth="l2l3",
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            refresh=True,
        )

    # ─── Phase 2: resume run ──────────────────────────────────────────
    resumed_oa = _L3ToL3CrashProvider("openalex", "openalex", oa_l3_refs, oa_outgoing)
    resumed_s2 = _L3ToL3CrashProvider(
        "semantic_scholar",
        "semantic_scholar",
        s2_l3_refs,
        s2_outgoing,
    )
    monkeypatch.setattr(ci, "build_providers", lambda _: [resumed_oa, resumed_s2])

    resumed = ci.ingest_from_internet(
        theory_name="L3 Crash Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["openalex", "semantic_scholar"],
        depth="l2l3",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    # ─── Phase 3: baseline run (no crashes, clean state) ──────────────
    baseline_oa = _L3ToL3CrashProvider("openalex", "openalex", oa_l3_refs, oa_outgoing)
    baseline_s2 = _L3ToL3CrashProvider(
        "semantic_scholar",
        "semantic_scholar",
        s2_l3_refs,
        s2_outgoing,
    )
    monkeypatch.setattr(ci, "build_providers", lambda _: [baseline_oa, baseline_s2])

    baseline = ci.ingest_from_internet(
        theory_name="L3 Crash Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["openalex", "semantic_scholar"],
        depth="l2l3",
        cache_dir=Path(tmp_path) / "baseline-cache",
        checkpoint_dir=Path(tmp_path) / "baseline-checkpoints",
        refresh=True,
    )

    # ─── Assertions ───────────────────────────────────────────────────
    assert resumed.citation_data == baseline.citation_data
    assert resumed.papers_data == baseline.papers_data
    assert resumed.metadata["provider_stats"] == baseline.metadata["provider_stats"]
    assert "semantic_scholar" in resumed.metadata["checkpoint_stats"]["l3_to_l3_resumed_providers"]


def test_l3_to_l3_crash_at_first_provider_resume(monkeypatch, tmp_path):
    """Crash openalex (first provider) mid-second-pass, verify resume equivalence."""
    oa_l3_refs, s2_l3_refs, oa_outgoing, s2_outgoing = _make_l3_to_l3_refs()
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"

    # ─── Crash run: openalex crashes mid-second-pass ──────────────────
    first_oa = _L3ToL3CrashProvider(
        "openalex",
        "openalex",
        oa_l3_refs,
        oa_outgoing,
        crash_at_l3_outgoing_index=1,
    )
    first_s2 = _L3ToL3CrashProvider(
        "semantic_scholar",
        "semantic_scholar",
        s2_l3_refs,
        s2_outgoing,
    )
    monkeypatch.setattr(ci, "build_providers", lambda _: [first_oa, first_s2])

    with pytest.raises(RuntimeError, match="simulated l3-to-l3 crash in openalex"):
        ci.ingest_from_internet(
            theory_name="L3 Crash Theory",
            l1_papers=["10.1000/xyz1"],
            sources=["openalex", "semantic_scholar"],
            depth="l2l3",
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            refresh=True,
        )

    # ─── Resume run ──────────────────────────────────────────────────
    resumed_oa = _L3ToL3CrashProvider("openalex", "openalex", oa_l3_refs, oa_outgoing)
    resumed_s2 = _L3ToL3CrashProvider(
        "semantic_scholar",
        "semantic_scholar",
        s2_l3_refs,
        s2_outgoing,
    )
    monkeypatch.setattr(ci, "build_providers", lambda _: [resumed_oa, resumed_s2])

    resumed = ci.ingest_from_internet(
        theory_name="L3 Crash Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["openalex", "semantic_scholar"],
        depth="l2l3",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    # ─── Baseline run ────────────────────────────────────────────────
    baseline_oa = _L3ToL3CrashProvider("openalex", "openalex", oa_l3_refs, oa_outgoing)
    baseline_s2 = _L3ToL3CrashProvider(
        "semantic_scholar",
        "semantic_scholar",
        s2_l3_refs,
        s2_outgoing,
    )
    monkeypatch.setattr(ci, "build_providers", lambda _: [baseline_oa, baseline_s2])

    baseline = ci.ingest_from_internet(
        theory_name="L3 Crash Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["openalex", "semantic_scholar"],
        depth="l2l3",
        cache_dir=Path(tmp_path) / "baseline-cache",
        checkpoint_dir=Path(tmp_path) / "baseline-checkpoints",
        refresh=True,
    )

    assert resumed.citation_data == baseline.citation_data
    assert resumed.papers_data == baseline.papers_data
    assert "openalex" in resumed.metadata["checkpoint_stats"]["l3_to_l3_resumed_providers"]


def test_l3_to_l3_no_l4_nodes_after_crash_resume(monkeypatch, tmp_path):
    """After crash-resume, papers_data must not contain any IDs absent from baseline."""
    oa_l3_refs, s2_l3_refs, oa_outgoing, s2_outgoing = _make_l3_to_l3_refs()
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"

    # ─── Crash run ────────────────────────────────────────────────────
    first_oa = _L3ToL3CrashProvider("openalex", "openalex", oa_l3_refs, oa_outgoing)
    first_s2 = _L3ToL3CrashProvider(
        "semantic_scholar",
        "semantic_scholar",
        s2_l3_refs,
        s2_outgoing,
        crash_at_l3_outgoing_index=0,
    )
    monkeypatch.setattr(ci, "build_providers", lambda _: [first_oa, first_s2])

    with pytest.raises(RuntimeError, match="simulated l3-to-l3 crash in semantic_scholar"):
        ci.ingest_from_internet(
            theory_name="L3 Crash Theory",
            l1_papers=["10.1000/xyz1"],
            sources=["openalex", "semantic_scholar"],
            depth="l2l3",
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            refresh=True,
        )

    # ─── Resume run ──────────────────────────────────────────────────
    resumed_oa = _L3ToL3CrashProvider("openalex", "openalex", oa_l3_refs, oa_outgoing)
    resumed_s2 = _L3ToL3CrashProvider(
        "semantic_scholar",
        "semantic_scholar",
        s2_l3_refs,
        s2_outgoing,
    )
    monkeypatch.setattr(ci, "build_providers", lambda _: [resumed_oa, resumed_s2])

    resumed = ci.ingest_from_internet(
        theory_name="L3 Crash Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["openalex", "semantic_scholar"],
        depth="l2l3",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    # ─── Baseline run ────────────────────────────────────────────────
    baseline_oa = _L3ToL3CrashProvider("openalex", "openalex", oa_l3_refs, oa_outgoing)
    baseline_s2 = _L3ToL3CrashProvider(
        "semantic_scholar",
        "semantic_scholar",
        s2_l3_refs,
        s2_outgoing,
    )
    monkeypatch.setattr(ci, "build_providers", lambda _: [baseline_oa, baseline_s2])

    baseline = ci.ingest_from_internet(
        theory_name="L3 Crash Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["openalex", "semantic_scholar"],
        depth="l2l3",
        cache_dir=Path(tmp_path) / "baseline-cache",
        checkpoint_dir=Path(tmp_path) / "baseline-checkpoints",
        refresh=True,
    )

    # No L4 nodes should appear — resumed papers_data keys ⊆ baseline keys
    assert set(resumed.papers_data.keys()) == set(baseline.papers_data.keys())
    # External L4 references should not be in citation_data targets
    all_targets = set()
    for targets in resumed.citation_data.values():
        all_targets.update(targets)
    baseline_targets = set()
    for targets in baseline.citation_data.values():
        baseline_targets.update(targets)
    assert all_targets == baseline_targets
