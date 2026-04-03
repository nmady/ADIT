import io
import logging
import threading
import time
import urllib.error
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import citation_ingestion as ci


class _FakeProvider(ci.CitationProvider):
    name = "fake"
    capabilities = ci.ProviderCapabilities(True, True, True)

    def __init__(self):
        self.calls = 0
        self.seed_calls = 0

    def fetch_seed_metadata(self, l1_papers):
        self.seed_calls += 1
        return {
            "doi:10.1000/xyz1": ci.IngestionPaper(
                paper_id="doi:10.1000/xyz1",
                title="Foundational Paper",
                abstract="Canonical theory text.",
                citations=100,
                year=2000,
                doi="10.1000/xyz1",
                source_ids={"fake": "seed:doi:10.1000/xyz1"},
            )
        }

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        self.calls += 1
        return (
            {
                "openalex:W100": {"doi:10.1000/xyz1"},
                "semantic_scholar:abc123": {"doi:10.1000/xyz1"},
            },
            {
                "openalex:W100": ci.IngestionPaper(
                    paper_id="openalex:W100",
                    title="Theory Application Study",
                    year=2022,
                    citations=8,
                    doi="10.1000/abc",
                    source_ids={"openalex": "https://openalex.org/W100"},
                ),
                "semantic_scholar:abc123": ci.IngestionPaper(
                    paper_id="semantic_scholar:abc123",
                    title="Theory Application Study",
                    year=2022,
                    citations=12,
                    doi="10.1000/abc",
                    source_ids={"semantic_scholar": "abc123"},
                ),
            },
        )

    def fetch_l3_references(self, l2_paper_ids, max_l3=None):
        return (
            {"openalex:W100": {"doi:10.1000/l3a"}},
            {
                "doi:10.1000/l3a": ci.IngestionPaper(
                    paper_id="doi:10.1000/l3a",
                    title="L3 supporting work",
                    year=2018,
                    citations=5,
                    doi="10.1000/l3a",
                )
            },
        )


def test_normalize_identifier_handles_doi_and_openalex():
    assert ci.normalize_identifier("10.1234/ABC") == "doi:10.1234/abc"
    assert ci.normalize_identifier("https://doi.org/10.5555/xyz") == "doi:10.5555/xyz"
    assert ci.normalize_identifier("W12345") == "openalex:W12345"
    assert ci.normalize_identifier("https://openalex.org/W999") == "openalex:W999"


def test_build_providers_ignores_unknown_sources():
    providers = ci.build_providers(["openalex", "not-real", "crossref"])
    names = [p.name for p in providers]
    assert names == ["openalex", "crossref"]


def test_build_providers_core_reads_env_key(monkeypatch):
    monkeypatch.setenv("CORE_API_KEY", "secret-key")
    providers = ci.build_providers(["core"])

    assert len(providers) == 1
    assert providers[0].name == "core"
    assert getattr(providers[0], "api_key", None) == "secret-key"


def test_build_providers_core_without_env_key(monkeypatch):
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    providers = ci.build_providers(["core"])

    assert len(providers) == 1
    assert providers[0].name == "core"
    assert getattr(providers[0], "api_key", "missing") is None


def test_core_auth_headers_is_empty_without_key():
    assert ci._core_auth_headers(None) == {}


def test_core_auth_headers_uses_bearer_key():
    assert ci._core_auth_headers("abc123") == {"Authorization": "Bearer abc123"}


def test_semantic_scholar_auth_headers_is_empty_without_key():
    assert ci._semantic_scholar_auth_headers(None) == {}


def test_semantic_scholar_auth_headers_uses_bearer_key():
    assert ci._semantic_scholar_auth_headers("abc123") == {"Authorization": "Bearer abc123"}


def test_build_providers_semantic_scholar_reads_env_key(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "secret-key")
    providers = ci.build_providers(["semantic_scholar"])

    assert len(providers) == 1
    assert providers[0].name == "semantic_scholar"
    assert getattr(providers[0], "api_key", None) == "secret-key"


def test_build_providers_semantic_scholar_without_env_key(monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    providers = ci.build_providers(["semantic_scholar"])

    assert len(providers) == 1
    assert providers[0].name == "semantic_scholar"
    assert getattr(providers[0], "api_key", "missing") is None


def test_should_keep_openalex_item_requires_linked_l1():
    assert ci._should_keep_openalex_item({}, ["doi:10.1000/l1"], "Theory") is True
    assert ci._should_keep_openalex_item({"title": "Theory in title"}, [], "Theory") is False


def test_should_keep_semantic_item_requires_linked_l1():
    assert ci._should_keep_semantic_item({}, {"doi:10.1000/l1"}, "Theory") is True
    assert ci._should_keep_semantic_item({"title": "Theory in title"}, set(), "Theory") is False


def test_crossref_l2_discovery_is_disabled(monkeypatch):
    provider = ci.CrossrefProvider()

    safe_get_called = False

    def fake_safe_get(*args, **kwargs):
        nonlocal safe_get_called
        safe_get_called = True
        return {}

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)

    edges, papers = provider.fetch_l2_and_metadata(
        l1_papers=["doi:10.1000/l1"],
        theory_name="My Fake Theory",
        key_constructs=["usefulness"],
        max_l2=50,
    )

    assert edges == {}
    assert papers == {}
    assert safe_get_called is False


def test_crossref_enrichment_targets_include_only_l2_with_dois():
    all_papers = {
        "openalex:W1": ci.IngestionPaper(paper_id="openalex:W1", doi="10.1000/a"),
        "semantic_scholar:S2": ci.IngestionPaper(paper_id="semantic_scholar:S2", doi="10.1000/b"),
        "openalex:W3": ci.IngestionPaper(paper_id="openalex:W3"),
    }

    targets = ci._crossref_enrichment_targets(
        ["openalex:W1", "semantic_scholar:S2", "openalex:W3"],
        all_papers,
    )

    assert targets == ["doi:10.1000/a", "doi:10.1000/b"]


def test_ingest_from_internet_dedupes_and_uses_cache(monkeypatch, tmp_path):
    provider = _FakeProvider()

    def fake_build_providers(_sources):
        return [provider]

    monkeypatch.setattr(ci, "build_providers", fake_build_providers)

    result1 = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["fake"],
        depth="l2l3",
        cache_dir=Path(tmp_path),
        refresh=False,
        max_l2=10,
        max_l3=10,
    )

    assert provider.calls == 1
    assert provider.seed_calls == 1
    assert result1.metadata["paper_count"] >= 3
    assert result1.metadata["edge_count"] >= 2

    # Duplicate L2 candidates should collapse into a single citing node.
    assert len(result1.citation_data) == 1
    citing_id = next(iter(result1.citation_data.keys()))
    assert len(result1.citation_data[citing_id]) == 2
    merged_entry = result1.papers_data[citing_id]
    assert merged_entry["doi"] == "10.1000/abc"
    assert merged_entry["source_ids"]["openalex"] == "https://openalex.org/W100"
    assert merged_entry["source_ids"]["semantic_scholar"] == "abc123"

    # Second run should come from cache and avoid calling provider again.
    result2 = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["fake"],
        depth="l2l3",
        cache_dir=Path(tmp_path),
        refresh=False,
        max_l2=10,
        max_l3=10,
    )

    assert provider.calls == 1
    assert provider.seed_calls == 1
    assert result2.metadata["cache_key"] == result1.metadata["cache_key"]


def test_ingest_from_internet_hydrates_l1_metadata_from_seed_lookup(monkeypatch, tmp_path):
    provider = _FakeProvider()

    def fake_build_providers(_sources):
        return [provider]

    monkeypatch.setattr(ci, "build_providers", fake_build_providers)

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["fake"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        max_l2=10,
        max_l3=0,
    )

    l1_entry = result.papers_data["doi:10.1000/xyz1"]
    assert l1_entry["title"] == "Foundational Paper"
    assert l1_entry["abstract"] == "Canonical theory text."
    assert l1_entry["citations"] == 100
    assert l1_entry["year"] == 2000
    assert l1_entry["source_ids"] == {"fake": "seed:doi:10.1000/xyz1"}


def test_ingest_from_internet_metadata_includes_fetch_stats(monkeypatch, tmp_path):
    provider = _FakeProvider()

    def fake_build_providers(_sources):
        return [provider]

    monkeypatch.setattr(ci, "build_providers", fake_build_providers)

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["fake"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        max_l2=10,
        max_l3=0,
    )

    assert "fetch_stats" in result.metadata
    stats = result.metadata["fetch_stats"]
    assert isinstance(stats["total_requests"], int)
    assert isinstance(stats["total_failures"], int)
    assert isinstance(stats["per_provider_failures"], dict)
    assert isinstance(stats["per_status_failures"], dict)
    assert stats["per_provider_failures"]["fake"] == 0


def test_crossref_enriches_existing_l2_metadata_without_adding_l2_edges(monkeypatch, tmp_path):
    fake_provider = _FakeProvider()
    crossref_provider = ci.CrossrefProvider()

    def fake_build_providers(_sources):
        return [fake_provider, crossref_provider]

    crossref_calls = []

    def fake_crossref_seed_metadata(self, l1_papers):
        crossref_calls.append(list(l1_papers))
        if l1_papers == ["doi:10.1000/xyz1"]:
            return {}
        assert l1_papers == ["doi:10.1000/abc"]
        return {
            "doi:10.1000/abc": ci.IngestionPaper(
                paper_id="doi:10.1000/abc",
                title="Crossref Enriched Title",
                citations=99,
                year=2021,
                doi="10.1000/abc",
                source_ids={"crossref": "10.1000/abc"},
            )
        }

    monkeypatch.setattr(ci, "build_providers", fake_build_providers)
    monkeypatch.setattr(ci.CrossrefProvider, "fetch_seed_metadata", fake_crossref_seed_metadata)

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["fake", "crossref"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        max_l2=10,
        max_l3=0,
    )

    assert result.metadata["provider_stats"]["crossref"]["l2_nodes"] == 0
    assert result.metadata["provider_stats"]["crossref"]["l2_edges"] == 0
    assert result.metadata["provider_stats"]["crossref"]["metadata_enriched"] == 1
    assert crossref_calls == [["doi:10.1000/xyz1"], ["doi:10.1000/abc"]]

    citing_id = next(iter(result.citation_data.keys()))
    assert result.papers_data[citing_id]["source_ids"]["crossref"] == "10.1000/abc"


# ---------------------------------------------------------------------------
# Cited-by traversal: exhaustive pagination
# ---------------------------------------------------------------------------


class _PagedCitedByProvider(ci.CitationProvider):
    """Fake provider that supports cited-by traversal and returns 2 pages of citers."""

    name = "paged"
    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=True)

    def __init__(self):
        self.citers_calls: list = []

    def fetch_seed_metadata(self, l1_papers):
        return {
            l1: ci.IngestionPaper(
                paper_id=l1,
                title="Foundational",
                citations=50,
                year=2000,
                doi="10.1000/xyz1",
                source_ids={"paged": "PAGED001"},
            )
            for l1 in l1_papers
        }

    def fetch_citers_for_l1(self, l1_provider_id, max_results=None):
        self.citers_calls.append(l1_provider_id)
        page1 = {
            "paged:P001": ci.IngestionPaper(
                paper_id="paged:P001", title="Citer One", year=2021, citations=5
            ),
            "paged:P002": ci.IngestionPaper(
                paper_id="paged:P002", title="Citer Two", year=2022, citations=3
            ),
        }
        page2 = {
            "paged:P003": ci.IngestionPaper(
                paper_id="paged:P003", title="Citer Three", year=2023, citations=1
            ),
        }
        all_papers = {**page1, **page2}
        if max_results is not None:
            truncated = dict(list(all_papers.items())[:max_results])
            status = "partial" if len(truncated) < len(all_papers) else "complete"
            return truncated, len(all_papers), status
        return all_papers, len(all_papers), "complete"

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        return {}, {}

    def fetch_l3_references(self, l2_paper_ids, max_l3=None):
        return {}, {}


def test_ingest_exhaustive_fetches_all_pages(monkeypatch, tmp_path):
    """Exhaustive mode should collect all citers across provider pages."""
    provider = _PagedCitedByProvider()

    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["paged"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        exhaustive=True,
    )

    # All 3 citers must appear in citation_data.
    all_citing = set(result.citation_data.keys())
    assert len(all_citing) == 3

    # fetch_citers_for_l1 was called once, for the resolved L1 ID.
    assert len(provider.citers_calls) == 1


def test_ingest_sample_mode_caps_results(monkeypatch, tmp_path):
    """Non-exhaustive mode should cap citers at max_l2."""
    provider = _PagedCitedByProvider()

    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["paged"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        exhaustive=False,
        max_l2=2,
    )

    assert len(result.citation_data) <= 2


def test_ingest_completeness_in_metadata(monkeypatch, tmp_path):
    """metadata['completeness'] should report status per L1 per provider."""
    provider = _PagedCitedByProvider()

    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["paged"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        exhaustive=True,
    )

    assert "completeness" in result.metadata
    completeness = result.metadata["completeness"]
    # Should have an entry for the L1 paper.
    assert len(completeness) == 1
    l1_entry = next(iter(completeness.values()))
    assert "paged" in l1_entry
    assert l1_entry["paged"]["status"] == "complete"
    assert l1_entry["paged"]["fetched"] == 3
    assert l1_entry["paged"]["expected"] == 3


def test_ingest_sample_mode_marks_completeness_partial(monkeypatch, tmp_path):
    """Non-exhaustive mode should report partial completeness when capped."""
    provider = _PagedCitedByProvider()

    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["paged"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        exhaustive=False,
        max_l2=2,
    )

    l1_entry = next(iter(result.metadata["completeness"].values()))
    assert l1_entry["paged"]["status"] == "partial"
    assert l1_entry["paged"]["fetched"] == 2
    assert l1_entry["paged"]["expected"] == 3
    assert l1_entry["paged"]["fetched"] < l1_entry["paged"]["expected"]


class _EarlyStopCitedByProvider(ci.CitationProvider):
    """Fake provider that stops early and reports partial fetch status."""

    name = "earlystop"
    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=True)

    def fetch_seed_metadata(self, l1_papers):
        return {
            l1: ci.IngestionPaper(
                paper_id=l1,
                title="Foundational",
                citations=50,
                year=2000,
                doi="10.1000/xyz1",
                source_ids={"earlystop": "EARLY001"},
            )
            for l1 in l1_papers
        }

    def fetch_citers_for_l1(self, l1_provider_id, max_results=None):
        # Simulate a provider run that fetched some pages and then failed.
        return (
            {
                "earlystop:C001": ci.IngestionPaper(
                    paper_id="earlystop:C001", title="Citer One", year=2021, citations=4
                ),
                "earlystop:C002": ci.IngestionPaper(
                    paper_id="earlystop:C002", title="Citer Two", year=2022, citations=2
                ),
            },
            5,
            "partial",
        )

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        return {}, {}

    def fetch_l3_references(self, l2_paper_ids, max_l3=None):
        return {}, {}


class _ProgressCitedByProvider(ci.CitationProvider):
    """Fake provider that emits in-progress callback updates with expected citer counts."""

    name = "progress"
    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=True)

    def fetch_seed_metadata(self, l1_papers):
        return {
            l1: ci.IngestionPaper(
                paper_id=l1,
                title="Foundational",
                citations=42,
                year=2005,
                doi="10.1000/xyz1",
                source_ids={"progress": "PROGRESS001"},
            )
            for l1 in l1_papers
        }

    def fetch_citers_for_l1(
        self,
        l1_provider_id,
        max_results=None,
        resume_state=None,
        progress_callback=None,
    ):
        papers = {
            "progress:C001": ci.IngestionPaper(
                paper_id="progress:C001", title="Citer One", year=2020, citations=2
            ),
            "progress:C002": ci.IngestionPaper(
                paper_id="progress:C002", title="Citer Two", year=2021, citations=3
            ),
            "progress:C003": ci.IngestionPaper(
                paper_id="progress:C003", title="Citer Three", year=2022, citations=4
            ),
        }
        if progress_callback is not None:
            progress_callback({"status": "in_progress", "fetched_count": 1, "expected_count": 3})
            progress_callback({"status": "in_progress", "fetched_count": 2, "expected_count": 3})
        return papers, 3, "complete"

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        return {}, {}

    def fetch_l3_references(self, l2_paper_ids, max_l3=None):
        return {}, {}


def test_ingest_completeness_partial_when_provider_stops_early(monkeypatch, tmp_path):
    """Provider-level partial status should propagate into metadata completeness."""
    provider = _EarlyStopCitedByProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["earlystop"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        exhaustive=True,
    )

    assert len(result.citation_data) == 2
    l1_entry = next(iter(result.metadata["completeness"].values()))
    assert l1_entry["earlystop"]["status"] == "partial"
    assert l1_entry["earlystop"]["fetched"] == 2
    assert l1_entry["earlystop"]["expected"] == 5
    assert l1_entry["earlystop"]["fetched"] < l1_entry["earlystop"]["expected"]


def test_ingest_progress_reports_citer_fraction_from_callback(monkeypatch, tmp_path, capsys):
    """In-progress traversal should emit citer fetched/expected updates when provided."""
    provider = _ProgressCitedByProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["progress"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        exhaustive=True,
    )
    captured = capsys.readouterr()

    assert "Seed 1/1: citers 1/3" in captured.err
    assert "Seed 1/1: citers 2/3" in captured.err


class _L3BudgetProvider(ci.CitationProvider):
    """Fake provider that returns fixed L2 nodes and budget-aware L3 references."""

    name = "l3budget"
    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=True)

    def __init__(self):
        self.max_l3_calls = []

    def fetch_seed_metadata(self, l1_papers):
        return {
            l1: ci.IngestionPaper(
                paper_id=l1,
                title="Foundational",
                citations=80,
                year=2001,
                doi="10.1000/xyz1",
                source_ids={"l3budget": "L3SEED001"},
            )
            for l1 in l1_papers
        }

    def fetch_citers_for_l1(self, l1_provider_id, max_results=None):
        papers = {
            "l3budget:L201": ci.IngestionPaper(
                paper_id="l3budget:L201", title="L2 One", year=2020, citations=5
            ),
            "l3budget:L202": ci.IngestionPaper(
                paper_id="l3budget:L202", title="L2 Two", year=2021, citations=6
            ),
        }
        if max_results is not None:
            papers = dict(list(papers.items())[:max_results])
        status = "complete" if max_results is None or len(papers) == 2 else "partial"
        return papers, 2, status

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        return {}, {}

    def fetch_l3_references(self, l2_paper_ids, max_l3=None):
        self.max_l3_calls.append(max_l3)
        candidates = [
            ("l3budget:L201", "l3budget:R301"),
            ("l3budget:L201", "l3budget:R302"),
            ("l3budget:L202", "l3budget:R303"),
            ("l3budget:L202", "l3budget:R304"),
        ]
        budget = len(candidates) if max_l3 is None else max(0, max_l3)
        edges = {}
        papers = {}

        for parent_id, ref_id in candidates[:budget]:
            if parent_id not in l2_paper_ids:
                continue
            edges.setdefault(parent_id, set()).add(ref_id)
            papers.setdefault(
                ref_id,
                ci.IngestionPaper(
                    paper_id=ref_id,
                    title=f"Reference {ref_id}",
                    year=2015,
                    citations=1,
                ),
            )

        return edges, papers


def test_ingest_l3_without_budget_fetches_all_available_refs(monkeypatch, tmp_path):
    """When max_l3 is unset, provider should contribute all available L3 edges."""
    provider = _L3BudgetProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["l3budget"],
        depth="l2l3",
        cache_dir=Path(tmp_path),
        refresh=True,
        exhaustive=True,
    )

    assert provider.max_l3_calls == [None]
    assert result.metadata["provider_stats"]["l3budget"]["l3_edges"] == 4
    l3_cited = {
        cited_id
        for cited_set in result.citation_data.values()
        for cited_id in cited_set
        if cited_id.startswith("l3budget:R")
    }
    assert len(l3_cited) == 4


def test_ingest_l3_budget_caps_reference_edges(monkeypatch, tmp_path):
    """Explicit max_l3 should cap provider-contributed L3 reference edges."""
    provider = _L3BudgetProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["l3budget"],
        depth="l2l3",
        cache_dir=Path(tmp_path),
        refresh=True,
        exhaustive=True,
        max_l3=2,
    )

    assert provider.max_l3_calls == [2]
    assert result.metadata["provider_stats"]["l3budget"]["l3_edges"] == 2
    l3_cited = {
        cited_id
        for cited_set in result.citation_data.values()
        for cited_id in cited_set
        if cited_id.startswith("l3budget:R")
    }
    assert len(l3_cited) == 2


def test_ingest_l3_zero_budget_adds_no_l3_edges(monkeypatch, tmp_path):
    """max_l3=0 with depth=l2l3 should preserve only L2 edges."""
    provider = _L3BudgetProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["l3budget"],
        depth="l2l3",
        cache_dir=Path(tmp_path),
        refresh=True,
        exhaustive=True,
        max_l3=0,
    )

    assert provider.max_l3_calls == [0]
    assert result.metadata["provider_stats"]["l3budget"]["l3_edges"] == 0
    l3_cited = {
        cited_id
        for cited_set in result.citation_data.values()
        for cited_id in cited_set
        if cited_id.startswith("l3budget:R")
    }
    assert len(l3_cited) == 0


# ---------------------------------------------------------------------------
# _safe_get: retry on transient errors, stop on permanent errors
# ---------------------------------------------------------------------------


def _make_http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example.com", code=code, msg="err", hdrs=None, fp=None
    )


def test_safe_get_retries_on_429_then_succeeds(monkeypatch):
    """_safe_get should retry on HTTP 429 and return the response when the retry succeeds."""
    call_count = [0]

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] < 3:
            raise _make_http_error(429)
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'
        return mock_resp

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    clock = {"value": 1_000.0}
    monkeypatch.setattr(ci.time, "time", lambda: clock["value"])
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    ci._reset_ingest_stats(["test"])

    result = ci._safe_get("https://example.com/test", provider="test")
    assert result == {"ok": True}
    assert call_count[0] == 3


def test_safe_get_stops_immediately_on_permanent_failure(monkeypatch):
    """_safe_get should not retry on HTTP 403 and should return None immediately."""
    call_count = [0]

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        raise _make_http_error(403)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    clock = {"value": 1_000.0}
    monkeypatch.setattr(ci.time, "time", lambda: clock["value"])
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    ci._reset_ingest_stats(["test"])

    result = ci._safe_get("https://example.com/test", provider="test")
    assert result is None
    assert call_count[0] == 1  # No retries for permanent failures.
    assert ci._INGEST_STATS["total_failures"] == 1


def test_safe_get_exhausts_retries_and_returns_none(monkeypatch):
    """_safe_get should return None after exhausting all retries on transient errors."""
    call_count = [0]

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        raise _make_http_error(500)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    ci._reset_ingest_stats(["test"])

    result = ci._safe_get("https://example.com/test", provider="test", max_retries=3)
    assert result is None
    assert call_count[0] == 3
    assert ci._INGEST_STATS["total_failures"] == 1


def test_provider_checkpoint_round_trip_transient_failures(tmp_path):
    checkpoint_root = Path(tmp_path) / "checkpoints"
    key = "req-key"
    provider_name = "openalex"
    failures = {
        "openalex:safe_get:https://api.openalex.org/works/W1": {
            "op": "safe_get",
            "provider": provider_name,
            "target_id": "https://api.openalex.org/works/W1",
            "resume_state": {"timeout": 20, "max_retries": 5, "headers": {"Accept": "json"}},
            "error_code": 429,
            "error_type": "http_error",
            "attempts": 2,
            "last_attempt_ts": time.time(),
            "server_retry_after": 30.0,
        }
    }

    ci._write_provider_checkpoint_state(
        checkpoint_root=checkpoint_root,
        key=key,
        provider_name=provider_name,
        provider_pagination_state={},
        provider_l3_state={},
        transient_failures=failures,
    )

    payload = ci._load_provider_checkpoint_state(
        checkpoint_root=checkpoint_root,
        key=key,
        provider_name=provider_name,
        reset_checkpoints=False,
    )
    assert payload is not None
    restored = ci._deserialize_transient_failures(
        {provider_name: payload.get("transient_failures")}
    )
    assert provider_name in restored
    assert list(restored[provider_name]) == ["openalex:safe_get:https://api.openalex.org/works/W1"]


def test_replay_provider_transient_failures_retries_eligible(monkeypatch):
    ci._drain_transient_request_failures()
    stats = {}
    failures = {
        "openalex:safe_get:https://api.openalex.org/works/W1": {
            "op": "safe_get",
            "provider": "openalex",
            "target_id": "https://api.openalex.org/works/W1",
            "resume_state": {"timeout": 20, "max_retries": 2, "headers": {}},
            "attempts": 1,
            "last_attempt_ts": time.time() - 120,
            "server_retry_after": None,
        }
    }

    monkeypatch.setattr(ci, "_safe_get", lambda *args, **kwargs: {"ok": True})

    ci._replay_provider_transient_failures("openalex", failures, stats)

    assert failures == {}
    assert stats["transient_failures_retried"] == 1
    assert stats["transient_failures_resumed_success"] == 1


def test_replay_provider_transient_failures_skips_unready(monkeypatch):
    stats = {}
    failures = {
        "openalex:safe_get:https://api.openalex.org/works/W2": {
            "op": "safe_get",
            "provider": "openalex",
            "target_id": "https://api.openalex.org/works/W2",
            "resume_state": {"timeout": 20, "max_retries": 2, "headers": {}},
            "attempts": 1,
            "last_attempt_ts": time.time(),
            "server_retry_after": 300.0,
        }
    }

    called = {"count": 0}

    def _fake_safe_get(*args, **kwargs):
        called["count"] += 1
        return {"ok": True}

    monkeypatch.setattr(ci, "_safe_get", _fake_safe_get)

    ci._replay_provider_transient_failures("openalex", failures, stats)

    assert called["count"] == 0
    assert len(failures) == 1


def test_safe_get_suppresses_http_error_body_without_debug_http(monkeypatch, caplog):
    """_safe_get should suppress response bodies unless debug-http mode is enabled."""

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            url="https://example.com/test",
            code=400,
            msg="bad request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"offset + limit must be < 10000"}'),
        )

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    ci._reset_ingest_stats(["test"])
    ci.set_debug_http(False)

    with caplog.at_level(logging.WARNING):
        result = ci._safe_get("https://example.com/test", provider="test")

    assert result is None
    assert "offset + limit must be < 10000" not in caplog.text
    assert "suppressed" in caplog.text


def test_safe_get_logs_http_error_body_when_debug_http_enabled(monkeypatch, caplog):
    """_safe_get should include response bodies when debug-http mode is enabled."""

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            url="https://example.com/test",
            code=400,
            msg="bad request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"offset + limit must be < 10000"}'),
        )

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    ci._reset_ingest_stats(["test"])
    ci.set_debug_http(True)

    try:
        with caplog.at_level(logging.WARNING):
            result = ci._safe_get("https://example.com/test", provider="test")
    finally:
        ci.set_debug_http(False)

    assert result is None
    assert "offset + limit must be < 10000" in caplog.text


def test_semantic_citers_caps_limit_before_api_boundary(monkeypatch):
    """Semantic Scholar pagination should cap page size so offset + limit stays below 10000."""
    provider = ci.SemanticScholarProvider()
    request_pairs = []

    def fake_safe_get(
        url,
        timeout=20,
        provider=None,
        max_retries=ci._SAFE_GET_MAX_RETRIES,
        headers=None,
    ):
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        offset = int(params["offset"][0])
        limit = int(params["limit"][0])
        request_pairs.append((offset, limit))
        assert offset + limit < 10000

        data = []
        for index in range(limit):
            paper_index = offset + index
            data.append(
                {
                    "citingPaper": {
                        "paperId": f"paper-{paper_index}",
                        "title": f"Paper {paper_index}",
                        "year": 2020,
                        "citationCount": 1,
                        "externalIds": {},
                        "abstract": "",
                    }
                }
            )

        return {
            "total": 12000,
            "data": data,
            "next": "has-more",
        }

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    papers, expected_count, status = provider.fetch_citers_for_l1("semantic_scholar:seed")

    assert request_pairs[-1] == (9000, 999)
    assert len(request_pairs) == 10
    assert len(papers) == 9999
    assert expected_count == 12000
    assert status == "partial"


def test_semantic_seed_metadata_passes_auth_headers_when_configured(monkeypatch):
    provider = ci.SemanticScholarProvider(api_key="secret-key")
    captured_headers = []

    def fake_safe_get(
        url, timeout=20, provider=None, max_retries=ci._SAFE_GET_MAX_RETRIES, headers=None
    ):
        captured_headers.append(headers)
        return {
            "paperId": "abc123",
            "title": "Seed Paper",
            "abstract": "A seeded abstract.",
            "year": 2020,
            "citationCount": 10,
            "externalIds": {"DOI": "10.1000/xyz1"},
        }

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    papers = provider.fetch_seed_metadata(["doi:10.1000/xyz1"])

    assert papers["doi:10.1000/xyz1"].title == "Seed Paper"
    assert captured_headers == [{"Authorization": "Bearer secret-key"}]


def test_semantic_citers_pass_auth_headers_when_configured(monkeypatch):
    provider = ci.SemanticScholarProvider(api_key="secret-key")
    captured_headers = []

    def fake_safe_get(
        url, timeout=20, provider=None, max_retries=ci._SAFE_GET_MAX_RETRIES, headers=None
    ):
        captured_headers.append(headers)
        return {
            "total": 1,
            "data": [
                {
                    "citingPaper": {
                        "paperId": "paper-1",
                        "title": "Paper 1",
                        "year": 2020,
                        "citationCount": 1,
                        "externalIds": {},
                        "abstract": "",
                    }
                }
            ],
        }

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    papers, expected_count, status = provider.fetch_citers_for_l1("semantic_scholar:seed")

    assert len(papers) == 1
    assert expected_count == 1
    assert status == "complete"
    assert captured_headers == [{"Authorization": "Bearer secret-key"}]


def test_semantic_l2_search_passes_auth_headers_when_configured(monkeypatch):
    provider = ci.SemanticScholarProvider(api_key="secret-key")
    captured_headers = []

    def fake_safe_get(
        url, timeout=20, provider=None, max_retries=ci._SAFE_GET_MAX_RETRIES, headers=None
    ):
        captured_headers.append(headers)
        return {
            "data": [
                {
                    "paperId": "paper-1",
                    "title": "Paper 1",
                    "year": 2020,
                    "citationCount": 1,
                    "references": [{"paperId": "xyz1", "externalIds": {"DOI": "10.1000/l1"}}],
                    "externalIds": {},
                    "abstract": "",
                }
            ]
        }

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    edges, papers = provider.fetch_l2_and_metadata(
        ["doi:10.1000/l1"],
        theory_name="Theory",
        key_constructs=["usefulness"],
        max_l2=10,
    )

    assert edges == {"semantic_scholar:paper-1": {"doi:10.1000/l1"}}
    assert "semantic_scholar:paper-1" in papers
    assert captured_headers == [{"Authorization": "Bearer secret-key"}]


def test_semantic_l3_requests_pass_auth_headers_when_configured(monkeypatch):
    provider = ci.SemanticScholarProvider(api_key="secret-key")
    captured_headers = []

    def fake_safe_get(
        url, timeout=20, provider=None, max_retries=ci._SAFE_GET_MAX_RETRIES, headers=None
    ):
        captured_headers.append(headers)
        return {
            "references": [
                {
                    "paperId": "abc123",
                    "externalIds": {"DOI": "10.1000/l3a"},
                }
            ]
        }

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    edges, papers = provider.fetch_l3_references(["semantic_scholar:seed"], max_l3=10)

    assert edges["semantic_scholar:seed"] == {"doi:10.1000/l3a"}
    assert papers["doi:10.1000/l3a"].source_ids == {"semantic_scholar": "abc123"}
    assert captured_headers == [{"Authorization": "Bearer secret-key"}]


def test_semantic_citers_stop_without_requesting_at_offset_9999(monkeypatch):
    """Semantic Scholar pagination should stop cleanly once the next request would cross the API ceiling."""
    provider = ci.SemanticScholarProvider()
    request_count = [0]
    request_offsets = []

    def fake_safe_get(
        url,
        timeout=20,
        provider=None,
        max_retries=ci._SAFE_GET_MAX_RETRIES,
        headers=None,
    ):
        request_count[0] += 1
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        offset = int(params["offset"][0])
        limit = int(params["limit"][0])
        request_offsets.append(offset)
        assert request_count[0] <= 10
        assert offset + limit < 10000
        return {
            "total": 10050,
            "data": [
                {
                    "citingPaper": {
                        "paperId": f"paper-{offset + index}",
                        "title": f"Paper {offset + index}",
                        "year": 2020,
                        "citationCount": 1,
                        "externalIds": {},
                        "abstract": "",
                    }
                }
                for index in range(limit)
            ],
            "next": "has-more",
        }

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    papers, expected_count, status = provider.fetch_citers_for_l1("semantic_scholar:seed")

    assert request_count[0] == 10
    assert request_offsets[-1] == 9000
    assert len(papers) == 9999
    assert expected_count == 10050
    assert status == "partial"


def test_semantic_l3_requests_minimal_fields_and_uses_doi_ids(monkeypatch):
    provider = ci.SemanticScholarProvider()
    requested_urls = []

    def fake_safe_get(
        url,
        timeout=20,
        provider=None,
        max_retries=ci._SAFE_GET_MAX_RETRIES,
        headers=None,
    ):
        requested_urls.append(url)
        return {
            "references": [
                {
                    "paperId": "abc123",
                    "externalIds": {"DOI": "10.1000/l3a"},
                },
                {
                    "paperId": "no-doi-ref",
                    "externalIds": {},
                },
            ]
        }

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    edges, papers = provider.fetch_l3_references(["semantic_scholar:seed"], max_l3=10)

    assert requested_urls
    assert requested_urls[0].endswith("?fields=references.paperId,references.externalIds")
    assert edges["semantic_scholar:seed"] == {"doi:10.1000/l3a", "semantic_scholar:no-doi-ref"}
    assert papers["doi:10.1000/l3a"].doi == "10.1000/l3a"
    assert papers["doi:10.1000/l3a"].source_ids == {"semantic_scholar": "abc123"}
    assert papers["semantic_scholar:no-doi-ref"].source_ids == {"semantic_scholar": "no-doi-ref"}
    assert papers["doi:10.1000/l3a"].title == ""
    assert papers["doi:10.1000/l3a"].abstract == ""


def test_semantic_l3_default_budget_fetches_all_available_refs(monkeypatch):
    provider = ci.SemanticScholarProvider()

    def fake_safe_get(
        url,
        timeout=20,
        provider=None,
        max_retries=ci._SAFE_GET_MAX_RETRIES,
        headers=None,
    ):
        return {
            "references": [
                {"paperId": "r1", "externalIds": {}},
                {"paperId": "r2", "externalIds": {}},
                {"paperId": "r3", "externalIds": {}},
            ]
        }

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    edges, papers = provider.fetch_l3_references(["semantic_scholar:seed"])

    assert edges["semantic_scholar:seed"] == {
        "semantic_scholar:r1",
        "semantic_scholar:r2",
        "semantic_scholar:r3",
    }
    assert len(papers) == 3


def test_openalex_l3_malformed_resume_state_fails_open(monkeypatch):
    provider = ci.OpenAlexProvider()

    def fake_safe_get(url, timeout=20, provider=None, max_retries=ci._SAFE_GET_MAX_RETRIES):
        if url.endswith("W100?select=referenced_works"):
            return {"referenced_works": ["https://openalex.org/W200"]}
        if url.endswith("W200?select=id,doi,title,publication_year"):
            return {
                "id": "https://openalex.org/W200",
                "doi": "https://doi.org/10.1000/l3a",
                "title": "Hydrated L3",
                "publication_year": 2018,
            }
        return None

    malformed_resume_state = {
        "next_l2_index": "bad-index",
        "budget_remaining": "bad-budget",
        "edges": ["not-a-dict"],
        "papers": "not-a-dict",
    }

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    edges, papers = provider.fetch_l3_references(
        ["openalex:W100"],
        max_l3=2,
        resume_state=malformed_resume_state,
    )

    assert edges == {"openalex:W100": {"openalex:W200"}}
    assert papers["openalex:W200"].title == "Hydrated L3"


def test_semantic_l3_malformed_resume_state_fails_open(monkeypatch):
    provider = ci.SemanticScholarProvider()

    def fake_safe_get(
        url,
        timeout=20,
        provider=None,
        max_retries=ci._SAFE_GET_MAX_RETRIES,
        headers=None,
    ):
        return {
            "references": [
                {
                    "paperId": "abc123",
                    "externalIds": {"DOI": "10.1000/l3a"},
                }
            ]
        }

    malformed_resume_state = {
        "next_l2_index": "bad-index",
        "budget_remaining": "bad-budget",
        "edges": ["not-a-dict"],
        "papers": "not-a-dict",
    }

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    edges, papers = provider.fetch_l3_references(
        ["semantic_scholar:seed"],
        max_l3=2,
        resume_state=malformed_resume_state,
    )

    assert edges == {"semantic_scholar:seed": {"doi:10.1000/l3a"}}
    assert papers["doi:10.1000/l3a"].source_ids == {"semantic_scholar": "abc123"}


def test_openalex_l3_requests_minimal_edges_then_identity_hydration(monkeypatch):
    provider = ci.OpenAlexProvider()
    requested_urls = []

    def fake_safe_get(url, timeout=20, provider=None, max_retries=ci._SAFE_GET_MAX_RETRIES):
        requested_urls.append(url)
        if url.endswith("?select=referenced_works"):
            return {"referenced_works": ["https://openalex.org/W200"]}
        if url.endswith("?select=id,doi,title,publication_year"):
            return {
                "id": "https://openalex.org/W200",
                "doi": "https://doi.org/10.1000/l3a",
                "title": "Hydrated L3",
                "publication_year": 2018,
            }
        return None

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    edges, papers = provider.fetch_l3_references(["openalex:W100"], max_l3=10)

    assert requested_urls == [
        "https://api.openalex.org/works/W100?select=referenced_works",
        "https://api.openalex.org/works/W200?select=id,doi,title,publication_year",
    ]
    assert edges == {"openalex:W100": {"openalex:W200"}}
    assert papers["openalex:W200"].doi == "https://doi.org/10.1000/l3a"
    assert papers["openalex:W200"].title == "Hydrated L3"
    assert papers["openalex:W200"].year == 2018
    assert papers["openalex:W200"].source_ids == {"openalex": "https://openalex.org/W200"}


def test_openalex_l3_default_budget_fetches_all_available_refs(monkeypatch):
    provider = ci.OpenAlexProvider()

    def fake_safe_get(url, timeout=20, provider=None, max_retries=ci._SAFE_GET_MAX_RETRIES):
        if url.endswith("W100?select=referenced_works"):
            return {
                "referenced_works": [
                    "https://openalex.org/W201",
                    "https://openalex.org/W202",
                    "https://openalex.org/W203",
                ]
            }
        if url.endswith("W201?select=id,doi,title,publication_year"):
            return {
                "id": "https://openalex.org/W201",
                "doi": "https://doi.org/10.1000/oa201",
                "title": "OA Ref 201",
                "publication_year": 2016,
            }
        if url.endswith("W202?select=id,doi,title,publication_year"):
            return {
                "id": "https://openalex.org/W202",
                "doi": "https://doi.org/10.1000/oa202",
                "title": "OA Ref 202",
                "publication_year": 2017,
            }
        if url.endswith("W203?select=id,doi,title,publication_year"):
            return {
                "id": "https://openalex.org/W203",
                "doi": "https://doi.org/10.1000/oa203",
                "title": "OA Ref 203",
                "publication_year": 2018,
            }
        return None

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    edges, papers = provider.fetch_l3_references(["openalex:W100"])

    assert edges["openalex:W100"] == {
        "openalex:W201",
        "openalex:W202",
        "openalex:W203",
    }
    assert len(papers) == 3


def test_openalex_l3_budget_caps_reference_edges(monkeypatch):
    provider = ci.OpenAlexProvider()
    hydrated_ids = []

    def fake_safe_get(url, timeout=20, provider=None, max_retries=ci._SAFE_GET_MAX_RETRIES):
        if url.endswith("W100?select=referenced_works"):
            return {
                "referenced_works": [
                    "https://openalex.org/W201",
                    "https://openalex.org/W202",
                    "https://openalex.org/W203",
                ]
            }
        if "?select=id,doi,title,publication_year" in url:
            token = url.split("/works/")[-1].split("?")[0]
            hydrated_ids.append(token)
            return {
                "id": f"https://openalex.org/{token}",
                "doi": f"https://doi.org/10.1000/{token.lower()}",
                "title": f"Hydrated {token}",
                "publication_year": 2019,
            }
        return None

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    edges, papers = provider.fetch_l3_references(["openalex:W100"], max_l3=2)

    assert len(edges["openalex:W100"]) == 2
    assert len(papers) == 2
    assert sorted(hydrated_ids) == ["W201", "W202"]


def test_core_l3_default_budget_fetches_all_available_refs(monkeypatch):
    provider = ci.CoreProvider()

    def fake_lookup_work(pid):
        assert pid == "doi:10.1000/l2"
        return {
            "references": [
                {"id": 1, "doi": "10.1000/core-r1", "title": "Core Ref 1"},
                {"id": 2, "doi": "10.1000/core-r2", "title": "Core Ref 2"},
                {"id": 3, "doi": "10.1000/core-r3", "title": "Core Ref 3"},
            ]
        }

    monkeypatch.setattr(provider, "_lookup_work", fake_lookup_work)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    edges, papers = provider.fetch_l3_references(["doi:10.1000/l2"])

    assert edges["doi:10.1000/l2"] == {
        "doi:10.1000/core-r1",
        "doi:10.1000/core-r2",
        "doi:10.1000/core-r3",
    }
    assert len(papers) == 3


def test_core_l3_budget_caps_reference_edges(monkeypatch):
    provider = ci.CoreProvider()

    def fake_lookup_work(pid):
        assert pid == "doi:10.1000/l2"
        return {
            "references": [
                {"id": 1, "doi": "10.1000/core-r1", "title": "Core Ref 1"},
                {"id": 2, "doi": "10.1000/core-r2", "title": "Core Ref 2"},
                {"id": 3, "doi": "10.1000/core-r3", "title": "Core Ref 3"},
            ]
        }

    monkeypatch.setattr(provider, "_lookup_work", fake_lookup_work)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    edges, papers = provider.fetch_l3_references(["doi:10.1000/l2"], max_l3=2)

    assert len(edges["doi:10.1000/l2"]) == 2
    assert len(papers) == 2


def test_core_l3_malformed_resume_state_fails_open(monkeypatch):
    provider = ci.CoreProvider()

    def fake_lookup_work(pid):
        if pid == "core:l2a":
            return {
                "references": [
                    {"id": "a1", "doi": "10.1000/core-a1", "title": "Core A1"},
                ]
            }
        if pid == "core:l2b":
            return {
                "references": [
                    {"id": "b1", "doi": "10.1000/core-b1", "title": "Core B1"},
                ]
            }
        return None

    malformed_resume_state = {
        "next_l2_index": "not-an-int",
        "budget_remaining": "not-an-int",
        "edges": ["not-a-dict"],
        "papers": "not-a-dict",
    }

    monkeypatch.setattr(provider, "_lookup_work", fake_lookup_work)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    edges, papers = provider.fetch_l3_references(
        ["core:l2a", "core:l2b"],
        max_l3=2,
        resume_state=malformed_resume_state,
    )

    assert edges == {
        "core:l2a": {"doi:10.1000/core-a1"},
        "core:l2b": {"doi:10.1000/core-b1"},
    }
    assert len(papers) == 2


def test_crossref_l3_contract_returns_empty_graph():
    provider = ci.CrossrefProvider()
    edges, papers = provider.fetch_l3_references(["doi:10.1000/l2"], max_l3=10)
    assert edges == {}
    assert papers == {}


class _MultiProviderA(ci.CitationProvider):
    name = "multi_a"
    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=True)

    def fetch_seed_metadata(self, l1_papers):
        return {
            l1: ci.IngestionPaper(
                paper_id=l1,
                title="Seed A",
                source_ids={"multi_a": "SEED-A"},
            )
            for l1 in l1_papers
        }

    def fetch_citers_for_l1(self, l1_provider_id, max_results=None):
        papers = {
            "multi_a:L2A": ci.IngestionPaper(
                paper_id="multi_a:L2A", title="L2 A", year=2020, citations=5
            )
        }
        return papers, 1, "complete"

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        return {}, {}

    def fetch_l3_references(self, l2_paper_ids, max_l3=None):
        candidates = ["multi_a:R1", "multi_a:R2", "multi_a:R3"]
        budget = len(candidates) if max_l3 is None else max(0, max_l3)
        edges = {"multi_a:L2A": set(candidates[:budget])}
        papers = {
            rid: ci.IngestionPaper(paper_id=rid, title=rid, year=2018, citations=1)
            for rid in candidates[:budget]
        }
        return edges, papers


class _MultiProviderB(ci.CitationProvider):
    name = "multi_b"
    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=True)

    def fetch_seed_metadata(self, l1_papers):
        return {
            l1: ci.IngestionPaper(
                paper_id=l1,
                title="Seed B",
                source_ids={"multi_b": "SEED-B"},
            )
            for l1 in l1_papers
        }

    def fetch_citers_for_l1(self, l1_provider_id, max_results=None):
        papers = {
            "multi_b:L2B": ci.IngestionPaper(
                paper_id="multi_b:L2B", title="L2 B", year=2021, citations=4
            )
        }
        return papers, 1, "complete"

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        return {}, {}

    def fetch_l3_references(self, l2_paper_ids, max_l3=None):
        candidates = ["multi_b:R1", "multi_b:R2", "multi_b:R3"]
        budget = len(candidates) if max_l3 is None else max(0, max_l3)
        edges = {"multi_b:L2B": set(candidates[:budget])}
        papers = {
            rid: ci.IngestionPaper(paper_id=rid, title=rid, year=2019, citations=1)
            for rid in candidates[:budget]
        }
        return edges, papers


def test_ingest_l3_multi_provider_uncapped_aggregates_union(monkeypatch, tmp_path):
    providers = [_MultiProviderA(), _MultiProviderB()]
    monkeypatch.setattr(ci, "build_providers", lambda _: providers)

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["multi_a", "multi_b"],
        depth="l2l3",
        cache_dir=Path(tmp_path),
        refresh=True,
        exhaustive=True,
        max_l3=None,
    )

    assert result.metadata["provider_stats"]["multi_a"]["l3_edges"] == 3
    assert result.metadata["provider_stats"]["multi_b"]["l3_edges"] == 3

    l3_ids = {
        cited_id
        for cited_set in result.citation_data.values()
        for cited_id in cited_set
        if cited_id.startswith("multi_a:R") or cited_id.startswith("multi_b:R")
    }
    assert len(l3_ids) == 6


def test_ingest_l3_multi_provider_capped_is_deterministic(monkeypatch, tmp_path):
    providers = [_MultiProviderA(), _MultiProviderB()]
    monkeypatch.setattr(ci, "build_providers", lambda _: providers)

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["multi_a", "multi_b"],
        depth="l2l3",
        cache_dir=Path(tmp_path),
        refresh=True,
        exhaustive=True,
        max_l3=2,
    )

    # max_l3 is currently applied per provider in the orchestrator flow.
    assert result.metadata["provider_stats"]["multi_a"]["l3_edges"] == 2
    assert result.metadata["provider_stats"]["multi_b"]["l3_edges"] == 2

    l3_ids = {
        cited_id
        for cited_set in result.citation_data.values()
        for cited_id in cited_set
        if cited_id.startswith("multi_a:R") or cited_id.startswith("multi_b:R")
    }
    assert len(l3_ids) == 4


def test_dedupe_and_materialize_merges_l3_nodes_by_doi_and_preserves_sources():
    all_edges = {
        "semantic_scholar:l2": {"doi:10.1000/l3a"},
        "crossref:l2": {"crossref:10.1000/l3a"},
    }
    all_papers = {
        "semantic_scholar:l2": ci.IngestionPaper(
            paper_id="semantic_scholar:l2",
            title="Parent 1",
            source_ids={"semantic_scholar": "l2"},
        ),
        "crossref:l2": ci.IngestionPaper(
            paper_id="crossref:l2",
            title="Parent 2",
            source_ids={"crossref": "l2"},
        ),
        "doi:10.1000/l3a": ci.IngestionPaper(
            paper_id="doi:10.1000/l3a",
            doi="10.1000/l3a",
            source_ids={"semantic_scholar": "abc123"},
        ),
        "crossref:10.1000/l3a": ci.IngestionPaper(
            paper_id="crossref:10.1000/l3a",
            doi="10.1000/l3a",
            source_ids={"crossref": "10.1000/l3a"},
        ),
    }

    citation_out, papers_out, alias_to_final = ci._dedupe_and_materialize(all_edges, all_papers)

    assert alias_to_final["doi:10.1000/l3a"] == alias_to_final["crossref:10.1000/l3a"]
    final_l3 = alias_to_final["doi:10.1000/l3a"]
    assert papers_out[final_l3]["doi"] == "10.1000/l3a"
    assert papers_out[final_l3]["source_ids"] == {
        "semantic_scholar": "abc123",
        "crossref": "10.1000/l3a",
    }
    assert final_l3 in citation_out[alias_to_final["semantic_scholar:l2"]]
    assert final_l3 in citation_out[alias_to_final["crossref:l2"]]


# ---------------------------------------------------------------------------
# Verbose progress output
# ---------------------------------------------------------------------------


def test_default_progress_messages_are_emitted(monkeypatch, tmp_path, capsys):
    """Standard ingestion milestones should print to stderr by default."""
    provider = _FakeProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["fake"],
        depth="l2l3",
        cache_dir=Path(tmp_path),
        refresh=True,
        max_l2=10,
        max_l3=10,
    )
    captured = capsys.readouterr()

    assert "[ADIT] Ingesting theory" in captured.err
    assert "Provider 1/1" in captured.err
    assert (
        "L2 nodes collected (provider-local, pre-dedup)" in captured.err
        or "Total L2 nodes collected across all seeds (provider-local)" in captured.err
    )
    assert "Ingestion complete" in captured.err


def test_quiet_mode_suppresses_progress_and_verbose_output(monkeypatch, tmp_path, capsys):
    """Quiet mode should suppress both default progress and verbose diagnostics."""
    provider = _FakeProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["fake"],
        depth="l2l3",
        cache_dir=Path(tmp_path),
        refresh=True,
        max_l2=10,
        max_l3=10,
        verbose=True,
        quiet=True,
    )
    captured = capsys.readouterr()
    ci.set_quiet(False)

    assert captured.err == ""


def test_progress_inline_overwrites_when_tty(monkeypatch, capsys):
    """Inline progress should use carriage-return updates on interactive stderr."""
    ci.set_quiet(False)
    monkeypatch.setattr(ci.sys.stderr, "isatty", lambda: True, raising=False)

    ci._progress_inline("inline 1/3")
    ci._progress_inline("inline 2/3")
    ci._progress_done("inline complete")

    captured = capsys.readouterr()
    assert "\rinline 1/3" in captured.err
    assert "\rinline 2/3" in captured.err
    assert "inline complete" in captured.err


def test_progress_inline_falls_back_to_newlines_when_not_tty(monkeypatch, capsys):
    """Inline progress should degrade to durable newline output for non-TTY stderr."""
    ci.set_quiet(False)
    monkeypatch.setattr(ci.sys.stderr, "isatty", lambda: False, raising=False)

    ci._progress_inline("inline fallback")

    captured = capsys.readouterr()
    assert "inline fallback\n" in captured.err


def test_verbose_off_produces_no_output(monkeypatch, capsys):
    """_vprint and _countdown_sleep should produce no output when verbose is off."""
    ci.set_quiet(False)
    ci.set_verbose(False)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    ci._vprint("should not appear")
    ci._countdown_sleep(2.0, "label")
    captured = capsys.readouterr()
    assert captured.err == ""


def test_verbose_countdown_writes_ticks_to_stderr(monkeypatch, capsys):
    """_countdown_sleep should write countdown ticks to stderr and clear the line."""
    ci.set_quiet(False)
    ci.set_verbose(True)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    ci._countdown_sleep(2.0, "test-provider attempt 1/3")
    captured = capsys.readouterr()
    ci.set_verbose(False)
    assert "retrying in" in captured.err
    # Final clear sequence leaves line ending with \r
    assert captured.err.endswith("\r")


def test_verbose_safe_get_prints_retry_message(monkeypatch, capsys):
    """When verbose, _safe_get should print retry messages to stderr on transient failures."""
    call_count = [0]

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] < 2:
            raise _make_http_error(429)
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'
        return mock_resp

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    ci._reset_ingest_stats(["test"])
    ci.set_quiet(False)
    ci.set_verbose(True)

    result = ci._safe_get("https://example.com/test", provider="test", max_retries=3)
    captured = capsys.readouterr()
    ci.set_verbose(False)

    assert result == {"ok": True}
    assert "attempt" in captured.err
    assert "test" in captured.err


def test_verbose_safe_get_prints_permanent_failure(monkeypatch, capsys):
    """When verbose, _safe_get should print a skip message on permanent HTTP failures."""

    def fake_urlopen(req, timeout=None):
        raise _make_http_error(404)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    ci._reset_ingest_stats(["test"])
    ci.set_quiet(False)
    ci.set_verbose(True)

    result = ci._safe_get("https://example.com/not-found", provider="test")
    captured = capsys.readouterr()
    ci.set_verbose(False)

    assert result is None
    assert "404" in captured.err
    assert "skipping" in captured.err.lower()


def test_safe_get_emits_periodic_failure_summary(monkeypatch, capsys):
    """Periodic failure summaries should emit aggregate counts in long runs."""

    def fake_urlopen(req, timeout=None):
        raise _make_http_error(404)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci, "_FAILURE_SUMMARY_REQUEST_INTERVAL", 1)
    monkeypatch.setattr(ci, "_FAILURE_SUMMARY_SECONDS_INTERVAL", 9999.0)
    ci._reset_ingest_stats(["test"])
    ci.set_quiet(False)
    ci.set_verbose(False)

    ci._safe_get("https://example.com/not-found", provider="test")
    captured = capsys.readouterr()

    assert "HTTP failures so far" in captured.err
    assert "404=1" in captured.err


# ---------------------------------------------------------------------------
# Retry-After header handling tests
# ---------------------------------------------------------------------------


def test_safe_get_429_retry_after_seconds_header(monkeypatch):
    """_safe_get should honor Retry-After header with integer seconds on 429."""
    call_count = [0]
    sleep_calls = []

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] < 3:
            err = urllib.error.HTTPError(
                url="https://example.com",
                code=429,
                msg="Too Many Requests",
                hdrs={"Retry-After": "7"},
                fp=None,
            )
            raise err
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'
        return mock_resp

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci.time, "sleep", fake_sleep)
    ci._reset_ingest_stats(["test"])

    result = ci._safe_get("https://example.com/test", provider="test")

    assert result == {"ok": True}
    assert call_count[0] == 3
    # Both retries should use 7s from Retry-After header
    assert sleep_calls[0] == 7  # Actually _countdown_sleep, but monkeypatch handles both


def test_safe_get_429_retry_after_respects_cap(monkeypatch):
    """_safe_get should cap Retry-After at _SAFE_GET_RETRY_AFTER_MAX_SECONDS."""
    call_count = [0]
    sleep_calls = []

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] < 2:
            # Server requests 999 seconds, but we should cap at 300
            err = urllib.error.HTTPError(
                url="https://example.com",
                code=429,
                msg="Too Many Requests",
                hdrs={"Retry-After": "999"},
                fp=None,
            )
            raise err
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'
        return mock_resp

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci.time, "sleep", fake_sleep)
    ci._reset_ingest_stats(["test"])

    result = ci._safe_get("https://example.com/test", provider="test")

    assert result == {"ok": True}
    # Should use capped value (300 seconds), not the 999 requested
    assert sleep_calls[0] == ci._SAFE_GET_RETRY_AFTER_MAX_SECONDS


def test_safe_get_429_retry_after_invalid_falls_back(monkeypatch):
    """_safe_get should fall back to exponential backoff when Retry-After is invalid."""
    call_count = [0]
    sleep_calls = []

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] < 3:
            # Invalid header format
            err = urllib.error.HTTPError(
                url="https://example.com",
                code=429,
                msg="Too Many Requests",
                hdrs={"Retry-After": "not-a-valid-value"},
                fp=None,
            )
            raise err
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'
        return mock_resp

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci.time, "sleep", fake_sleep)
    ci._reset_ingest_stats(["test"])

    result = ci._safe_get("https://example.com/test", provider="test")

    assert result == {"ok": True}
    # Should use exponential backoff: 1.0 to 1.2 (first attempt), ~2.0-2.4 (second)
    # Just verify that it's not the invalid header value and retried successfully
    assert call_count[0] == 3


def test_safe_get_non_429_uses_exponential_backoff(monkeypatch):
    """_safe_get should use exponential backoff for 5xx errors, not Retry-After."""
    call_count = [0]
    sleep_calls = []

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] < 3:
            # 503 Service Unavailable should use exponential backoff
            raise _make_http_error(503)
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'
        return mock_resp

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci.time, "sleep", fake_sleep)
    ci._reset_ingest_stats(["test"])

    result = ci._safe_get("https://example.com/test", provider="test")

    assert result == {"ok": True}
    # First retry: 1.0-1.2s range (with jitter)
    # Second retry: 2.0-2.4s range (with jitter)
    # Just verify retries happened and weren't instant
    assert len(sleep_calls) == 2
    assert all(s > 0 for s in sleep_calls)


# ---------------------------------------------------------------------------
# Resumable checkpoints (Phase 1: provider-atomic)
# ---------------------------------------------------------------------------


class _CheckpointProvider(ci.CitationProvider):
    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=False)

    def __init__(
        self,
        name: str,
        edge_id: str,
        crash: bool = False,
        fail_on_call: bool = False,
    ):
        self.name = name
        self.edge_id = edge_id
        self.crash = crash
        self.fail_on_call = fail_on_call
        self.seed_calls = 0
        self.l2_calls = 0

    def fetch_seed_metadata(self, l1_papers):
        if self.fail_on_call:
            raise AssertionError(f"{self.name} should have been skipped via checkpoint")
        self.seed_calls += 1
        return {
            l1_papers[0]: ci.IngestionPaper(
                paper_id=l1_papers[0],
                title=f"Seed from {self.name}",
                source_ids={self.name: f"seed-{self.name}"},
            )
        }

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        if self.fail_on_call:
            raise AssertionError(f"{self.name} should have been skipped via checkpoint")
        self.l2_calls += 1
        if self.crash:
            raise RuntimeError(f"simulated crash in {self.name}")
        return (
            {self.edge_id: {l1_papers[0]}},
            {
                self.edge_id: ci.IngestionPaper(
                    paper_id=self.edge_id,
                    title=f"{self.name} L2",
                    year=2020,
                    citations=1,
                )
            },
        )

    def fetch_l3_references(self, l2_paper_ids, max_l3=None):
        return {}, {}


class _TransientRetryCheckpointProvider(ci.CitationProvider):
    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=False)

    def __init__(self, name: str, edge_id: str):
        self.name = name
        self.edge_id = edge_id

    def fetch_seed_metadata(self, l1_papers):
        return {
            l1_papers[0]: ci.IngestionPaper(
                paper_id=l1_papers[0],
                title=f"Seed from {self.name}",
                source_ids={self.name: f"seed-{self.name}"},
            )
        }

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        result = ci._safe_get(
            "https://example.com/transient-retry",
            provider=self.name,
            max_retries=1,
        )
        if result is None:
            return {}, {}
        return (
            {self.edge_id: {l1_papers[0]}},
            {
                self.edge_id: ci.IngestionPaper(
                    paper_id=self.edge_id,
                    title=f"{self.name} L2",
                    year=2020,
                    citations=1,
                )
            },
        )

    def fetch_l3_references(self, l2_paper_ids, max_l3=None):
        return {}, {}


class _TransientL3RetryProvider(ci.CitationProvider):
    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=False)

    def __init__(self, name: str):
        self.name = name

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
        return (
            {
                f"{self.name}:L2A": {seed},
                f"{self.name}:L2B": {seed},
            },
            {
                f"{self.name}:L2A": ci.IngestionPaper(
                    paper_id=f"{self.name}:L2A",
                    title="L2 A",
                    year=2020,
                    source_ids={self.name: "l2a"},
                ),
                f"{self.name}:L2B": ci.IngestionPaper(
                    paper_id=f"{self.name}:L2B",
                    title="L2 B",
                    year=2021,
                    source_ids={self.name: "l2b"},
                ),
            },
        )

    def fetch_l3_references(
        self, l2_paper_ids, max_l3=None, resume_state=None, progress_callback=None
    ):
        edges = ci._deserialize_edges((resume_state or {}).get("edges"))
        papers = ci._deserialize_papers((resume_state or {}).get("papers"))
        start_index = (
            ci._parse_optional_int((resume_state or {}).get("next_l2_index"), default=0) or 0
        )

        for idx in range(start_index, len(l2_paper_ids)):
            parent_id = l2_paper_ids[idx]
            payload = ci._safe_get(
                f"https://example.com/l3/{parent_id}",
                provider=self.name,
                max_retries=1,
            )
            if payload and payload.get("ref"):
                ref_id = ci.normalize_identifier(str(payload["ref"]))
                edges.setdefault(parent_id, set()).add(ref_id)
                papers.setdefault(
                    ref_id,
                    ci.IngestionPaper(
                        paper_id=ref_id,
                        title=f"Ref {ref_id}",
                        citations=1,
                        year=2018,
                        doi=ref_id.split(":", 1)[1] if ref_id.startswith("doi:") else None,
                    ),
                )

            ci._emit_traversal_progress(
                progress_callback=progress_callback,
                status="in_progress",
                index_key="next_l2_index",
                index_value=idx + 1,
                budget_remaining=None,
                edges=edges,
                papers=papers,
            )

        ci._emit_traversal_progress(
            progress_callback=progress_callback,
            status="complete",
            index_key="next_l2_index",
            index_value=len(l2_paper_ids),
            budget_remaining=None,
            edges=edges,
            papers=papers,
        )
        return edges, papers


def test_checkpoint_resume_skips_completed_provider(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"

    provider_run1 = _CheckpointProvider("checkpoint_a", "checkpoint_a:L2")
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider_run1])

    result1 = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["checkpoint_a"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )
    assert provider_run1.seed_calls == 1
    assert provider_run1.l2_calls == 1

    checkpoint_file = checkpoint_dir / f"{result1.metadata['cache_key']}.checkpoint.json"
    assert checkpoint_file.exists()

    provider_run2 = _CheckpointProvider(
        "checkpoint_a",
        "checkpoint_a:L2",
        fail_on_call=True,
    )
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider_run2])

    result2 = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["checkpoint_a"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    assert result2.citation_data == result1.citation_data
    assert result2.papers_data == result1.papers_data


def test_checkpoint_reset_forces_refetch(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    provider = _CheckpointProvider("checkpoint_reset", "checkpoint_reset:L2")
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["checkpoint_reset"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )
    assert provider.l2_calls == 1

    ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["checkpoint_reset"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
        reset_checkpoints=True,
    )
    assert provider.l2_calls == 2


def test_checkpoint_corruption_is_ignored(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    provider = _CheckpointProvider("checkpoint_corrupt", "checkpoint_corrupt:L2")
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    request_payload = ci._request_payload(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        key_constructs=None,
        sources=["checkpoint_corrupt"],
        depth="l2",
        max_l2=200,
        max_l3=None,
        exhaustive=True,
    )
    key = ci._cache_key(request_payload)
    checkpoint_path = checkpoint_dir / f"{key}.checkpoint.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("not valid json", encoding="utf-8")

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["checkpoint_corrupt"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    assert result.metadata["provider_stats"]["checkpoint_corrupt"]["l2_nodes"] == 1
    assert provider.l2_calls == 1


def test_checkpoint_resume_retries_and_clears_transient_failures(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    provider_name = "checkpoint_transient"
    edge_id = "checkpoint_transient:L2"
    provider = _TransientRetryCheckpointProvider(provider_name, edge_id)
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    fail_once = {"value": True}

    def fake_urlopen(req, timeout=None):
        if fail_once["value"]:
            fail_once["value"] = False
            raise urllib.error.HTTPError(
                url="https://example.com/transient-retry",
                code=500,
                msg="temporary server error",
                hdrs=None,
                fp=None,
            )
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'
        return mock_resp

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    clock = {"value": 1_000.0}
    monkeypatch.setattr(ci.time, "time", lambda: clock["value"])
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    first = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=[provider_name],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    key = first.metadata["cache_key"]
    first_state = ci._load_checkpoint_state(checkpoint_dir, key, reset_checkpoints=False)
    assert first_state is not None
    assert provider_name in first_state.get("provider_transient_failures", {})
    assert first.metadata["checkpoint_stats"]["transient_failures_queued"] >= 1

    clock["value"] += 120.0

    resumed = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=[provider_name],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    resumed_state = ci._load_checkpoint_state(checkpoint_dir, key, reset_checkpoints=False)
    assert resumed_state is not None
    assert resumed_state.get("provider_transient_failures", {}).get(provider_name, {}) == {}
    assert resumed.metadata["checkpoint_stats"]["transient_failures_retried"] >= 1
    assert resumed.metadata["checkpoint_stats"]["transient_failures_resumed_success"] >= 1
    assert edge_id in resumed.citation_data


def test_l2_to_l3_transient_failures_retry_on_resume(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    provider_name = "checkpoint_l3_transient"
    provider = _TransientL3RetryProvider(provider_name)
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    fail_l2b_once = {"value": True}

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if url.endswith(f"{provider_name}:L2A"):
            body = b'{"ref": "10.1000/l3a"}'
        elif url.endswith(f"{provider_name}:L2B"):
            if fail_l2b_once["value"]:
                fail_l2b_once["value"] = False
                raise urllib.error.HTTPError(
                    url=url,
                    code=500,
                    msg="temporary l3 failure",
                    hdrs=None,
                    fp=None,
                )
            body = b'{"ref": "10.1000/l3b"}'
        else:
            body = b'{"ok": true}'

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = body
        return mock_resp

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    clock = {"value": 2_000.0}
    monkeypatch.setattr(ci.time, "time", lambda: clock["value"])
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    first = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=[provider_name],
        depth="l2l3",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    key = first.metadata["cache_key"]
    first_state = ci._load_checkpoint_state(checkpoint_dir, key, reset_checkpoints=False)
    assert first_state is not None
    assert provider_name in first_state.get("provider_transient_failures", {})

    clock["value"] += 120.0
    resumed = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=[provider_name],
        depth="l2l3",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    resumed_state = ci._load_checkpoint_state(checkpoint_dir, key, reset_checkpoints=False)
    assert resumed_state is not None
    assert resumed_state.get("provider_transient_failures", {}).get(provider_name, {}) == {}
    assert (
        resumed.metadata.get("transient_failure_summary", {})
        .get(provider_name, {})
        .get("queued", 0)
        == 0
    )

    all_targets = {target for values in resumed.citation_data.values() for target in values}
    assert "doi:10.1000/l3a" in all_targets
    assert "doi:10.1000/l3b" in all_targets


def test_checkpoint_crash_resume_matches_uninterrupted_baseline(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"

    provider_a_first = _CheckpointProvider("checkpoint_a", "checkpoint_a:L2")
    provider_b_crash = _CheckpointProvider("checkpoint_b", "checkpoint_b:L2", crash=True)
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider_a_first, provider_b_crash])

    with pytest.raises(RuntimeError, match="simulated crash"):
        ci.ingest_from_internet(
            theory_name="My Fake Theory",
            l1_papers=["10.1000/xyz1"],
            sources=["checkpoint_a", "checkpoint_b"],
            depth="l2",
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            refresh=True,
        )

    provider_a_resume = _CheckpointProvider(
        "checkpoint_a",
        "checkpoint_a:L2",
        fail_on_call=True,
    )
    provider_b_resume = _CheckpointProvider("checkpoint_b", "checkpoint_b:L2")
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider_a_resume, provider_b_resume])

    resumed = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["checkpoint_a", "checkpoint_b"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    baseline_a = _CheckpointProvider("checkpoint_a", "checkpoint_a:L2")
    baseline_b = _CheckpointProvider("checkpoint_b", "checkpoint_b:L2")
    monkeypatch.setattr(ci, "build_providers", lambda _: [baseline_a, baseline_b])

    baseline = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["checkpoint_a", "checkpoint_b"],
        depth="l2",
        cache_dir=Path(tmp_path) / "baseline-cache",
        checkpoint_dir=Path(tmp_path) / "baseline-checkpoints",
        refresh=True,
    )

    assert resumed.citation_data == baseline.citation_data
    assert resumed.papers_data == baseline.papers_data
    assert resumed.metadata["provider_stats"] == baseline.metadata["provider_stats"]


def test_checkpoint_stats_report_hit_miss_and_provider_skip(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"

    provider_run1 = _CheckpointProvider("checkpoint_stats", "checkpoint_stats:L2")
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider_run1])

    first = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["checkpoint_stats"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    first_stats = first.metadata["checkpoint_stats"]
    assert first_stats["hit"] is False
    assert first_stats["miss"] is True
    assert first_stats["providers_executed"] == 1
    assert first_stats["executed_provider_names"] == ["checkpoint_stats"]
    assert first_stats["providers_skipped"] == 0

    provider_run2 = _CheckpointProvider(
        "checkpoint_stats",
        "checkpoint_stats:L2",
        fail_on_call=True,
    )
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider_run2])

    second = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["checkpoint_stats"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    second_stats = second.metadata["checkpoint_stats"]
    assert second_stats["hit"] is True
    assert second_stats["miss"] is False
    assert second_stats["providers_skipped"] == 1
    assert second_stats["skipped_provider_names"] == ["checkpoint_stats"]
    assert second_stats["providers_executed"] == 0


def test_checkpoint_stats_marks_cache_short_circuit(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    provider = _CheckpointProvider("checkpoint_cache", "checkpoint_cache:L2")
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["checkpoint_cache"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=False,
    )

    cached = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["checkpoint_cache"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=False,
    )

    cached_stats = cached.metadata["checkpoint_stats"]
    assert cached_stats["cache_short_circuit"] is True
    assert cached_stats["providers_executed"] == 0


def test_openalex_mid_pagination_resume_from_checkpoint(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    crash_on_second_page = {"enabled": True}

    def fake_safe_get(
        url, timeout=20, provider=None, max_retries=ci._SAFE_GET_MAX_RETRIES, headers=None
    ):
        if url.endswith("/works/WSEED"):
            return {
                "id": "https://openalex.org/WSEED",
                "title": "Seed",
                "publication_year": 2000,
                "cited_by_count": 10,
                "doi": "10.1000/seed",
                "abstract_inverted_index": {},
            }

        if "api.openalex.org/works?" in url:
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            if params.get("filter", [""])[0] != "cites:WSEED":
                return None
            cursor = params.get("cursor", ["*"])[0]

            if cursor == "*":
                return {
                    "meta": {"count": 2, "next_cursor": "C2"},
                    "results": [
                        {
                            "id": "https://openalex.org/WCITER1",
                            "title": "Citer 1",
                            "publication_year": 2020,
                            "cited_by_count": 1,
                            "doi": "10.1000/citer1",
                            "abstract_inverted_index": {},
                        }
                    ],
                }

            if cursor == "C2":
                if crash_on_second_page["enabled"]:
                    raise RuntimeError("openalex simulated page crash")
                return {
                    "meta": {"count": 2, "next_cursor": None},
                    "results": [
                        {
                            "id": "https://openalex.org/WCITER2",
                            "title": "Citer 2",
                            "publication_year": 2021,
                            "cited_by_count": 1,
                            "doi": "10.1000/citer2",
                            "abstract_inverted_index": {},
                        }
                    ],
                }

        return None

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="openalex simulated page crash"):
        ci.ingest_from_internet(
            theory_name="My Fake Theory",
            l1_papers=["openalex:WSEED"],
            sources=["openalex"],
            depth="l2",
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            refresh=True,
        )

    request_payload = ci._request_payload(
        theory_name="My Fake Theory",
        l1_papers=["openalex:WSEED"],
        key_constructs=None,
        sources=["openalex"],
        depth="l2",
        max_l2=200,
        max_l3=None,
        exhaustive=True,
    )
    key = ci._cache_key(request_payload)
    state = ci._load_checkpoint_state(checkpoint_dir, key, reset_checkpoints=False)
    assert state is not None
    assert state["provider_pagination_state"]["openalex"]["openalex:WSEED"]["cursor"] == "C2"

    crash_on_second_page["enabled"] = False
    resumed = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["openalex:WSEED"],
        sources=["openalex"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    baseline = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["openalex:WSEED"],
        sources=["openalex"],
        depth="l2",
        cache_dir=Path(tmp_path) / "baseline-cache",
        checkpoint_dir=Path(tmp_path) / "baseline-checkpoints",
        refresh=True,
    )

    assert resumed.citation_data == baseline.citation_data
    assert resumed.papers_data == baseline.papers_data
    assert resumed.metadata["checkpoint_stats"]["hit"] is True


def test_semantic_mid_pagination_resume_from_checkpoint(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    crash_on_second_page = {"enabled": True}

    def fake_safe_get(
        url, timeout=20, provider=None, max_retries=ci._SAFE_GET_MAX_RETRIES, headers=None
    ):
        if "/graph/v1/paper/seed1?fields=" in url:
            return {
                "paperId": "seed1",
                "title": "Seed",
                "abstract": "",
                "year": 2000,
                "citationCount": 0,
                "externalIds": {},
            }

        if "/graph/v1/paper/seed1/citations" in url:
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            offset = int(params.get("offset", ["0"])[0])

            if offset == 0:
                return {
                    "total": 2,
                    "next": "has-more",
                    "data": [
                        {
                            "citingPaper": {
                                "paperId": "citer-1",
                                "title": "Citer 1",
                                "year": 2020,
                                "citationCount": 1,
                                "externalIds": {},
                                "abstract": "",
                            }
                        }
                    ],
                }

            if offset == 1:
                if crash_on_second_page["enabled"]:
                    raise RuntimeError("semantic simulated page crash")
                return {
                    "total": 2,
                    "data": [
                        {
                            "citingPaper": {
                                "paperId": "citer-2",
                                "title": "Citer 2",
                                "year": 2021,
                                "citationCount": 1,
                                "externalIds": {},
                                "abstract": "",
                            }
                        }
                    ],
                }

        return None

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="semantic simulated page crash"):
        ci.ingest_from_internet(
            theory_name="My Fake Theory",
            l1_papers=["semantic_scholar:seed1"],
            sources=["semantic_scholar"],
            depth="l2",
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            refresh=True,
        )

    request_payload = ci._request_payload(
        theory_name="My Fake Theory",
        l1_papers=["semantic_scholar:seed1"],
        key_constructs=None,
        sources=["semantic_scholar"],
        depth="l2",
        max_l2=200,
        max_l3=None,
        exhaustive=True,
    )
    key = ci._cache_key(request_payload)
    state = ci._load_checkpoint_state(checkpoint_dir, key, reset_checkpoints=False)
    assert state is not None
    assert (
        state["provider_pagination_state"]["semantic_scholar"]["semantic_scholar:seed1"]["offset"]
        == 1
    )

    crash_on_second_page["enabled"] = False
    resumed = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["semantic_scholar:seed1"],
        sources=["semantic_scholar"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    baseline = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["semantic_scholar:seed1"],
        sources=["semantic_scholar"],
        depth="l2",
        cache_dir=Path(tmp_path) / "baseline-cache",
        checkpoint_dir=Path(tmp_path) / "baseline-checkpoints",
        refresh=True,
    )

    assert resumed.citation_data == baseline.citation_data
    assert resumed.papers_data == baseline.papers_data
    assert resumed.metadata["checkpoint_stats"]["hit"] is True


def test_openalex_l3_resume_from_checkpoint(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    crash_on_l3_second_parent = {"enabled": True}

    def fake_safe_get(
        url, timeout=20, provider=None, max_retries=ci._SAFE_GET_MAX_RETRIES, headers=None
    ):
        if url.endswith("/works/WSEED"):
            return {
                "id": "https://openalex.org/WSEED",
                "title": "Seed",
                "publication_year": 2000,
                "cited_by_count": 10,
                "doi": "10.1000/seed",
                "abstract_inverted_index": {},
            }

        if "api.openalex.org/works?" in url:
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            if params.get("filter", [""])[0] != "cites:WSEED":
                return None
            return {
                "meta": {"count": 2, "next_cursor": None},
                "results": [
                    {
                        "id": "https://openalex.org/WL2A",
                        "title": "L2 A",
                        "publication_year": 2020,
                        "cited_by_count": 1,
                        "doi": "10.1000/l2a",
                        "abstract_inverted_index": {},
                    },
                    {
                        "id": "https://openalex.org/WL2B",
                        "title": "L2 B",
                        "publication_year": 2021,
                        "cited_by_count": 1,
                        "doi": "10.1000/l2b",
                        "abstract_inverted_index": {},
                    },
                ],
            }

        if url.endswith("/works/WL2A?select=referenced_works"):
            return {"referenced_works": ["https://openalex.org/WRA1"]}

        if url.endswith("/works/WL2B?select=referenced_works"):
            if crash_on_l3_second_parent["enabled"]:
                raise RuntimeError("openalex l3 simulated crash")
            return {"referenced_works": ["https://openalex.org/WRB1"]}

        if url.endswith("/works/WRA1?select=id,doi,title,publication_year"):
            return {
                "id": "https://openalex.org/WRA1",
                "doi": "https://doi.org/10.1000/l3a",
                "title": "L3 A",
                "publication_year": 2018,
            }

        if url.endswith("/works/WRB1?select=id,doi,title,publication_year"):
            return {
                "id": "https://openalex.org/WRB1",
                "doi": "https://doi.org/10.1000/l3b",
                "title": "L3 B",
                "publication_year": 2019,
            }

        return None

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="openalex l3 simulated crash"):
        ci.ingest_from_internet(
            theory_name="My Fake Theory",
            l1_papers=["openalex:WSEED"],
            sources=["openalex"],
            depth="l2l3",
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            refresh=True,
        )

    request_payload = ci._request_payload(
        theory_name="My Fake Theory",
        l1_papers=["openalex:WSEED"],
        key_constructs=None,
        sources=["openalex"],
        depth="l2l3",
        max_l2=200,
        max_l3=None,
        exhaustive=True,
    )
    key = ci._cache_key(request_payload)
    state = ci._load_checkpoint_state(checkpoint_dir, key, reset_checkpoints=False)
    assert state is not None
    assert state["provider_l3_state"]["openalex"]["next_l2_index"] == 1

    crash_on_l3_second_parent["enabled"] = False
    resumed = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["openalex:WSEED"],
        sources=["openalex"],
        depth="l2l3",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    baseline = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["openalex:WSEED"],
        sources=["openalex"],
        depth="l2l3",
        cache_dir=Path(tmp_path) / "baseline-cache",
        checkpoint_dir=Path(tmp_path) / "baseline-checkpoints",
        refresh=True,
    )

    assert resumed.citation_data == baseline.citation_data
    assert resumed.papers_data == baseline.papers_data
    assert "openalex" in resumed.metadata["checkpoint_stats"]["l3_resumed_providers"]


def test_semantic_l3_resume_from_checkpoint(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    crash_on_l3_second_parent = {"enabled": True}

    def fake_safe_get(
        url, timeout=20, provider=None, max_retries=ci._SAFE_GET_MAX_RETRIES, headers=None
    ):
        if "/graph/v1/paper/seed1?fields=" in url:
            return {
                "paperId": "seed1",
                "title": "Seed",
                "abstract": "",
                "year": 2000,
                "citationCount": 0,
                "externalIds": {},
            }

        if "/graph/v1/paper/seed1/citations" in url:
            return {
                "total": 2,
                "data": [
                    {
                        "citingPaper": {
                            "paperId": "l2a",
                            "title": "L2 A",
                            "year": 2020,
                            "citationCount": 1,
                            "externalIds": {},
                            "abstract": "",
                        }
                    },
                    {
                        "citingPaper": {
                            "paperId": "l2b",
                            "title": "L2 B",
                            "year": 2021,
                            "citationCount": 1,
                            "externalIds": {},
                            "abstract": "",
                        }
                    },
                ],
            }

        if "/graph/v1/paper/l2a?fields=references.paperId,references.externalIds" in url:
            return {
                "references": [
                    {
                        "paperId": "ref-a",
                        "externalIds": {"DOI": "10.1000/l3a"},
                    }
                ]
            }

        if "/graph/v1/paper/l2b?fields=references.paperId,references.externalIds" in url:
            if crash_on_l3_second_parent["enabled"]:
                raise RuntimeError("semantic l3 simulated crash")
            return {
                "references": [
                    {
                        "paperId": "ref-b",
                        "externalIds": {"DOI": "10.1000/l3b"},
                    }
                ]
            }

        return None

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="semantic l3 simulated crash"):
        ci.ingest_from_internet(
            theory_name="My Fake Theory",
            l1_papers=["semantic_scholar:seed1"],
            sources=["semantic_scholar"],
            depth="l2l3",
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            refresh=True,
        )

    request_payload = ci._request_payload(
        theory_name="My Fake Theory",
        l1_papers=["semantic_scholar:seed1"],
        key_constructs=None,
        sources=["semantic_scholar"],
        depth="l2l3",
        max_l2=200,
        max_l3=None,
        exhaustive=True,
    )
    key = ci._cache_key(request_payload)
    state = ci._load_checkpoint_state(checkpoint_dir, key, reset_checkpoints=False)
    assert state is not None
    assert state["provider_l3_state"]["semantic_scholar"]["next_l2_index"] == 1

    crash_on_l3_second_parent["enabled"] = False
    resumed = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["semantic_scholar:seed1"],
        sources=["semantic_scholar"],
        depth="l2l3",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    baseline = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["semantic_scholar:seed1"],
        sources=["semantic_scholar"],
        depth="l2l3",
        cache_dir=Path(tmp_path) / "baseline-cache",
        checkpoint_dir=Path(tmp_path) / "baseline-checkpoints",
        refresh=True,
    )

    assert resumed.citation_data == baseline.citation_data
    assert resumed.papers_data == baseline.papers_data
    assert "semantic_scholar" in resumed.metadata["checkpoint_stats"]["l3_resumed_providers"]


def test_core_l3_resume_from_checkpoint(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    crash_on_l3_second_parent = {"enabled": True}

    class _CoreL3ResumeProvider(ci.CoreProvider):
        name = "core"

        def __init__(self):
            super().__init__(api_key=None)

        def fetch_seed_metadata(self, l1_papers):
            return {
                l1: ci.IngestionPaper(
                    paper_id=l1,
                    title="Seed",
                    source_ids={"core": "seed-core-id"},
                )
                for l1 in l1_papers
            }

        def fetch_citers_for_l1(
            self,
            l1_provider_id,
            max_results=None,
            resume_state=None,
            progress_callback=None,
        ):
            papers = {
                "core:l2a": ci.IngestionPaper(
                    paper_id="core:l2a",
                    title="L2 A",
                    year=2020,
                    source_ids={"core": "l2a"},
                ),
                "core:l2b": ci.IngestionPaper(
                    paper_id="core:l2b",
                    title="L2 B",
                    year=2021,
                    source_ids={"core": "l2b"},
                ),
            }
            return papers, 2, "complete"

        def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
            seed = ci.normalize_identifier(l1_papers[0])
            edges = {
                "core:l2a": {seed},
                "core:l2b": {seed},
            }
            papers = {
                "core:l2a": ci.IngestionPaper(
                    paper_id="core:l2a",
                    title="L2 A",
                    year=2020,
                    source_ids={"core": "l2a"},
                ),
                "core:l2b": ci.IngestionPaper(
                    paper_id="core:l2b",
                    title="L2 B",
                    year=2021,
                    source_ids={"core": "l2b"},
                ),
            }
            return edges, papers

        def _lookup_work(self, paper_id):
            if paper_id == "core:l2a":
                return {
                    "references": [
                        {
                            "doi": "10.1000/corel3a",
                            "id": "ref-a",
                            "title": "Core L3 A",
                        }
                    ]
                }
            if paper_id == "core:l2b":
                if crash_on_l3_second_parent["enabled"]:
                    raise RuntimeError("core l3 simulated crash")
                return {
                    "references": [
                        {
                            "doi": "10.1000/corel3b",
                            "id": "ref-b",
                            "title": "Core L3 B",
                        }
                    ]
                }
            return None

    provider = _CoreL3ResumeProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="core l3 simulated crash"):
        ci.ingest_from_internet(
            theory_name="My Fake Theory",
            l1_papers=["doi:10.1000/seed"],
            sources=["core"],
            depth="l2l3",
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            refresh=True,
        )

    request_payload = ci._request_payload(
        theory_name="My Fake Theory",
        l1_papers=["doi:10.1000/seed"],
        key_constructs=None,
        sources=["core"],
        depth="l2l3",
        max_l2=200,
        max_l3=None,
        exhaustive=True,
    )
    key = ci._cache_key(request_payload)
    state = ci._load_checkpoint_state(checkpoint_dir, key, reset_checkpoints=False)
    assert state is not None
    assert state["provider_l3_state"]["core"]["next_l2_index"] == 1

    crash_on_l3_second_parent["enabled"] = False
    resumed = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["doi:10.1000/seed"],
        sources=["core"],
        depth="l2l3",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    baseline_provider = _CoreL3ResumeProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [baseline_provider])
    baseline = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["doi:10.1000/seed"],
        sources=["core"],
        depth="l2l3",
        cache_dir=Path(tmp_path) / "baseline-cache",
        checkpoint_dir=Path(tmp_path) / "baseline-checkpoints",
        refresh=True,
    )

    assert resumed.citation_data == baseline.citation_data
    assert resumed.papers_data == baseline.papers_data
    assert "core" in resumed.metadata["checkpoint_stats"]["l3_resumed_providers"]


def test_core_l3_budget_resume_continues_remaining_budget(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    crash_on_l3_second_parent = {"enabled": True}

    class _CoreBudgetResumeProvider(ci.CoreProvider):
        name = "core"

        def __init__(self):
            super().__init__(api_key=None)

        def fetch_seed_metadata(self, l1_papers):
            return {
                l1: ci.IngestionPaper(
                    paper_id=l1,
                    title="Seed",
                    source_ids={"core": "seed-core-id"},
                )
                for l1 in l1_papers
            }

        def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
            seed = ci.normalize_identifier(l1_papers[0])
            edges = {
                "core:l2a": {seed},
                "core:l2b": {seed},
            }
            papers = {
                "core:l2a": ci.IngestionPaper(
                    paper_id="core:l2a",
                    title="L2 A",
                    year=2020,
                    source_ids={"core": "l2a"},
                ),
                "core:l2b": ci.IngestionPaper(
                    paper_id="core:l2b",
                    title="L2 B",
                    year=2021,
                    source_ids={"core": "l2b"},
                ),
            }
            return edges, papers

        def _lookup_work(self, paper_id):
            if paper_id == "core:l2a":
                return {
                    "references": [
                        {"doi": "10.1000/corea1", "id": "a1", "title": "Core A1"},
                        {"doi": "10.1000/corea2", "id": "a2", "title": "Core A2"},
                    ]
                }
            if paper_id == "core:l2b":
                if crash_on_l3_second_parent["enabled"]:
                    raise RuntimeError("core l3 budget simulated crash")
                return {
                    "references": [
                        {"doi": "10.1000/coreb1", "id": "b1", "title": "Core B1"},
                        {"doi": "10.1000/coreb2", "id": "b2", "title": "Core B2"},
                    ]
                }
            return None

    provider = _CoreBudgetResumeProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="core l3 budget simulated crash"):
        ci.ingest_from_internet(
            theory_name="My Fake Theory",
            l1_papers=["doi:10.1000/seed"],
            sources=["core"],
            depth="l2l3",
            max_l3=3,
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            refresh=True,
        )

    request_payload = ci._request_payload(
        theory_name="My Fake Theory",
        l1_papers=["doi:10.1000/seed"],
        key_constructs=None,
        sources=["core"],
        depth="l2l3",
        max_l2=200,
        max_l3=3,
        exhaustive=True,
    )
    key = ci._cache_key(request_payload)
    state = ci._load_checkpoint_state(checkpoint_dir, key, reset_checkpoints=False)
    assert state is not None
    assert state["provider_l3_state"]["core"]["next_l2_index"] == 1
    assert state["provider_l3_state"]["core"]["budget_remaining"] == 1

    crash_on_l3_second_parent["enabled"] = False
    resumed = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["doi:10.1000/seed"],
        sources=["core"],
        depth="l2l3",
        max_l3=3,
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    baseline_provider = _CoreBudgetResumeProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [baseline_provider])
    baseline = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["doi:10.1000/seed"],
        sources=["core"],
        depth="l2l3",
        max_l3=3,
        cache_dir=Path(tmp_path) / "baseline-cache",
        checkpoint_dir=Path(tmp_path) / "baseline-checkpoints",
        refresh=True,
    )

    assert resumed.citation_data == baseline.citation_data
    assert resumed.papers_data == baseline.papers_data
    assert resumed.metadata["provider_stats"]["core"]["l3_edges"] == 3
    assert "core" in resumed.metadata["checkpoint_stats"]["l3_resumed_providers"]


class _ResumeCaptureProvider(ci.CitationProvider):
    name = "resume_capture"
    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=True)

    def __init__(self):
        self.seen_resume_states = []

    def fetch_seed_metadata(self, l1_papers):
        return {
            l1: ci.IngestionPaper(
                paper_id=l1,
                title="Seed",
                source_ids={self.name: "SEED-1"},
            )
            for l1 in l1_papers
        }

    def fetch_citers_for_l1(
        self,
        l1_provider_id,
        max_results=None,
        resume_state=None,
        progress_callback=None,
    ):
        self.seen_resume_states.append(resume_state)
        return (
            {
                "resume_capture:C1": ci.IngestionPaper(
                    paper_id="resume_capture:C1",
                    title="Citer",
                    year=2022,
                    citations=1,
                )
            },
            1,
            "complete",
        )

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        return {}, {}

    def fetch_l3_references(self, l2_paper_ids, max_l3=None):
        return {}, {}


def test_stale_pagination_state_is_ignored(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    provider = _ResumeCaptureProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    request_payload = ci._request_payload(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        key_constructs=None,
        sources=["resume_capture"],
        depth="l2",
        max_l2=200,
        max_l3=None,
        exhaustive=True,
    )
    key = ci._cache_key(request_payload)
    stale_ts = time.time() - ci._PAGINATION_STATE_MAX_AGE_SECONDS - 10
    ci._write_checkpoint_state(
        checkpoint_root=checkpoint_dir,
        key=key,
        completed_providers=set(),
        all_edges={},
        all_papers={"doi:10.1000/xyz1": ci.IngestionPaper(paper_id="doi:10.1000/xyz1")},
        provider_stats={},
        combined_completeness={},
        provider_pagination_state={
            "resume_capture": {
                "doi:10.1000/xyz1": {
                    "status": "in_progress",
                    "offset": 42,
                    "updated_at": stale_ts,
                    "papers": {},
                }
            }
        },
    )

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["resume_capture"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    assert provider.seen_resume_states == [None]
    stats = result.metadata["checkpoint_stats"]
    assert stats["stale_state_ignored_count"] == 1
    assert stats["stale_state_ignored_seeds"] == ["resume_capture:doi:10.1000/xyz1"]


def test_fresh_pagination_state_is_reused(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    provider = _ResumeCaptureProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    request_payload = ci._request_payload(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        key_constructs=None,
        sources=["resume_capture"],
        depth="l2",
        max_l2=200,
        max_l3=None,
        exhaustive=True,
    )
    key = ci._cache_key(request_payload)
    fresh_ts = time.time()
    ci._write_checkpoint_state(
        checkpoint_root=checkpoint_dir,
        key=key,
        completed_providers=set(),
        all_edges={},
        all_papers={"doi:10.1000/xyz1": ci.IngestionPaper(paper_id="doi:10.1000/xyz1")},
        provider_stats={},
        combined_completeness={},
        provider_pagination_state={
            "resume_capture": {
                "doi:10.1000/xyz1": {
                    "status": "in_progress",
                    "offset": 7,
                    "updated_at": fresh_ts,
                    "papers": {},
                }
            }
        },
    )

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["resume_capture"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    assert provider.seen_resume_states and provider.seen_resume_states[0] is not None
    assert provider.seen_resume_states[0]["offset"] == 7
    assert result.metadata["checkpoint_stats"]["stale_state_ignored_count"] == 0


def test_checkpoint_staleness_override_keeps_recent_state(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    provider = _ResumeCaptureProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    request_payload = ci._request_payload(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        key_constructs=None,
        sources=["resume_capture"],
        depth="l2",
        max_l2=200,
        max_l3=None,
        exhaustive=True,
    )
    key = ci._cache_key(request_payload)
    # This timestamp is stale under the default 6h policy, but fresh under 24h override.
    updated_ts = time.time() - ci._PAGINATION_STATE_MAX_AGE_SECONDS - 10
    ci._write_checkpoint_state(
        checkpoint_root=checkpoint_dir,
        key=key,
        completed_providers=set(),
        all_edges={},
        all_papers={"doi:10.1000/xyz1": ci.IngestionPaper(paper_id="doi:10.1000/xyz1")},
        provider_stats={},
        combined_completeness={},
        provider_pagination_state={
            "resume_capture": {
                "doi:10.1000/xyz1": {
                    "status": "in_progress",
                    "offset": 42,
                    "updated_at": updated_ts,
                    "papers": {},
                }
            }
        },
    )

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["resume_capture"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_staleness_seconds=24 * 60 * 60,
        refresh=True,
    )

    assert provider.seen_resume_states and provider.seen_resume_states[0] is not None
    assert provider.seen_resume_states[0]["offset"] == 42
    assert result.metadata["checkpoint_stats"]["stale_state_ignored_count"] == 0


class _L3ResumeCaptureProvider(ci.CitationProvider):
    name = "resume_l3_capture"
    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=False)

    def __init__(self):
        self.seen_l3_resume_states = []

    def fetch_seed_metadata(self, l1_papers):
        return {}

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        seed = ci.normalize_identifier(l1_papers[0])
        return (
            {"resume_l3_capture:l2": {seed}},
            {
                "resume_l3_capture:l2": ci.IngestionPaper(
                    paper_id="resume_l3_capture:l2",
                    title="L2",
                    year=2022,
                )
            },
        )

    def fetch_l3_references(
        self,
        l2_paper_ids,
        max_l3=None,
        resume_state=None,
        progress_callback=None,
    ):
        self.seen_l3_resume_states.append(dict(resume_state or {}))
        return {}, {}


def test_stale_l3_resume_state_is_ignored(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    provider = _L3ResumeCaptureProvider()
    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    request_payload = ci._request_payload(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        key_constructs=None,
        sources=["resume_l3_capture"],
        depth="l2l3",
        max_l2=200,
        max_l3=None,
        exhaustive=True,
    )
    key = ci._cache_key(request_payload)
    stale_ts = time.time() - ci._PAGINATION_STATE_MAX_AGE_SECONDS - 10
    ci._write_checkpoint_state(
        checkpoint_root=checkpoint_dir,
        key=key,
        completed_providers=set(),
        all_edges={},
        all_papers={"doi:10.1000/xyz1": ci.IngestionPaper(paper_id="doi:10.1000/xyz1")},
        provider_stats={},
        combined_completeness={},
        provider_l3_state={
            "resume_l3_capture": {
                "next_l2_index": 3,
                "budget_remaining": 7,
                "updated_at": stale_ts,
                "edges": {},
                "papers": {},
            }
        },
    )

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["resume_l3_capture"],
        depth="l2l3",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    assert provider.seen_l3_resume_states == [{}]
    stats = result.metadata["checkpoint_stats"]
    assert stats["l3_stale_state_ignored_count"] == 1
    assert stats["l3_stale_state_ignored_providers"] == ["resume_l3_capture"]


class _L2CrashMatrixProvider(ci.CitationProvider):
    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=False)

    def __init__(self, name: str, edge_id: str, crash: bool = False, fail_on_call: bool = False):
        self.name = name
        self.edge_id = edge_id
        self.crash = crash
        self.fail_on_call = fail_on_call

    def fetch_seed_metadata(self, l1_papers):
        if self.fail_on_call:
            raise AssertionError(f"{self.name} should have been skipped via checkpoint")
        return {
            l1_papers[0]: ci.IngestionPaper(
                paper_id=l1_papers[0],
                title=f"Seed from {self.name}",
                source_ids={self.name: f"seed-{self.name}"},
            )
        }

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        if self.fail_on_call:
            raise AssertionError(f"{self.name} should have been skipped via checkpoint")
        if self.crash:
            raise RuntimeError(f"simulated l2 crash in {self.name}")
        return (
            {self.edge_id: {l1_papers[0]}},
            {
                self.edge_id: ci.IngestionPaper(
                    paper_id=self.edge_id,
                    title=f"{self.name} L2",
                    year=2022,
                    citations=1,
                )
            },
        )

    def fetch_l3_references(self, l2_paper_ids, max_l3=None):
        return {}, {}


def test_checkpoint_resume_multi_provider_l2_crash_matches_baseline(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"

    first_openalex = _L2CrashMatrixProvider("openalex", "openalex:L2A")
    first_semantic = _L2CrashMatrixProvider("semantic_scholar", "semantic_scholar:L2A", crash=True)
    first_core = _L2CrashMatrixProvider("core", "core:L2A")
    monkeypatch.setattr(
        ci, "build_providers", lambda _: [first_openalex, first_semantic, first_core]
    )

    with pytest.raises(RuntimeError, match="simulated l2 crash"):
        ci.ingest_from_internet(
            theory_name="My Fake Theory",
            l1_papers=["10.1000/xyz1"],
            sources=["openalex", "semantic_scholar", "core"],
            depth="l2",
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            refresh=True,
        )

    resumed_openalex = _L2CrashMatrixProvider("openalex", "openalex:L2A", fail_on_call=True)
    resumed_semantic = _L2CrashMatrixProvider("semantic_scholar", "semantic_scholar:L2A")
    resumed_core = _L2CrashMatrixProvider("core", "core:L2A")
    monkeypatch.setattr(
        ci,
        "build_providers",
        lambda _: [resumed_openalex, resumed_semantic, resumed_core],
    )

    resumed = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["openalex", "semantic_scholar", "core"],
        depth="l2",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    baseline_openalex = _L2CrashMatrixProvider("openalex", "openalex:L2A")
    baseline_semantic = _L2CrashMatrixProvider("semantic_scholar", "semantic_scholar:L2A")
    baseline_core = _L2CrashMatrixProvider("core", "core:L2A")
    monkeypatch.setattr(
        ci,
        "build_providers",
        lambda _: [baseline_openalex, baseline_semantic, baseline_core],
    )

    baseline = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["openalex", "semantic_scholar", "core"],
        depth="l2",
        cache_dir=Path(tmp_path) / "baseline-cache",
        checkpoint_dir=Path(tmp_path) / "baseline-checkpoints",
        refresh=True,
    )

    assert resumed.citation_data == baseline.citation_data
    assert resumed.papers_data == baseline.papers_data


class _L3CrashMatrixProvider(ci.CitationProvider):
    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=False)

    def __init__(
        self,
        name: str,
        l2_prefix: str,
        l3_refs: dict[str, list[str]],
        crash_at_l3_index=None,
    ):
        self.name = name
        self.l2_prefix = l2_prefix
        self.l3_refs = l3_refs
        self.crash_at_l3_index = crash_at_l3_index

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
        self,
        l2_paper_ids,
        max_l3=None,
        resume_state=None,
        progress_callback=None,
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


def test_checkpoint_resume_multi_provider_l3_crash_matches_baseline(monkeypatch, tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"

    oa_l3_refs = {
        "openalex:L2A": ["doi:10.1000/shared-l3", "doi:10.1000/oa-l3a"],
        "openalex:L2B": ["doi:10.1000/oa-l3b"],
    }
    s2_l3_refs = {
        "semantic_scholar:L2A": ["doi:10.1000/shared-l3", "doi:10.1000/s2-l3a"],
        "semantic_scholar:L2B": ["doi:10.1000/s2-l3b"],
    }
    core_l3_refs = {
        "core:L2A": ["doi:10.1000/shared-l3", "doi:10.1000/core-l3a"],
        "core:L2B": ["doi:10.1000/core-l3b"],
    }

    first_openalex = _L3CrashMatrixProvider("openalex", "openalex", oa_l3_refs)
    first_semantic = _L3CrashMatrixProvider(
        "semantic_scholar",
        "semantic_scholar",
        s2_l3_refs,
        crash_at_l3_index=1,
    )
    first_core = _L3CrashMatrixProvider("core", "core", core_l3_refs)
    monkeypatch.setattr(
        ci, "build_providers", lambda _: [first_openalex, first_semantic, first_core]
    )

    with pytest.raises(RuntimeError, match="simulated l3 crash in semantic_scholar"):
        ci.ingest_from_internet(
            theory_name="My Fake Theory",
            l1_papers=["10.1000/xyz1"],
            sources=["openalex", "semantic_scholar", "core"],
            depth="l2l3",
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            refresh=True,
        )

    resumed_openalex = _L3CrashMatrixProvider("openalex", "openalex", oa_l3_refs)
    resumed_semantic = _L3CrashMatrixProvider("semantic_scholar", "semantic_scholar", s2_l3_refs)
    resumed_core = _L3CrashMatrixProvider("core", "core", core_l3_refs)
    monkeypatch.setattr(
        ci,
        "build_providers",
        lambda _: [resumed_openalex, resumed_semantic, resumed_core],
    )

    resumed = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["openalex", "semantic_scholar", "core"],
        depth="l2l3",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    baseline_openalex = _L3CrashMatrixProvider("openalex", "openalex", oa_l3_refs)
    baseline_semantic = _L3CrashMatrixProvider("semantic_scholar", "semantic_scholar", s2_l3_refs)
    baseline_core = _L3CrashMatrixProvider("core", "core", core_l3_refs)
    monkeypatch.setattr(
        ci,
        "build_providers",
        lambda _: [baseline_openalex, baseline_semantic, baseline_core],
    )

    baseline = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["openalex", "semantic_scholar", "core"],
        depth="l2l3",
        cache_dir=Path(tmp_path) / "baseline-cache",
        checkpoint_dir=Path(tmp_path) / "baseline-checkpoints",
        refresh=True,
    )

    assert resumed.citation_data == baseline.citation_data
    assert resumed.papers_data == baseline.papers_data
    assert resumed.metadata["provider_stats"] == baseline.metadata["provider_stats"]
    assert "semantic_scholar" in resumed.metadata["checkpoint_stats"]["l3_resumed_providers"]


def test_ingest_rejects_nonpositive_max_workers(tmp_path):
    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        ci.ingest_from_internet(
            theory_name="My Fake Theory",
            l1_papers=["10.1000/xyz1"],
            sources=["openalex"],
            depth="l2",
            cache_dir=Path(tmp_path),
            refresh=True,
            max_workers=0,
        )


def test_ingest_rejects_nonpositive_transient_retry_max_attempts(tmp_path):
    with pytest.raises(ValueError, match="transient_retry_max_attempts must be a positive integer"):
        ci.ingest_from_internet(
            theory_name="My Fake Theory",
            l1_papers=["10.1000/xyz1"],
            sources=["openalex"],
            depth="l2",
            cache_dir=Path(tmp_path),
            refresh=True,
            transient_retry_max_attempts=0,
        )


def test_ingest_rejects_nonpositive_transient_retry_max_age_seconds(tmp_path):
    with pytest.raises(
        ValueError, match="transient_retry_max_age_seconds must be a positive integer"
    ):
        ci.ingest_from_internet(
            theory_name="My Fake Theory",
            l1_papers=["10.1000/xyz1"],
            sources=["openalex"],
            depth="l2",
            cache_dir=Path(tmp_path),
            refresh=True,
            transient_retry_max_age_seconds=0,
        )


def test_parallel_wave1_execution_path_runs(monkeypatch, tmp_path):
    p_openalex = _L2CrashMatrixProvider("openalex", "openalex:L2A")
    p_semantic = _L2CrashMatrixProvider("semantic_scholar", "semantic_scholar:L2A")
    monkeypatch.setattr(ci, "build_providers", lambda _: [p_openalex, p_semantic])

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["openalex", "semantic_scholar"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        max_workers=2,
    )

    # Both providers should contribute at least one edge in parallel wave-1 mode.
    assert result.metadata["edge_count"] >= 2


def test_parallel_l2_to_l3_execution_path_runs(monkeypatch, tmp_path):
    tracker = {"active": 0, "overlap": False, "lock": threading.Lock()}

    class _ParallelL3Provider(ci.CitationProvider):
        capabilities = ci.ProviderCapabilities(
            True,
            True,
            True,
            supports_cited_by_traversal=True,
            supports_l3_outgoing=False,
        )

        def __init__(self, name: str):
            self.name = name

        def fetch_seed_metadata(self, l1_papers):
            return {
                "doi:10.1000/xyz1": ci.IngestionPaper(
                    paper_id="doi:10.1000/xyz1",
                    source_ids={self.name: f"seed:{self.name}:L1"},
                )
            }

        def fetch_citers_for_l1(
            self, l1_provider_id, max_results=None, resume_state=None, progress_callback=None
        ):
            parent_id = f"{self.name}:L2A"
            return (
                {
                    parent_id: ci.IngestionPaper(
                        paper_id=parent_id,
                        source_ids={self.name: f"{self.name}:L2A"},
                    )
                },
                1,
                "complete",
            )

        def fetch_l3_references(
            self, l2_paper_ids, max_l3=None, resume_state=None, progress_callback=None
        ):
            with tracker["lock"]:
                tracker["active"] += 1
                if tracker["active"] >= 2:
                    tracker["overlap"] = True

            time.sleep(0.06)

            with tracker["lock"]:
                tracker["active"] -= 1

            parent = l2_paper_ids[0]
            ref_id = f"doi:10.1000/{self.name}-l3"
            return (
                {parent: {ref_id}},
                {ref_id: ci.IngestionPaper(paper_id=ref_id, doi=ref_id.split(":", 1)[1])},
            )

    p1 = _ParallelL3Provider("parallel_l3_a")
    p2 = _ParallelL3Provider("parallel_l3_b")
    monkeypatch.setattr(ci, "build_providers", lambda _: [p1, p2])

    result = ci.ingest_from_internet(
        theory_name="My Fake Theory",
        l1_papers=["10.1000/xyz1"],
        sources=["parallel_l3_a", "parallel_l3_b"],
        depth="l2l3",
        cache_dir=Path(tmp_path),
        refresh=True,
        max_workers=2,
    )

    assert tracker["overlap"] is True
    assert result.metadata["provider_stats"]["parallel_l3_a"]["l3_edges"] == 1
    assert result.metadata["provider_stats"]["parallel_l3_b"]["l3_edges"] == 1
