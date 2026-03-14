from pathlib import Path

import citation_ingestion as ci


class _FakeProvider(ci.CitationProvider):
    name = "fake"
    capabilities = ci.ProviderCapabilities(True, True, True)

    def __init__(self):
        self.calls = 0

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
                ),
                "semantic_scholar:abc123": ci.IngestionPaper(
                    paper_id="semantic_scholar:abc123",
                    title="Theory Application Study",
                    year=2022,
                    citations=12,
                    doi="10.1000/abc",
                ),
                "doi:10.1000/xyz1": ci.IngestionPaper(
                    paper_id="doi:10.1000/xyz1",
                    title="Foundational Paper",
                    year=2000,
                    citations=100,
                    doi="10.1000/xyz1",
                ),
            },
        )

    def fetch_l3_references(self, l2_paper_ids, max_l3=500):
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


def test_ingest_from_internet_dedupes_and_uses_cache(monkeypatch, tmp_path):
    provider = _FakeProvider()

    def fake_build_providers(_sources):
        return [provider]

    monkeypatch.setattr(ci, "build_providers", fake_build_providers)

    result1 = ci.ingest_from_internet(
        theory_name="Technology Acceptance Model",
        l1_papers=["10.1000/xyz1"],
        sources=["fake"],
        depth="l2l3",
        cache_dir=Path(tmp_path),
        refresh=False,
        max_l2=10,
        max_l3=10,
    )

    assert provider.calls == 1
    assert result1.metadata["paper_count"] >= 3
    assert result1.metadata["edge_count"] >= 2

    # Duplicate L2 candidates should collapse into a single citing node.
    assert len(result1.citation_data) == 1
    citing_id = next(iter(result1.citation_data.keys()))
    assert len(result1.citation_data[citing_id]) == 2

    # Second run should come from cache and avoid calling provider again.
    result2 = ci.ingest_from_internet(
        theory_name="Technology Acceptance Model",
        l1_papers=["10.1000/xyz1"],
        sources=["fake"],
        depth="l2l3",
        cache_dir=Path(tmp_path),
        refresh=False,
        max_l2=10,
        max_l3=10,
    )

    assert provider.calls == 1
    assert result2.metadata["cache_key"] == result1.metadata["cache_key"]
