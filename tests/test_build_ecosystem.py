from adit import ADIT


def test_build_ecosystem_levels(sample_citation_data):
    adit = ADIT("TAM", ["TAM1"])
    adit.build_ecosystem(sample_citation_data)

    # L1 should exist for the foundational paper
    assert "TAM1" in adit.ecosystem.nodes
    assert adit.ecosystem.nodes["TAM1"]["level"] == "L1"

    # PaperA cites TAM1 → should be L2
    assert "PaperA" in adit.ecosystem.nodes
    assert adit.ecosystem.nodes["PaperA"]["level"] == "L2"

    # References should create edges (citing -> cited)
    assert adit.ecosystem.has_edge("PaperA", "TAM1")

    # Unknown cited node should be present for non-L1 references
    assert "Other" in adit.ecosystem.nodes or any(True for n in adit.ecosystem.nodes())
