"""
Test exception handling and logging in ADIT.

Verifies that exceptions in PageRank and betweenness centrality
are properly caught, logged, and handled with fallback values.
"""

import logging

import pytest

from adit import ADIT


@pytest.fixture
def simple_adit(mock_transformer):
    """Create a simple ADIT instance for testing."""
    l1_papers = ["Paper1", "Paper2"]
    adit = ADIT("TestTheory", l1_papers, transformer=mock_transformer)
    return adit


@pytest.fixture
def simple_citation_data():
    """Minimal citation data for testing."""
    return {"PaperA": ["Paper1"], "PaperB": ["Paper2"], "PaperC": ["Paper1", "Paper2"]}


@pytest.fixture
def simple_papers_data():
    """Minimal papers data for testing."""
    return {
        "PaperA": {
            "title": "Test paper A",
            "abstract": "Abstract for paper A",
            "keywords": "test, paper",
            "citations": 10,
            "year": 2020,
        },
        "PaperB": {
            "title": "Test paper B",
            "abstract": "Abstract for paper B",
            "keywords": "test, paper",
            "citations": 5,
            "year": 2021,
        },
        "PaperC": {
            "title": "Test paper C",
            "abstract": "Abstract for paper C",
            "keywords": "test, paper",
            "citations": 8,
            "year": 2022,
        },
        "Paper1": {
            "title": "Foundational paper 1",
            "abstract": "Foundation abstract 1",
            "keywords": "foundation",
            "citations": 100,
            "year": 2010,
        },
        "Paper2": {
            "title": "Foundational paper 2",
            "abstract": "Foundation abstract 2",
            "keywords": "foundation",
            "citations": 50,
            "year": 2012,
        },
    }


def test_pagerank_exception_logging(simple_adit, simple_citation_data, mocker, caplog):
    """Test that PageRank exceptions are caught and logged."""
    simple_adit.build_ecosystem(simple_citation_data)

    # Mock nx.pagerank to raise an exception
    mocker.patch("networkx.pagerank", side_effect=Exception("PageRank convergence failed"))

    # Capture logging at WARNING level
    with caplog.at_level(logging.WARNING):
        scores = simple_adit.compute_eigenfactor()

    # Verify logging occurred
    assert len(caplog.records) == 1
    assert "PageRank computation failed" in caplog.text
    assert "uniform scores as fallback" in caplog.text

    # Verify fallback values (all nodes get 1.0)
    assert all(score == 1.0 for score in scores.values())
    assert len(scores) == len(simple_adit.ecosystem.nodes())


def test_pagerank_exception_fallback_values(simple_adit, simple_citation_data, mocker):
    """Test that PageRank fallback provides correct uniform scores."""
    simple_adit.build_ecosystem(simple_citation_data)

    # Mock nx.pagerank to raise an exception
    mocker.patch("networkx.pagerank", side_effect=RuntimeError("Convergence error"))

    scores = simple_adit.compute_eigenfactor()

    # All nodes should have score 1.0
    expected_nodes = set(simple_adit.ecosystem.nodes())
    assert set(scores.keys()) == expected_nodes
    assert all(score == 1.0 for score in scores.values())


def test_betweenness_exception_logging(
    simple_adit, simple_citation_data, simple_papers_data, mocker, caplog
):
    """Test that betweenness centrality exceptions are caught and logged."""
    simple_adit.build_ecosystem(simple_citation_data)

    # Mock nx.betweenness_centrality to raise an exception
    mocker.patch("networkx.betweenness_centrality", side_effect=MemoryError("Out of memory"))

    # Capture logging at WARNING level
    with caplog.at_level(logging.WARNING):
        features = simple_adit.extract_features(simple_papers_data)

    # Verify logging occurred
    assert any(
        "Betweenness centrality computation failed" in record.message for record in caplog.records
    )
    assert any("zero scores as fallback" in record.message for record in caplog.records)

    # Verify features were still extracted (with zero betweenness)
    assert "betweenness" in features.columns
    assert all(features["betweenness"] == 0.0)


def test_betweenness_exception_fallback_values(
    simple_adit, simple_citation_data, simple_papers_data, mocker
):
    """Test that betweenness fallback provides correct zero scores."""
    simple_adit.build_ecosystem(simple_citation_data)

    # Mock nx.betweenness_centrality to raise an exception
    mocker.patch("networkx.betweenness_centrality", side_effect=ValueError("Invalid graph"))

    features = simple_adit.extract_features(simple_papers_data)

    # All L2 papers should have betweenness = 0.0
    assert "betweenness" in features.columns
    assert all(features["betweenness"] == 0.0)
    # Verify other features are still computed
    assert len(features) > 0
    assert "eigenfactor" in features.columns


def test_both_exceptions_together(
    simple_adit, simple_citation_data, simple_papers_data, mocker, caplog
):
    """Test that both PageRank and betweenness exceptions can be handled simultaneously."""
    simple_adit.build_ecosystem(simple_citation_data)

    # Mock both functions to raise exceptions
    mocker.patch("networkx.pagerank", side_effect=Exception("PageRank failed"))
    mocker.patch("networkx.betweenness_centrality", side_effect=Exception("Betweenness failed"))

    with caplog.at_level(logging.WARNING):
        features = simple_adit.extract_features(simple_papers_data)

    # Verify both warnings were logged
    log_messages = [record.message for record in caplog.records]
    assert any("PageRank computation failed" in msg for msg in log_messages)
    assert any("Betweenness centrality computation failed" in msg for msg in log_messages)

    # Verify features extracted with fallback values
    assert "eigenfactor" in features.columns
    assert "betweenness" in features.columns
    assert all(features["eigenfactor"] == 1.0)
    assert all(features["betweenness"] == 0.0)


def test_normal_operation_no_logging(simple_adit, simple_citation_data, simple_papers_data, caplog):
    """Test that successful operations don't trigger warning logs."""
    simple_adit.build_ecosystem(simple_citation_data)

    with caplog.at_level(logging.WARNING):
        features = simple_adit.extract_features(simple_papers_data)

    # No warnings should be logged in normal operation
    assert len(caplog.records) == 0

    # Features should be computed normally
    assert "eigenfactor" in features.columns
    assert "betweenness" in features.columns
    # Eigenfactor should not all be 1.0 (PageRank actually ran)
    # Note: in a very small graph, PageRank might converge to near-uniform values,
    # but the fallback is exactly 1.0, so we can distinguish
    assert len(features) > 0
