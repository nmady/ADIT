"""Focused tests for publication year normalization behavior."""

import numpy as np

from adit import ADIT


def test_pub_year_all_same_year_returns_zero(mock_transformer):
    """When all years are equal, normalized publication year should be 0.0."""
    adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
    adit.build_ecosystem({"PaperA": ["TAM1"], "PaperB": ["TAM1"]})

    papers_data = {
        "PaperA": {
            "title": "Paper A",
            "abstract": "A",
            "keywords": "k",
            "citations": 1,
            "year": 2000,
        },
        "PaperB": {
            "title": "Paper B",
            "abstract": "B",
            "keywords": "k",
            "citations": 2,
            "year": 2000,
        },
        "TAM1": {
            "title": "TAM",
            "abstract": "Foundational",
            "keywords": "TAM",
            "citations": 10,
            "year": 2000,
        },
    }

    features = adit.extract_features(papers_data)
    assert len(features) == 2
    assert np.allclose(features["pub_year"].values, 0.0)


def test_pub_year_exact_values_with_wide_range(mock_transformer):
    """Normalized year values should match the expected min-max scaling formula."""
    adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
    adit.build_ecosystem({"PaperA": ["TAM1"], "PaperB": ["TAM1"]})

    papers_data = {
        "PaperA": {
            "title": "Paper A",
            "abstract": "A",
            "keywords": "k",
            "citations": 1,
            "year": 2010,
        },
        "PaperB": {
            "title": "Paper B",
            "abstract": "B",
            "keywords": "k",
            "citations": 2,
            "year": 2015,
        },
        "TAM1": {
            "title": "TAM",
            "abstract": "Foundational",
            "keywords": "TAM",
            "citations": 10,
            "year": 2000,
        },
    }

    # min_year=2000, max_year=2015, range=15
    # expected PaperA=(2010-2000)/15=0.666..., PaperB=(2015-2000)/15=1.0
    features = adit.extract_features(papers_data).set_index("paper_id")

    assert np.isclose(features.loc["PaperA", "pub_year"], 10 / 15)
    assert np.isclose(features.loc["PaperB", "pub_year"], 1.0)


def test_pub_year_missing_year_defaults_to_2010(mock_transformer):
    """Missing year should default to 2010 and still produce bounded normalized values."""
    adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
    adit.build_ecosystem({"PaperA": ["TAM1"], "PaperB": ["TAM1"]})

    papers_data = {
        "PaperA": {
            "title": "Paper A",
            "abstract": "A",
            "keywords": "k",
            "citations": 1,
            # missing year -> defaults to 2010
        },
        "PaperB": {
            "title": "Paper B",
            "abstract": "B",
            "keywords": "k",
            "citations": 2,
            "year": 2020,
        },
        "TAM1": {
            "title": "TAM",
            "abstract": "Foundational",
            "keywords": "TAM",
            "citations": 10,
            "year": 2000,
        },
    }

    features = adit.extract_features(papers_data).set_index("paper_id")

    # min_year=2000, max_year=2020, range=20
    # PaperA default year 2010 => 0.5; PaperB => 1.0
    assert np.isclose(features.loc["PaperA", "pub_year"], 0.5)
    assert np.isclose(features.loc["PaperB", "pub_year"], 1.0)
    assert (features["pub_year"] >= 0).all()
    assert (features["pub_year"] <= 1).all()


def test_pub_year_no_nan_or_inf_for_single_l2_paper(mock_transformer):
    """A single L2 paper should never produce NaN/inf in normalized year."""
    adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
    adit.build_ecosystem({"PaperA": ["TAM1"]})

    papers_data = {
        "PaperA": {
            "title": "Paper A",
            "abstract": "A",
            "keywords": "k",
            "citations": 1,
            "year": 2018,
        },
        "TAM1": {
            "title": "TAM",
            "abstract": "Foundational",
            "keywords": "TAM",
            "citations": 10,
            "year": 2018,
        },
    }

    features = adit.extract_features(papers_data)
    assert len(features) == 1
    value = features.loc[0, "pub_year"]
    assert not np.isnan(value)
    assert not np.isinf(value)
