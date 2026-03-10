# tests/test_semantic_similarity_robustness.py
"""Robustness tests for semantic similarity feature extraction."""

import numpy as np

from adit import ADIT


class ZeroAwareTransformer:
    """Returns zero vector for empty text, unit-x vector otherwise."""

    def encode(self, text):
        if not text or not str(text).strip():
            return np.array([0.0, 0.0, 0.0])
        return np.array([1.0, 0.0, 0.0])


class OrthogonalTransformer:
    """Returns orthogonal vectors for theory vs paper to test near-zero cosine."""

    def encode(self, text):
        t = (text or "").lower()
        if "foundational" in t or "theory" in t:
            return np.array([1.0, 0.0, 0.0])
        return np.array([0.0, 1.0, 0.0])


def test_semantic_similarity_empty_l2_abstract_returns_zero():
    adit = ADIT("TAM", ["TAM1"], transformer=ZeroAwareTransformer())
    adit.build_ecosystem({"PaperA": ["TAM1"]})

    papers_data = {
        "PaperA": {
            "title": "Paper A",
            "abstract": "",  # empty L2 abstract
            "keywords": "k",
            "citations": 1,
            "year": 2010,
        },
        "TAM1": {
            "title": "Foundational TAM",
            "abstract": "Foundational theory paper",
            "keywords": "tam",
            "citations": 100,
            "year": 1990,
        },
    }

    features = adit.extract_features(papers_data)
    assert len(features) == 1
    assert features.loc[0, "semantic_similarity"] == 0.0


def test_semantic_similarity_empty_l1_abstracts_returns_zero():
    adit = ADIT("TAM", ["TAM1", "TAM2"], transformer=ZeroAwareTransformer())
    adit.build_ecosystem({"PaperA": ["TAM1"]})

    papers_data = {
        "PaperA": {
            "title": "Paper A",
            "abstract": "Some non-empty abstract",
            "keywords": "k",
            "citations": 1,
            "year": 2010,
        },
        "TAM1": {
            "title": "Foundational 1",
            "abstract": "",  # empty L1 abstract
            "keywords": "tam",
            "citations": 100,
            "year": 1990,
        },
        "TAM2": {
            "title": "Foundational 2",
            "abstract": "",  # empty L1 abstract
            "keywords": "tam",
            "citations": 80,
            "year": 1992,
        },
    }

    features = adit.extract_features(papers_data)
    assert len(features) == 1
    assert features.loc[0, "semantic_similarity"] == 0.0


def test_semantic_similarity_missing_abstract_field_defaults_safely():
    adit = ADIT("TAM", ["TAM1"], transformer=ZeroAwareTransformer())
    adit.build_ecosystem({"PaperA": ["TAM1"]})

    papers_data = {
        "PaperA": {
            "title": "Paper A",
            # missing abstract -> defaults to ""
            "keywords": "k",
            "citations": 1,
            "year": 2010,
        },
        "TAM1": {
            "title": "Foundational TAM",
            "abstract": "Foundational theory paper",
            "keywords": "tam",
            "citations": 100,
            "year": 1990,
        },
    }

    features = adit.extract_features(papers_data)
    assert len(features) == 1
    value = features.loc[0, "semantic_similarity"]
    assert value == 0.0
    assert not np.isnan(value)
    assert not np.isinf(value)


def test_semantic_similarity_is_finite_and_bounded():
    adit = ADIT("TAM", ["TAM1"], transformer=OrthogonalTransformer())
    adit.build_ecosystem({"PaperA": ["TAM1"]})

    papers_data = {
        "PaperA": {
            "title": "Paper A",
            "abstract": "Application paper text",
            "keywords": "k",
            "citations": 1,
            "year": 2010,
        },
        "TAM1": {
            "title": "Foundational TAM",
            "abstract": "Foundational theory paper",
            "keywords": "tam",
            "citations": 100,
            "year": 1990,
        },
    }

    features = adit.extract_features(papers_data)
    value = features.loc[0, "semantic_similarity"]
    assert not np.isnan(value)
    assert not np.isinf(value)
    assert -1.0 <= value <= 1.0
    assert np.isclose(value, 0.0)  # orthogonal vectors


def test_semantic_similarity_long_abstract_remains_stable():
    adit = ADIT("TAM", ["TAM1"], transformer=ZeroAwareTransformer())
    adit.build_ecosystem({"PaperA": ["TAM1"]})

    long_abstract = "word " * 20000
    papers_data = {
        "PaperA": {
            "title": "Paper A",
            "abstract": long_abstract,
            "keywords": "k",
            "citations": 1,
            "year": 2010,
        },
        "TAM1": {
            "title": "Foundational TAM",
            "abstract": "Foundational theory paper",
            "keywords": "tam",
            "citations": 100,
            "year": 1990,
        },
    }

    features = adit.extract_features(papers_data)
    value = features.loc[0, "semantic_similarity"]
    assert not np.isnan(value)
    assert not np.isinf(value)
    assert -1.0 <= value <= 1.0