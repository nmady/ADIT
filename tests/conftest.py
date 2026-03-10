import numpy as np
import pytest


class MockTransformer:
    """Simple mock transformer with deterministic small embeddings."""

    def encode(self, text):
        # return a fixed 3-d vector for deterministic similarity in tests
        return np.array([1.0, 0.0, 0.0])


@pytest.fixture
def mock_transformer():
    return MockTransformer()


@pytest.fixture
def sample_citation_data():
    return {"PaperA": ["TAM1"], "PaperB": ["TAM2"], "PaperC": ["Other"], "PaperE": ["TAM1", "TAM2"]}


@pytest.fixture
def sample_papers_data():
    return {
        "PaperA": {
            "title": "Extension of Technology Acceptance Model",
            "abstract": "This paper extends TAM with new constructs for mobile adoption.",
            "keywords": "TAM, technology acceptance, mobile",
            "citations": 50,
            "year": 2015,
        },
        "PaperB": {
            "title": "Empirical test of TAM",
            "abstract": "We test TAM in a new context with emphasis on ease of use and usefulness.",
            "keywords": "TAM, acceptance, empirical study",
            "citations": 30,
            "year": 2012,
        },
        "PaperC": {
            "title": "Unrelated topic in information systems",
            "abstract": "This paper studies something unrelated to technology acceptance.",
            "keywords": "information systems, management",
            "citations": 10,
            "year": 2010,
        },
        "PaperE": {
            "title": "TAM in healthcare: Ease of use and behavioral intention",
            "abstract": "Applies TAM to understand healthcare technology adoption with focus on usefulness.",
            "keywords": "TAM, acceptance, healthcare, behavioral intention",
            "citations": 25,
            "year": 2014,
        },
        "TAM1": {
            "title": "Technology Acceptance Model",
            "abstract": "Original TAM paper proposing model of technology acceptance.",
            "keywords": "TAM, acceptance, technology",
            "citations": 1000,
            "year": 1989,
        },
        "TAM2": {
            "title": "Technology Acceptance Model extension",
            "abstract": "Extension of TAM with more constructs.",
            "keywords": "TAM, acceptance",
            "citations": 500,
            "year": 1992,
        },
    }
