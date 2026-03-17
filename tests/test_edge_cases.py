"""Edge-case and validation tests for ADIT."""

import numpy as np
import pandas as pd

from adit import ADIT


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_ecosystem(self, mock_transformer):
        """Test handling of empty citation graph (no L2 papers)."""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
        # Build ecosystem with no papers citing L1
        citation_data = {"Unrelated": ["SomeOtherPaper"]}
        adit.build_ecosystem(citation_data)

        # Extract features should return empty DataFrame (no L2 papers)
        papers_data = {
            "TAM1": {
                "title": "TAM",
                "abstract": "TAM paper",
                "keywords": "TAM",
                "citations": 100,
                "year": 1989,
            }
        }
        features = adit.extract_features(papers_data)
        assert len(features) == 0, "Should return empty DataFrame for no L2 papers"

    def test_single_node_graph(self, mock_transformer):
        """Test with only the L1 paper, no L2."""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
        adit.build_ecosystem({})  # no citations

        papers_data = {
            "TAM1": {
                "title": "TAM",
                "abstract": "TAM paper",
                "keywords": "TAM",
                "citations": 100,
                "year": 1989,
            }
        }
        features = adit.extract_features(papers_data)
        assert len(features) == 0, "No L2 papers → empty features"

    def test_missing_paper_metadata_fields(self, mock_transformer, sample_citation_data):
        """Test robustness when papers_data has missing fields."""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
        adit.build_ecosystem(sample_citation_data)

        # Minimal papers_data with missing fields
        incomplete_papers = {
            "PaperA": {
                "title": "Paper A",
                # missing 'abstract', 'keywords', 'citations', 'year'
            },
            "TAM1": {
                "title": "TAM",
                "abstract": "Original TAM",
                "keywords": "TAM",
                "citations": 100,
                "year": 1989,
            },
        }

        # Should handle gracefully with defaults
        features = adit.extract_features(incomplete_papers)
        assert len(features) > 0
        # Check defaults are used
        assert not pd.isna(features.loc[0, "citation_count"]), "Should default to 0"
        # Missing year is preserved as NaN by design.
        assert pd.isna(features.loc[0, "pub_year"]), "Missing year should remain NaN"

    def test_zero_citations(self, mock_transformer, sample_citation_data):
        """Test paper with zero citations."""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
        adit.build_ecosystem(sample_citation_data)

        papers_data = {
            "PaperA": {
                "title": "Paper A",
                "abstract": "Abstract",
                "keywords": "keywords",
                "citations": 0,  # zero citations
                "year": 2010,
            },
            "TAM1": {
                "title": "TAM",
                "abstract": "TAM paper",
                "keywords": "TAM",
                "citations": 100,
                "year": 1989,
            },
        }
        features = adit.extract_features(papers_data)
        assert features.loc[0, "citation_count"] == 0

    def test_papers_without_constructs(self, mock_transformer, sample_citation_data):
        """Test paper that mentions no key constructs."""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
        adit.build_ecosystem(sample_citation_data)

        papers_data = {
            "PaperA": {
                "title": "Machine Learning for Data Science",
                "abstract": "This paper discusses algorithms and neural networks.",
                "keywords": "ML, algorithms",
                "citations": 50,
                "year": 2015,
            },
            "TAM1": {
                "title": "TAM",
                "abstract": "TAM paper",
                "keywords": "TAM",
                "citations": 100,
                "year": 1989,
            },
        }
        features = adit.extract_features(papers_data)
        # All construct flags should be 0
        construct_cols = [
            "has_usefulness",
            "has_ease_of_use",
            "has_acceptance",
            "has_intention",
            "has_attitude",
        ]
        assert all(features.loc[0, col] == 0 for col in construct_cols)

    def test_all_papers_same_year(self, mock_transformer, sample_citation_data):
        """Test year normalization when all papers have the same year."""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
        adit.build_ecosystem(sample_citation_data)

        papers_data = {
            "PaperA": {
                "title": "Paper A",
                "abstract": "Abstract",
                "keywords": "keywords",
                "citations": 50,
                "year": 2000,  # all same year
            },
            "PaperB": {
                "title": "Paper B",
                "abstract": "Abstract",
                "keywords": "keywords",
                "citations": 30,
                "year": 2000,
            },
            "PaperC": {
                "title": "Paper C",
                "abstract": "Abstract",
                "keywords": "keywords",
                "citations": 10,
                "year": 2000,
            },
            "PaperE": {
                "title": "Paper E",
                "abstract": "Abstract",
                "keywords": "keywords",
                "citations": 20,
                "year": 2000,
            },
            "TAM1": {
                "title": "TAM",
                "abstract": "TAM paper",
                "keywords": "TAM",
                "citations": 100,
                "year": 2000,
            },
        }
        features = adit.extract_features(papers_data)
        # When year_range is 0, normalization should handle gracefully (likely all 0.0)
        # Check no NaN or inf
        assert not np.any(np.isnan(features["pub_year"].values))
        assert not np.any(np.isinf(features["pub_year"].values))

    def test_very_old_and_new_papers(self, mock_transformer, sample_citation_data):
        """Test with papers spanning very large year range."""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
        adit.build_ecosystem(sample_citation_data)

        papers_data = {
            "PaperA": {
                "title": "Paper A",
                "abstract": "Abstract",
                "keywords": "keywords",
                "citations": 50,
                "year": 1950,  # very old
            },
            "PaperB": {
                "title": "Paper B",
                "abstract": "Abstract",
                "keywords": "keywords",
                "citations": 30,
                "year": 2025,  # very new
            },
            "PaperC": {
                "title": "Paper C",
                "abstract": "Abstract",
                "keywords": "keywords",
                "citations": 10,
                "year": 1989,
            },
            "PaperE": {
                "title": "Paper E",
                "abstract": "Abstract",
                "keywords": "keywords",
                "citations": 20,
                "year": 1995,
            },
            "TAM1": {
                "title": "TAM",
                "abstract": "TAM paper",
                "keywords": "TAM",
                "citations": 100,
                "year": 1989,
            },
        }
        features = adit.extract_features(papers_data)
        # Normalized years should be in [0, 1]
        assert (features["pub_year"] >= 0).all()
        assert (features["pub_year"] <= 1).all()

    def test_very_large_citation_count(self, mock_transformer, sample_citation_data):
        """Test with extremely high citation counts."""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
        adit.build_ecosystem(sample_citation_data)

        papers_data = {
            "PaperA": {
                "title": "Paper A",
                "abstract": "Abstract",
                "keywords": "keywords",
                "citations": 1000000,  # very high
                "year": 2010,
            },
            "TAM1": {
                "title": "TAM",
                "abstract": "TAM paper",
                "keywords": "TAM",
                "citations": 100,
                "year": 1989,
            },
        }
        features = adit.extract_features(papers_data)
        assert features.loc[0, "citation_count"] == 1000000
        assert not np.isnan(features.loc[0, "citation_count"])

    def test_empty_abstract(self, mock_transformer, sample_citation_data):
        """Test paper with empty abstract."""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
        adit.build_ecosystem(sample_citation_data)

        papers_data = {
            "PaperA": {
                "title": "Paper A",
                "abstract": "",  # empty
                "keywords": "keywords",
                "citations": 50,
                "year": 2010,
            },
            "TAM1": {
                "title": "TAM",
                "abstract": "TAM paper",
                "keywords": "TAM",
                "citations": 100,
                "year": 1989,
            },
        }
        features = adit.extract_features(papers_data)
        assert features.loc[0, "abstract_word_count"] == 0

    def test_multiword_theory_name(self, mock_transformer, sample_citation_data):
        """Test with multi-word theory name and acronym detection."""
        adit = ADIT("Technology Acceptance Model", ["TAM1"], transformer=mock_transformer)
        adit.build_ecosystem(sample_citation_data)

        papers_data = {
            "PaperA": {
                "title": "Technology Acceptance Model",
                "abstract": "We apply TAM to mobile devices.",
                "keywords": "TAM, technology",
                "citations": 50,
                "year": 2010,
            },
            "TAM1": {
                "title": "TAM",
                "abstract": "TAM paper",
                "keywords": "TAM",
                "citations": 100,
                "year": 1989,
            },
        }
        features = adit.extract_features(papers_data)
        # Acronym is 'tam' (first letters: T, A, M)
        assert (
            features.loc[0, "acronym_in_title"] == 1 or features.loc[0, "acronym_in_abstract"] == 1
        )

    def test_case_insensitivity(self, mock_transformer, sample_citation_data):
        """Test that theory name matching is case-insensitive."""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
        adit.build_ecosystem(sample_citation_data)

        papers_data = {
            "PaperA": {
                "title": "tam in research",  # lowercase
                "abstract": "TAM MODEL",  # uppercase
                "keywords": "TaM",  # mixed case
                "citations": 50,
                "year": 2010,
            },
            "TAM1": {
                "title": "TAM",
                "abstract": "TAM paper",
                "keywords": "TAM",
                "citations": 100,
                "year": 1989,
            },
        }
        features = adit.extract_features(papers_data)
        # Should detect theory name in all cases
        assert features.loc[0, "theory_in_title"] == 1
        assert features.loc[0, "theory_in_abstract"] == 1
        assert features.loc[0, "theory_in_keywords"] == 1

    def test_multiple_l1_papers(self, mock_transformer):
        """Test with multiple L1 papers."""
        adit = ADIT("TAM", ["TAM1", "TAM2"], transformer=mock_transformer)
        citation_data = {
            "PaperA": ["TAM1"],
            "PaperB": ["TAM2"],
            "PaperC": ["TAM1", "TAM2"],  # cites both
        }
        adit.build_ecosystem(citation_data)

        papers_data = {
            "PaperA": {
                "title": "Paper A",
                "abstract": "Cites TAM1",
                "keywords": "keywords",
                "citations": 50,
                "year": 2010,
            },
            "PaperB": {
                "title": "Paper B",
                "abstract": "Cites TAM2",
                "keywords": "keywords",
                "citations": 30,
                "year": 2012,
            },
            "PaperC": {
                "title": "Paper C",
                "abstract": "Cites both",
                "keywords": "keywords",
                "citations": 60,
                "year": 2011,
            },
            "TAM1": {
                "title": "TAM1",
                "abstract": "TAM original",
                "keywords": "TAM",
                "citations": 1000,
                "year": 1989,
            },
            "TAM2": {
                "title": "TAM2",
                "abstract": "TAM extended",
                "keywords": "TAM",
                "citations": 500,
                "year": 1992,
            },
        }
        features = adit.extract_features(papers_data)
        assert len(features) == 3, "Should have features for all 3 L2 papers"

    def test_special_characters_in_text(self, mock_transformer):
        """Test handling of special characters in text fields."""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
        # Use standalone citation data to avoid interference from sample_citation_data
        adit.build_ecosystem({"PaperA": ["TAM1"]})

        papers_data = {
            "PaperA": {
                "title": 'TAM: A Study of "Acceptance" & Implementation!',
                "abstract": "Testing (TAM) in environments [e.g., mobile] with 50% success.",
                "keywords": "TAM, e-learning; mobile-devices",
                "citations": 50,
                "year": 2010,
            },
            "TAM1": {
                "title": "TAM",
                "abstract": "TAM paper",
                "keywords": "TAM",
                "citations": 100,
                "year": 1989,
            },
        }
        features = adit.extract_features(papers_data)
        assert len(features) == 1
        assert not pd.isna(features.iloc[0]).any(), "Should handle special chars without NaN"
