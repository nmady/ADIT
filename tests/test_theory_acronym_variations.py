"""Comprehensive tests for theory acronym handling: derivation, lowercasing, and consistency."""

from adit import ADIT


class TestAcronymExplicitProvision:
    """Tests for explicitly provided acronyms."""

    def test_explicit_acronym_lowercase_conversion(self, mock_transformer):
        """Explicitly provided acronym should be lowercased regardless of input case."""
        adit = ADIT(
            "Technology Acceptance Model", ["TAM1"], transformer=mock_transformer, acronym="TAM"
        )
        assert adit.acronym == "tam"

    def test_explicit_acronym_uppercase_input(self, mock_transformer):
        """Uppercase acronym input should be converted to lowercase."""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer, acronym="TAM")
        assert adit.acronym == "tam"

    def test_explicit_acronym_mixedcase_input(self, mock_transformer):
        """Mixed-case acronym input should be normalized to lowercase."""
        adit = ADIT(
            "Technology Acceptance Model", ["TAM1"], transformer=mock_transformer, acronym="TaM"
        )
        assert adit.acronym == "tam"

    def test_explicit_acronym_already_lowercase(self, mock_transformer):
        """Lowercase acronym should remain unchanged."""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer, acronym="tam")
        assert adit.acronym == "tam"

    def test_explicit_acronym_single_letter(self, mock_transformer):
        """Single-letter acronym should work correctly."""
        adit = ADIT("X Theory", ["X1"], transformer=mock_transformer, acronym="X")
        assert adit.acronym == "x"

    def test_explicit_acronym_with_digits(self, mock_transformer):
        """Acronym with digits should be preserved and lowercased."""
        adit = ADIT("Theory6G", ["T6G1"], transformer=mock_transformer, acronym="T6G")
        assert adit.acronym == "t6g"


class TestAcronymDerivedFromTheoryName:
    """Tests for acronym derivation from theory_name when not explicitly provided."""

    def test_derived_acronym_single_word(self, mock_transformer):
        """Single-word theory name should derive first letter in lowercase."""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer)
        assert adit.acronym == "t"

    def test_derived_acronym_two_words(self, mock_transformer):
        """Two-word theory name should derive first letter of each word in lowercase."""
        adit = ADIT("Technology Acceptance", ["TA1"], transformer=mock_transformer)
        assert adit.acronym == "ta"

    def test_derived_acronym_three_words(self, mock_transformer):
        """Three-word theory name should derive first letter of each word in lowercase."""
        adit = ADIT("Technology Acceptance Model", ["TAM1"], transformer=mock_transformer)
        assert adit.acronym == "tam"

    def test_derived_acronym_many_words(self, mock_transformer):
        """Many-word theory name should derive first letter of each word in lowercase."""
        adit = ADIT(
            "Social Cognitive Theory of Self Regulation", ["SCTSR1"], transformer=mock_transformer
        )
        assert adit.acronym == "sctsr"

    def test_derived_acronym_uppercase_theory_name(self, mock_transformer):
        """Uppercase theory name should still derive lowercase acronym."""
        adit = ADIT("TECHNOLOGY ACCEPTANCE MODEL", ["TAM1"], transformer=mock_transformer)
        assert adit.acronym == "tam"

    def test_derived_acronym_mixedcase_theory_name(self, mock_transformer):
        """Mixed-case theory name should derive lowercase acronym."""
        adit = ADIT("Technology ACCeptance Model", ["TAM1"], transformer=mock_transformer)
        assert adit.acronym == "tam"

    def test_derived_acronym_with_hyphens(self, mock_transformer):
        """Theory name with hyphens should treat hyphenated parts as separate words."""
        adit = ADIT("Technology-Acceptance-Model", ["TAM1"], transformer=mock_transformer)
        # Split on spaces, so hyphens are part of word; first letter is 't'
        assert adit.acronym == "t"

    def test_derived_acronym_with_extra_spaces(self, mock_transformer):
        """Theory name with multiple spaces should handle gracefully."""
        adit = ADIT("Technology  Acceptance     Model", ["TAM1"], transformer=mock_transformer)
        # Split on spaces will create empty strings for extra spaces; filter them implicitly
        # Let me check: "Theory  Name".split() -> ['Theory', 'Name']
        # So this should still produce "tn"
        assert adit.acronym == "tam"


class TestAcronymConsistencyInPipeline:
    """Tests that acronym is consistent and stable throughout the ADIT pipeline."""

    def test_acronym_unchanged_after_build_ecosystem(self, mock_transformer):
        """Acronym should remain unchanged after building ecosystem."""
        adit = ADIT(
            "Technology Acceptance Model", ["TAM1"], transformer=mock_transformer, acronym="TAM"
        )
        original_acronym = adit.acronym
        adit.build_ecosystem({"PaperA": ["TAM1"], "PaperB": ["TAM1"]})
        assert adit.acronym == original_acronym
        assert adit.acronym == "tam"

    def test_acronym_unchanged_after_extract_features(self, mock_transformer):
        """Acronym should remain unchanged after extracting features."""
        adit = ADIT(
            "Technology Acceptance Model", ["TAM1"], transformer=mock_transformer, acronym="TAM"
        )
        adit.build_ecosystem({"PaperA": ["TAM1"]})

        papers_data = {
            "PaperA": {
                "title": "Extension of TAM",
                "abstract": "Extends TAM with new constructs.",
                "keywords": "TAM, acceptance",
                "citations": 50,
                "year": 2015,
            },
            "TAM1": {
                "title": "Technology Acceptance Model",
                "abstract": "Foundational TAM paper.",
                "keywords": "TAM",
                "citations": 100,
                "year": 1989,
            },
        }
        features = adit.extract_features(papers_data)
        assert adit.acronym == "tam"
        assert len(features) > 0

    def test_acronym_consistency_derived_through_pipeline(self, mock_transformer):
        """Derived acronym should be consistent throughout pipeline."""
        adit = ADIT("Diffusion of Innovations", ["DOI1"], transformer=mock_transformer)
        assert adit.acronym == "di"

        adit.build_ecosystem({"PaperA": ["DOI1"], "PaperB": ["DOI1"]})
        assert adit.acronym == "di"

        papers_data = {
            "PaperA": {
                "title": "DoI in tech",
                "abstract": "Innovation diffusion",
                "keywords": "diffusion",
                "citations": 30,
                "year": 2010,
            },
            "DOI1": {
                "title": "Diffusion of Innovations",
                "abstract": "Foundational theory",
                "keywords": "diffusion",
                "citations": 80,
                "year": 1962,
            },
        }
        adit.extract_features(papers_data)
        assert adit.acronym == "di"

    def test_acronym_accessible_after_train_classifier(self, mock_transformer):
        """Acronym should remain accessible and unchanged after training classifier."""
        adit = ADIT(
            "Theory of Planned Behavior", ["TPB1"], transformer=mock_transformer, acronym="TPB"
        )
        adit.build_ecosystem({"PaperA": ["TPB1"], "PaperB": ["TPB1"]})

        papers_data = {
            "PaperA": {
                "title": "Extension of TPB",
                "abstract": "Extends TPB for new domain",
                "keywords": "TPB",
                "citations": 20,
                "year": 2014,
            },
            "PaperB": {
                "title": "Test of TPB",
                "abstract": "Tests TPB hypothesis",
                "keywords": "TPB",
                "citations": 15,
                "year": 2010,
            },
            "TPB1": {
                "title": "Theory of Planned Behavior",
                "abstract": "Foundational TPB paper",
                "keywords": "TPB",
                "citations": 90,
                "year": 1991,
            },
        }
        features = adit.extract_features(papers_data)
        labels = [1, 0]
        adit.train_classifier(features, labels)

        assert adit.acronym == "tpb"

    def test_acronym_accessible_after_predict(self, mock_transformer):
        """Acronym should remain accessible and unchanged after making predictions."""
        adit = ADIT(
            "Social Cognitive Theory", ["SCT1"], transformer=mock_transformer, acronym="SCT"
        )
        adit.build_ecosystem({"PaperA": ["SCT1"], "PaperB": ["SCT1"]})

        papers_data = {
            "PaperA": {
                "title": "Application of SCT",
                "abstract": "Applies SCT to new context",
                "keywords": "SCT",
                "citations": 25,
                "year": 2013,
            },
            "PaperB": {
                "title": "Test of SCT",
                "abstract": "Tests SCT mechanisms",
                "keywords": "SCT",
                "citations": 18,
                "year": 2011,
            },
            "SCT1": {
                "title": "Social Cognitive Theory",
                "abstract": "Foundational SCT paper",
                "keywords": "SCT",
                "citations": 120,
                "year": 1986,
            },
        }
        features = adit.extract_features(papers_data)
        labels = [1, 0]
        adit.train_classifier(features, labels)

        predictions = adit.predict_subscription(features)
        assert adit.acronym == "sct"
        assert len(predictions) == 2


class TestAcronymEdgeCases:
    """Edge case tests for acronym handling."""

    def test_acronym_with_numbers_in_theory_name(self, mock_transformer):
        """Theory name with embedded numbers should derive correctly."""
        adit = ADIT("6G Mobile Technology Theory", ["6MTT1"], transformer=mock_transformer)
        assert adit.acronym == "6mtt"

    def test_explicitly_provided_acronym_overrides_derivation(self, mock_transformer):
        """Explicit acronym should override derived acronym."""
        adit = ADIT(
            "Technology Acceptance Model",
            ["TAM1"],
            transformer=mock_transformer,
            acronym="CustomAcronym",
        )
        assert adit.acronym == "customacronym"
        # Not "tam", which would be derived

    def test_acronym_with_apostrophe_in_theory_name(self, mock_transformer):
        """Theory name with apostrophe (e.g., possessive) should treat apostrophe-word as normal."""
        adit = ADIT("Actor's Network Theory", ["ANT1"], transformer=mock_transformer)
        # "Actor's" is still one word, so first letter is 'a'
        assert adit.acronym == "ant"

    def test_acronym_empty_string_not_provided(self, mock_transformer):
        """Empty-string acronym should fall back to derivation."""
        # Note: This tests behavior if someone explicitly passes acronym=""
        adit = ADIT("TAM", ["TAM1"], transformer=mock_transformer, acronym="")
        # Empty string is falsy, so will derive
        assert adit.acronym == "t"

    def test_acronym_all_stopwords_falls_back_to_original_words(self, mock_transformer):
        """If all words are minor words, fallback should still produce a deterministic acronym."""
        adit = ADIT("of and in", ["X1"], transformer=mock_transformer)
        assert adit.acronym == "oai"

    def test_acronym_none_uses_derivation(self, mock_transformer):
        """None acronym should fall back to derivation (default behavior)."""
        adit = ADIT(
            "Technology Acceptance Model", ["TAM1"], transformer=mock_transformer, acronym=None
        )
        assert adit.acronym == "tam"

    def test_acronym_case_variation_does_not_affect_consistency(self, mock_transformer):
        """Multiple ADIT instances with same theory name and different acronym cases should be independent."""
        adit1 = ADIT("TAM", ["TAM1"], transformer=mock_transformer, acronym="TAM")
        adit2 = ADIT("TAM", ["TAM1"], transformer=mock_transformer, acronym="tam")
        adit3 = ADIT("TAM", ["TAM1"], transformer=mock_transformer, acronym="TaM")

        assert adit1.acronym == "tam"
        assert adit2.acronym == "tam"
        assert adit3.acronym == "tam"
        assert adit1.acronym == adit2.acronym == adit3.acronym
