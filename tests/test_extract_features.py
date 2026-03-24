import numpy as np
import pandas as pd

from adit import ADIT


def test_extract_features_columns_and_values(
    mock_transformer, sample_citation_data, sample_papers_data
):
    adit = ADIT(
        "TAM",
        ["TAM1"],
        transformer=mock_transformer,
        key_constructs=["usefulness", "ease of use", "acceptance", "intention", "attitude"],
    )
    adit.build_ecosystem(sample_citation_data)

    features = adit.extract_features(sample_papers_data)

    # Basic checks: DataFrame and expected columns
    assert isinstance(features, pd.DataFrame)
    expected_cols = [
        "paper_id",
        "eigenfactor",
        "betweenness",
        "theory_attribution_ratio",
        "citation_count",
        "pub_year",
        "abstract_word_count",
        "theory_in_title",
        "theory_in_keywords",
        "theory_in_abstract",
        "acronym_in_title",
        "acronym_in_keywords",
        "acronym_in_abstract",
        "has_usefulness",
        "has_ease_of_use",
        "has_acceptance",
        "has_intention",
        "has_attitude",
        "semantic_similarity",
        "in_degree",
        "out_degree",
    ]

    for col in expected_cols:
        assert col in features.columns

    # Semantic similarity should be deterministic (mock returns identical unit vectors)
    assert np.allclose(features["semantic_similarity"].values, 1.0)

    # Binary flags should be 0/1
    for flag in [
        "has_usefulness",
        "has_acceptance",
        "has_ease_of_use",
        "has_intention",
        "has_attitude",
    ]:
        assert set(features[flag].unique()).issubset({0, 1})
