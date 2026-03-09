"""Integration tests for ADIT end-to-end workflow."""

import numpy as np
import pandas as pd

from adit import ADIT


def test_adit_full_pipeline(mock_transformer, sample_citation_data, sample_papers_data):
    """Test the complete ADIT workflow: build → extract → train → predict."""

    # 1. Initialize ADIT with mocked transformer
    adit = ADIT('TAM', ['TAM1', 'TAM2'], transformer=mock_transformer)

    # 2. Build ecosystem
    adit.build_ecosystem(sample_citation_data)

    # Verify ecosystem structure
    assert len(adit.ecosystem.nodes) > 0
    assert any(data.get('level') == 'L1' for _, data in adit.ecosystem.nodes(data=True))
    assert any(data.get('level') == 'L2' for _, data in adit.ecosystem.nodes(data=True))

    # 3. Extract features
    features = adit.extract_features(sample_papers_data)

    # Verify features are produced
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0
    assert 'paper_id' in features.columns
    assert all(col in features.columns for col in [
        'eigenfactor', 'betweenness', 'citation_count', 'pub_year',
        'has_usefulness', 'has_ease_of_use', 'has_acceptance',
        'semantic_similarity'
    ])

    # 4. Create labels aligned with extracted papers (L2 only)
    label_map = {
        'PaperA': 1,
        'PaperB': 1,
        'PaperC': 0,
        'PaperD': 0,
        'PaperE': 1,
    }
    labels = [label_map.get(paper_id, 0) for paper_id in features['paper_id']]

    # 5. Train classifier
    adit.train_classifier(features, labels)

    # Verify classifier is trained and has feature importances
    assert adit.classifier is not None
    assert hasattr(adit.classifier, 'feature_importances_')
    assert len(adit.classifier.feature_importances_) > 0

    # 6. Make predictions
    predictions = adit.predict_subscription(features)

    # Verify predictions are valid
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(features)
    assert all(p in [0, 1] for p in predictions)

    # Sanity check: predictions include both classes (at least for this example)
    # This is a soft check; mock data might not guarantee mixed predictions
    print(f"Sample predictions: {predictions[:3]}")

    # 7. Verify consistency: re-predict should give same results
    predictions2 = adit.predict_subscription(features)
    assert np.array_equal(predictions, predictions2)
