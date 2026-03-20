"""Tests for ADIT classifier training and prediction methods."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from adit import ADIT


def _build_features_and_labels(mock_transformer, sample_citation_data, sample_papers_data):
    """Build a deterministic feature matrix and aligned labels for classifier tests."""
    adit = ADIT("TAM", ["TAM1", "TAM2"], transformer=mock_transformer)
    adit.build_ecosystem(sample_citation_data)
    features = adit.extract_features(sample_papers_data)

    label_map = {
        "PaperA": 1,
        "PaperB": 1,
        "PaperC": 0,
        "PaperD": 0,
        "PaperE": 1,
    }
    labels = [label_map.get(paper_id, 0) for paper_id in features["paper_id"]]
    return adit, features, labels


def test_train_classifier_fits_model(mock_transformer, sample_citation_data, sample_papers_data):
    """Training should fit the underlying classifier and expose feature importances."""
    adit, features, labels = _build_features_and_labels(
        mock_transformer, sample_citation_data, sample_papers_data
    )

    adit.train_classifier(features, labels)

    assert hasattr(adit.classifier, "feature_importances_")
    assert len(adit.classifier.feature_importances_) == len(features.columns) - 1


def test_train_classifier_excludes_paper_id_column(
    mock_transformer, sample_citation_data, sample_papers_data, mocker
):
    """`paper_id` should be excluded from X before classifier.fit is called."""
    adit, features, labels = _build_features_and_labels(
        mock_transformer, sample_citation_data, sample_papers_data
    )

    fit_spy = mocker.spy(adit.classifier, "fit")
    adit.train_classifier(features, labels)

    # fit(X_train, y_train) should receive only numeric feature columns
    # after paper_id is removed and values are transformed to ndarray.
    X_train_arg = fit_spy.call_args.args[0]
    assert isinstance(X_train_arg, np.ndarray)
    assert X_train_arg.ndim == 2
    assert X_train_arg.shape[1] == len(features.columns) - 1


def test_train_classifier_raises_on_label_length_mismatch(
    mock_transformer, sample_citation_data, sample_papers_data
):
    """A mismatched label count should raise ValueError from sklearn split/fit."""
    adit, features, labels = _build_features_and_labels(
        mock_transformer, sample_citation_data, sample_papers_data
    )

    bad_labels = labels[:-1]
    with pytest.raises(ValueError):
        adit.train_classifier(features, bad_labels)


def test_predict_subscription_raises_if_not_trained(
    mock_transformer, sample_citation_data, sample_papers_data
):
    """Predicting before training should raise NotFittedError."""
    adit, features, _ = _build_features_and_labels(
        mock_transformer, sample_citation_data, sample_papers_data
    )

    with pytest.raises(NotFittedError):
        adit.predict_subscription(features)


def test_predict_subscription_after_training_returns_binary_predictions(
    mock_transformer, sample_citation_data, sample_papers_data
):
    """After training, predictions should align with input size and be binary."""
    adit, features, labels = _build_features_and_labels(
        mock_transformer, sample_citation_data, sample_papers_data
    )

    adit.train_classifier(features, labels)
    predictions = adit.predict_subscription(features)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(features)
    assert set(np.unique(predictions)).issubset({0, 1})
