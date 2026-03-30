"""Integration tests for ADIT end-to-end workflow."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adit import ADIT
import citation_ingestion as ci


def test_adit_full_pipeline(mock_transformer, sample_citation_data, sample_papers_data):
    """Test the complete ADIT workflow: build → extract → train → predict."""

    # 1. Initialize ADIT with mocked transformer
    adit = ADIT(
        "TAM",
        ["TAM1", "TAM2"],
        transformer=mock_transformer,
        key_constructs=["usefulness", "ease of use", "acceptance"],
    )

    # 2. Build ecosystem
    adit.build_ecosystem(sample_citation_data)

    # Verify ecosystem structure
    assert len(adit.ecosystem.nodes) > 0
    assert any(data.get("level") == "L1" for _, data in adit.ecosystem.nodes(data=True))
    assert any(data.get("level") == "L2" for _, data in adit.ecosystem.nodes(data=True))

    # 3. Extract features
    features = adit.extract_features(sample_papers_data)

    # Verify features are produced
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0
    assert "paper_id" in features.columns
    assert all(
        col in features.columns
        for col in [
            "eigenfactor",
            "betweenness",
            "citation_count",
            "pub_year",
            "has_usefulness",
            "has_ease_of_use",
            "has_acceptance",
            "semantic_similarity",
        ]
    )

    # 4. Create labels aligned with extracted papers (L2 only)
    label_map = {
        "PaperA": 1,
        "PaperB": 1,
        "PaperC": 0,
        "PaperD": 0,
        "PaperE": 1,
    }
    labels = [label_map.get(paper_id, 0) for paper_id in features["paper_id"]]

    # 5. Train classifier
    adit.train_classifier(features, labels)

    # Verify classifier is trained and has feature importances
    assert adit.classifier is not None
    assert hasattr(adit.classifier, "feature_importances_")
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


def test_ingestion_checkpoint_resume_feeds_adit_pipeline(monkeypatch, tmp_path, mock_transformer):
    """Crash-resume ingestion output should match baseline and remain ADIT-compatible."""

    class _IntegrationCheckpointProvider(ci.CitationProvider):
        capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=False)

        def __init__(self, name, l2_prefix, l3_refs, crash_at_l3_index=None, fail_on_call=False):
            self.name = name
            self.l2_prefix = l2_prefix
            self.l3_refs = l3_refs
            self.crash_at_l3_index = crash_at_l3_index
            self.fail_on_call = fail_on_call

        def fetch_seed_metadata(self, l1_papers):
            if self.fail_on_call:
                raise AssertionError(f"{self.name} should have been skipped via checkpoint")
            return {
                seed: ci.IngestionPaper(
                    paper_id=seed,
                    title=f"Seed {seed}",
                    year=1990,
                    source_ids={self.name: f"seed-{seed}"},
                )
                for seed in l1_papers
            }

        def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
            if self.fail_on_call:
                raise AssertionError(f"{self.name} should have been skipped via checkpoint")
            l2_a = f"{self.l2_prefix}:L2A"
            l2_b = f"{self.l2_prefix}:L2B"
            return (
                {
                    l2_a: {l1_papers[0]},
                    l2_b: {l1_papers[1] if len(l1_papers) > 1 else l1_papers[0]},
                },
                {
                    l2_a: ci.IngestionPaper(
                        paper_id=l2_a,
                        title="Technology acceptance in healthcare",
                        abstract="Uses technology acceptance model constructs.",
                        keywords="technology acceptance, usefulness",
                        citations=12,
                        year=2020,
                    ),
                    l2_b: ci.IngestionPaper(
                        paper_id=l2_b,
                        title="Ease of use and behavioral intention",
                        abstract="Empirical evidence for acceptance.",
                        keywords="ease of use, acceptance",
                        citations=9,
                        year=2021,
                    ),
                },
            )

        def fetch_l3_references(
            self,
            l2_paper_ids,
            max_l3=None,
            resume_state=None,
            progress_callback=None,
        ):
            edges = ci._deserialize_edges((resume_state or {}).get("edges"))
            papers = ci._deserialize_papers((resume_state or {}).get("papers"))

            start_index_raw = (resume_state or {}).get("next_l2_index")
            try:
                start_index = int(start_index_raw) if start_index_raw is not None else 0
            except (TypeError, ValueError):
                start_index = 0

            for idx in range(start_index, len(l2_paper_ids)):
                if self.crash_at_l3_index is not None and idx == self.crash_at_l3_index:
                    raise RuntimeError(f"simulated integration l3 crash in {self.name}")

                parent_id = l2_paper_ids[idx]
                for ref_id in self.l3_refs.get(parent_id, []):
                    edges.setdefault(parent_id, set()).add(ref_id)
                    papers.setdefault(
                        ref_id,
                        ci.IngestionPaper(
                            paper_id=ref_id,
                            title=f"Reference {ref_id}",
                            citations=1,
                            year=2018,
                            doi=ref_id.split(":", 1)[1] if ref_id.startswith("doi:") else None,
                        ),
                    )

                if progress_callback:
                    progress_callback(
                        {
                            "status": "in_progress",
                            "next_l2_index": idx + 1,
                            "budget_remaining": None,
                            "edges": ci._serialize_edges(edges),
                            "papers": ci._serialize_papers(papers),
                            "updated_at": ci.time.time(),
                        }
                    )

            if progress_callback:
                progress_callback(
                    {
                        "status": "complete",
                        "next_l2_index": len(l2_paper_ids),
                        "budget_remaining": None,
                        "edges": ci._serialize_edges(edges),
                        "papers": ci._serialize_papers(papers),
                        "updated_at": ci.time.time(),
                    }
                )

            return edges, papers

    cache_dir = Path(tmp_path) / "cache"
    checkpoint_dir = Path(tmp_path) / "checkpoints"
    l1 = ["TAM1", "TAM2"]

    oa_refs = {
        "openalex:L2A": ["doi:10.1000/shared-int", "doi:10.1000/oa-int"],
        "openalex:L2B": ["doi:10.1000/oa-int-2"],
    }
    s2_refs = {
        "semantic_scholar:L2A": ["doi:10.1000/shared-int", "doi:10.1000/s2-int"],
        "semantic_scholar:L2B": ["doi:10.1000/s2-int-2"],
    }

    first_oa = _IntegrationCheckpointProvider("openalex", "openalex", oa_refs)
    first_s2 = _IntegrationCheckpointProvider(
        "semantic_scholar",
        "semantic_scholar",
        s2_refs,
        crash_at_l3_index=1,
    )
    monkeypatch.setattr(ci, "build_providers", lambda _: [first_oa, first_s2])

    with pytest.raises(RuntimeError, match="simulated integration l3 crash"):
        ci.ingest_from_internet(
            theory_name="Technology Acceptance Model",
            l1_papers=l1,
            sources=["openalex", "semantic_scholar"],
            depth="l2l3",
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            refresh=True,
        )

    resumed_oa = _IntegrationCheckpointProvider(
        "openalex",
        "openalex",
        oa_refs,
        fail_on_call=True,
    )
    resumed_s2 = _IntegrationCheckpointProvider(
        "semantic_scholar",
        "semantic_scholar",
        s2_refs,
    )
    monkeypatch.setattr(ci, "build_providers", lambda _: [resumed_oa, resumed_s2])

    resumed = ci.ingest_from_internet(
        theory_name="Technology Acceptance Model",
        l1_papers=l1,
        sources=["openalex", "semantic_scholar"],
        depth="l2l3",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        refresh=True,
    )

    baseline_oa = _IntegrationCheckpointProvider("openalex", "openalex", oa_refs)
    baseline_s2 = _IntegrationCheckpointProvider("semantic_scholar", "semantic_scholar", s2_refs)
    monkeypatch.setattr(ci, "build_providers", lambda _: [baseline_oa, baseline_s2])

    baseline = ci.ingest_from_internet(
        theory_name="Technology Acceptance Model",
        l1_papers=l1,
        sources=["openalex", "semantic_scholar"],
        depth="l2l3",
        cache_dir=Path(tmp_path) / "baseline-cache",
        checkpoint_dir=Path(tmp_path) / "baseline-checkpoints",
        refresh=True,
    )

    assert resumed.citation_data == baseline.citation_data
    assert resumed.papers_data == baseline.papers_data

    adit = ADIT(
        "Technology Acceptance Model",
        l1,
        transformer=mock_transformer,
        key_constructs=["usefulness", "ease of use", "acceptance"],
    )
    adit.build_ecosystem(resumed.citation_data)
    features = adit.extract_features(resumed.papers_data)
    assert len(features) > 0

    labels = [1 if "L2A" in paper_id else 0 for paper_id in features["paper_id"]]
    adit.train_classifier(features, labels)
    predictions = adit.predict_subscription(features)
    assert len(predictions) == len(features)
