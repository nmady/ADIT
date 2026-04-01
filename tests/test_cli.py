"""Tests for Typer CLI argument and config-file execution paths."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from typer.testing import CliRunner

import cli
from adit import derive_acronym


class FakeADIT:
    """Lightweight ADIT test double to avoid model loading in CLI tests."""

    last_instance = None

    def __init__(self, theory_name, l1_papers, acronym=None, key_constructs=None):
        self.theory_name = theory_name
        self.l1_papers = l1_papers
        self.acronym = acronym.lower() if acronym else derive_acronym(theory_name)
        self.key_constructs = key_constructs or []
        self.trained = False
        FakeADIT.last_instance = self

    def build_ecosystem(self, citation_data):
        self.citation_data = citation_data

    def extract_features(self, papers_data):
        self.papers_data = papers_data
        return pd.DataFrame(
            {
                "paper_id": ["PaperA", "PaperB"],
                "eigenfactor": [1.0, 0.5],
                "betweenness": [0.0, 0.1],
            }
        )

    def train_classifier(self, features_df, labels):
        self.trained = True
        self.labels = labels

    def predict_subscription(self, features_df):
        return np.array([1, 0])


runner = CliRunner()


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cli_run_with_direct_args_and_labels(tmp_path, monkeypatch):
    """CLI should run with direct args, train, and write output files."""
    monkeypatch.setattr(cli, "ADIT", FakeADIT)

    citation_file = tmp_path / "citation.json"
    papers_file = tmp_path / "papers.json"
    labels_file = tmp_path / "labels.json"
    features_out = tmp_path / "out" / "features.csv"
    predictions_out = tmp_path / "out" / "predictions.csv"

    _write_json(citation_file, {"PaperA": ["TAM1"]})
    _write_json(
        papers_file,
        {
            "PaperA": {
                "title": "A",
                "abstract": "a",
                "keywords": "k",
                "citations": 1,
                "year": 2010,
            },
            "TAM1": {
                "title": "TAM",
                "abstract": "foundation",
                "keywords": "tam",
                "citations": 100,
                "year": 1990,
            },
        },
    )
    _write_json(labels_file, {"PaperA": 1, "PaperB": 0})

    result = runner.invoke(
        cli.app,
        [
            "--theory-name",
            "Technology Acceptance Model",
            "--acronym",
            "TAM",
            "--l1-papers",
            "TAM1,TAM2",
            "--citation-data",
            str(citation_file),
            "--papers-data",
            str(papers_file),
            "--labels-data",
            str(labels_file),
            "--output-features",
            str(features_out),
            "--output-predictions",
            str(predictions_out),
        ],
        color=False,
    )

    assert result.exit_code == 0, result.output
    assert "Extracted 2 L2 feature rows." in result.output
    assert "Generated 2 predictions." in result.output

    assert features_out.exists()
    assert predictions_out.exists()

    predictions_df = pd.read_csv(predictions_out)
    assert list(predictions_df.columns) == ["paper_id", "prediction"]
    assert predictions_df["prediction"].tolist() == [1, 0]

    inst = FakeADIT.last_instance
    assert inst is not None
    assert inst.theory_name == "Technology Acceptance Model"
    assert inst.acronym == "tam"
    assert inst.l1_papers == ["TAM1", "TAM2"]
    assert inst.trained is True


def test_cli_run_with_json_config_and_no_labels(tmp_path, monkeypatch):
    """Config-file mode should load defaults and skip training if labels are absent."""
    monkeypatch.setattr(cli, "ADIT", FakeADIT)

    citation_file = tmp_path / "citation.json"
    papers_file = tmp_path / "papers.json"
    config_file = tmp_path / "config.json"

    _write_json(citation_file, {"PaperA": ["TAM1"]})
    _write_json(
        papers_file,
        {
            "PaperA": {
                "title": "A",
                "abstract": "a",
                "keywords": "k",
                "citations": 1,
                "year": 2010,
            },
            "TAM1": {
                "title": "TAM",
                "abstract": "foundation",
                "keywords": "tam",
                "citations": 100,
                "year": 1990,
            },
        },
    )
    _write_json(
        config_file,
        {
            "theory_name": "Technology Acceptance Model",
            "acronym": "TAM",
            "l1_papers": ["TAM1", "TAM2"],
            "citation_data": str(citation_file),
            "papers_data": str(papers_file),
        },
    )

    result = runner.invoke(cli.app, ["--config", str(config_file)])

    assert result.exit_code == 0, result.output
    assert "Using theory='Technology Acceptance Model' acronym='tam'" in result.output
    assert "No labels_data provided; skipped training/prediction." in result.output

    inst = FakeADIT.last_instance
    assert inst is not None
    assert inst.l1_papers == ["TAM1", "TAM2"]
    assert inst.trained is False


def test_cli_cli_args_override_config_values(tmp_path, monkeypatch):
    """Direct CLI options should take precedence over config values."""
    monkeypatch.setattr(cli, "ADIT", FakeADIT)

    citation_file = tmp_path / "citation.json"
    papers_file = tmp_path / "papers.json"
    config_file = tmp_path / "config.json"

    _write_json(citation_file, {"PaperA": ["TAM1"]})
    _write_json(
        papers_file,
        {
            "PaperA": {
                "title": "A",
                "abstract": "a",
                "keywords": "k",
                "citations": 1,
                "year": 2010,
            },
            "TAM1": {
                "title": "TAM",
                "abstract": "foundation",
                "keywords": "tam",
                "citations": 100,
                "year": 1990,
            },
        },
    )
    _write_json(
        config_file,
        {
            "theory_name": "Config Theory",
            "acronym": "CFG",
            "l1_papers": ["CFG1"],
            "citation_data": str(citation_file),
            "papers_data": str(papers_file),
        },
    )

    result = runner.invoke(
        cli.app,
        [
            "--config",
            str(config_file),
            "--theory-name",
            "Overridden Theory",
            "--acronym",
            "OVR",
            "--l1-papers",
            "TAM1,TAM2",
        ],
    )

    assert result.exit_code == 0, result.output

    inst = FakeADIT.last_instance
    assert inst is not None
    assert inst.theory_name == "Overridden Theory"
    assert inst.acronym == "ovr"
    assert inst.l1_papers == ["TAM1", "TAM2"]


def test_cli_requires_theory_and_data(tmp_path):
    """CLI should fail with a clear message when required inputs are missing."""
    result = runner.invoke(cli.app, [])
    assert result.exit_code != 0
    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "Provide L1 papers via --l1-papers or --l1-file" in clean_output

    config_file = tmp_path / "bad_config.json"
    _write_json(config_file, {"theory_name": "Only Theory"})

    result2 = runner.invoke(cli.app, ["--config", str(config_file), "--l1-papers", "TAM1"])
    assert result2.exit_code != 0
    clean_output2 = re.sub(r"\x1b\[[0-9;]*m", "", result2.output)
    assert "citation_data path is required" in clean_output2


def test_cli_online_mode_uses_ingestion_without_local_files(monkeypatch):
    """Online mode should ingest citation inputs without requiring local JSON paths."""
    monkeypatch.setattr(cli, "ADIT", FakeADIT)

    def fake_ingest(**kwargs):
        assert kwargs["theory_name"] == "Technology Acceptance Model"
        assert kwargs["l1_papers"] == ["TAM1", "TAM2"]
        assert kwargs["depth"] == "l2"
        assert kwargs["sources"] == ["openalex", "crossref"]
        assert kwargs["max_l3"] is None
        return SimpleNamespace(
            citation_data={"PaperA": ["TAM1"]},
            papers_data={
                "PaperA": {
                    "title": "A",
                    "abstract": "a",
                    "keywords": "k",
                    "citations": 1,
                    "year": 2010,
                },
                "TAM1": {
                    "title": "TAM",
                    "abstract": "foundation",
                    "keywords": "tam",
                    "citations": 100,
                    "year": 1990,
                },
            },
            metadata={"paper_count": 2, "edge_count": 1},
        )

    monkeypatch.setattr(cli, "ingest_from_internet", fake_ingest)

    result = runner.invoke(
        cli.app,
        [
            "--online",
            "--theory-name",
            "Technology Acceptance Model",
            "--l1-papers",
            "TAM1,TAM2",
            "--sources",
            "openalex,crossref",
            "--depth",
            "l2",
        ],
        color=False,
    )

    assert result.exit_code == 0, result.output
    assert "Online ingestion complete:" in result.output
    assert "Extracted 2 L2 feature rows." in result.output


def test_cli_online_mode_honors_configured_max_l3_budget(tmp_path, monkeypatch):
    """Config should pass an explicit max_l3 budget to ingestion."""
    monkeypatch.setattr(cli, "ADIT", FakeADIT)

    config_file = tmp_path / "online-config.json"
    _write_json(
        config_file,
        {
            "online": True,
            "theory_name": "Technology Acceptance Model",
            "l1_papers": ["TAM1", "TAM2"],
            "sources": ["openalex", "crossref"],
            "depth": "l2",
            "max_l3": 25,
        },
    )

    def fake_ingest(**kwargs):
        assert kwargs["max_l3"] == 25
        return SimpleNamespace(
            citation_data={"PaperA": ["TAM1"]},
            papers_data={
                "PaperA": {
                    "title": "A",
                    "abstract": "a",
                    "keywords": "k",
                    "citations": 1,
                    "year": 2010,
                },
                "TAM1": {
                    "title": "TAM",
                    "abstract": "foundation",
                    "keywords": "tam",
                    "citations": 100,
                    "year": 1990,
                },
            },
            metadata={"paper_count": 2, "edge_count": 1},
        )

    monkeypatch.setattr(cli, "ingest_from_internet", fake_ingest)

    result = runner.invoke(cli.app, ["--config", str(config_file)], color=False)

    assert result.exit_code == 0, result.output
    assert "Online ingestion complete:" in result.output


def test_cli_online_mode_default_sources_include_core(monkeypatch):
    """Online mode without --sources should include CORE by default."""
    monkeypatch.setattr(cli, "ADIT", FakeADIT)

    def fake_ingest(**kwargs):
        assert kwargs["sources"] == ["openalex", "semantic_scholar", "crossref", "core"]
        return SimpleNamespace(
            citation_data={"PaperA": ["TAM1"]},
            papers_data={
                "PaperA": {
                    "title": "A",
                    "abstract": "a",
                    "keywords": "k",
                    "citations": 1,
                    "year": 2010,
                },
                "TAM1": {
                    "title": "TAM",
                    "abstract": "foundation",
                    "keywords": "tam",
                    "citations": 100,
                    "year": 1990,
                },
            },
            metadata={"paper_count": 2, "edge_count": 1},
        )

    monkeypatch.setattr(cli, "ingest_from_internet", fake_ingest)
    monkeypatch.delenv("CORE_API_KEY", raising=False)

    result = runner.invoke(
        cli.app,
        [
            "--online",
            "--theory-name",
            "Technology Acceptance Model",
            "--l1-papers",
            "TAM1,TAM2",
            "--depth",
            "l2",
        ],
        color=False,
    )

    assert result.exit_code == 0, result.output
    assert "Warning: CORE selected without CORE_API_KEY." in result.output


def test_cli_online_mode_rejects_invalid_depth(monkeypatch):
    """Online mode should validate supported depth values."""
    monkeypatch.setattr(cli, "ADIT", FakeADIT)

    result = runner.invoke(
        cli.app,
        [
            "--online",
            "--theory-name",
            "Technology Acceptance Model",
            "--l1-papers",
            "TAM1,TAM2",
            "--depth",
            "bad-depth",
        ],
        color=False,
    )

    assert result.exit_code != 0
    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "depth must be either 'l2' or 'l2l3'" in clean_output


def test_cli_only_ingest_skips_adit_pipeline(monkeypatch):
    """--only-ingest should run ingestion and exit before ADIT feature extraction/training."""

    class ShouldNotInstantiateADIT:
        def __init__(self, *args, **kwargs):
            raise AssertionError("ADIT should not be constructed in only-ingest mode")

    monkeypatch.setattr(cli, "ADIT", ShouldNotInstantiateADIT)

    def fake_ingest(**kwargs):
        return SimpleNamespace(
            citation_data={"PaperA": ["TAM1"]},
            papers_data={
                "PaperA": {
                    "title": "A",
                    "abstract": "a",
                    "keywords": "k",
                    "citations": 1,
                    "year": 2010,
                },
                "TAM1": {
                    "title": "TAM",
                    "abstract": "foundation",
                    "keywords": "tam",
                    "citations": 100,
                    "year": 1990,
                },
            },
            metadata={"paper_count": 2, "edge_count": 1},
        )

    monkeypatch.setattr(cli, "ingest_from_internet", fake_ingest)

    result = runner.invoke(
        cli.app,
        [
            "--online",
            "--only-ingest",
            "--theory-name",
            "Technology Acceptance Model",
            "--l1-papers",
            "TAM1,TAM2",
            "--sources",
            "openalex,crossref",
            "--depth",
            "l2",
        ],
        color=False,
    )

    assert result.exit_code == 0, result.output
    assert "Online ingestion complete:" in result.output
    assert "Ingestion-only mode enabled" in result.output
    assert "Extracted" not in result.output


def test_cli_online_checkpoint_options_are_passed_to_ingestion(monkeypatch):
    """CLI checkpoint flags should pass through to ingest_from_internet in online mode."""
    monkeypatch.setattr(cli, "ADIT", FakeADIT)
    captured_kwargs = {}

    def fake_ingest(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            citation_data={"PaperA": ["TAM1"]},
            papers_data={
                "PaperA": {
                    "title": "A",
                    "abstract": "a",
                    "keywords": "k",
                    "citations": 1,
                    "year": 2010,
                },
                "TAM1": {
                    "title": "TAM",
                    "abstract": "foundation",
                    "keywords": "tam",
                    "citations": 100,
                    "year": 1990,
                },
            },
            metadata={"paper_count": 2, "edge_count": 1},
        )

    monkeypatch.setattr(cli, "ingest_from_internet", fake_ingest)

    result = runner.invoke(
        cli.app,
        [
            "--online",
            "--theory-name",
            "Technology Acceptance Model",
            "--l1-papers",
            "TAM1,TAM2",
            "--depth",
            "l2",
            "--checkpoint-dir",
            "custom-checkpoints",
            "--checkpoint-staleness-seconds",
            "1800",
            "--reset-checkpoints",
            "--quiet",
        ],
        color=False,
    )

    assert result.exit_code == 0, result.output
    assert captured_kwargs["checkpoint_dir"] == Path("custom-checkpoints")
    assert captured_kwargs["checkpoint_staleness_seconds"] == 1800
    assert captured_kwargs["reset_checkpoints"] is True
    assert captured_kwargs["quiet"] is True


def test_cli_online_defaults_quiet_to_false(monkeypatch):
    """Online mode should pass quiet=False unless requested."""
    monkeypatch.setattr(cli, "ADIT", FakeADIT)
    captured_kwargs = {}

    def fake_ingest(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            citation_data={"PaperA": ["TAM1"]},
            papers_data={
                "PaperA": {
                    "title": "A",
                    "abstract": "a",
                    "keywords": "k",
                    "citations": 1,
                    "year": 2010,
                },
                "TAM1": {
                    "title": "TAM",
                    "abstract": "foundation",
                    "keywords": "tam",
                    "citations": 100,
                    "year": 1990,
                },
            },
            metadata={"paper_count": 2, "edge_count": 1},
        )

    monkeypatch.setattr(cli, "ingest_from_internet", fake_ingest)

    result = runner.invoke(
        cli.app,
        [
            "--online",
            "--theory-name",
            "Technology Acceptance Model",
            "--l1-papers",
            "TAM1,TAM2",
            "--depth",
            "l2",
        ],
        color=False,
    )

    assert result.exit_code == 0, result.output
    assert captured_kwargs["quiet"] is False


def test_cli_online_checkpoint_config_values_are_resolved(tmp_path, monkeypatch):
    """Config-file checkpoint settings should be resolved and passed to ingestion."""
    monkeypatch.setattr(cli, "ADIT", FakeADIT)
    captured_kwargs = {}

    config_file = tmp_path / "online-config.json"
    _write_json(
        config_file,
        {
            "online": True,
            "theory_name": "Technology Acceptance Model",
            "l1_papers": ["TAM1", "TAM2"],
            "depth": "l2",
            "checkpoint_dir": "cfg-checkpoints",
            "checkpoint_staleness_seconds": 900,
            "reset_checkpoints": True,
        },
    )

    def fake_ingest(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            citation_data={"PaperA": ["TAM1"]},
            papers_data={
                "PaperA": {
                    "title": "A",
                    "abstract": "a",
                    "keywords": "k",
                    "citations": 1,
                    "year": 2010,
                },
                "TAM1": {
                    "title": "TAM",
                    "abstract": "foundation",
                    "keywords": "tam",
                    "citations": 100,
                    "year": 1990,
                },
            },
            metadata={"paper_count": 2, "edge_count": 1},
        )

    monkeypatch.setattr(cli, "ingest_from_internet", fake_ingest)

    result = runner.invoke(cli.app, ["--config", str(config_file)], color=False)

    assert result.exit_code == 0, result.output
    assert captured_kwargs["checkpoint_dir"] == Path("cfg-checkpoints")
    assert captured_kwargs["checkpoint_staleness_seconds"] == 900
    assert captured_kwargs["reset_checkpoints"] is True


def test_cli_online_checkpoint_staleness_must_be_positive(monkeypatch):
    """checkpoint_staleness_seconds should reject non-positive values."""
    monkeypatch.setattr(cli, "ADIT", FakeADIT)

    result = runner.invoke(
        cli.app,
        [
            "--online",
            "--theory-name",
            "Technology Acceptance Model",
            "--l1-papers",
            "TAM1,TAM2",
            "--checkpoint-staleness-seconds",
            "0",
        ],
        color=False,
    )

    assert result.exit_code != 0
    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "checkpoint_staleness_seconds must be a positive integer" in clean_output
