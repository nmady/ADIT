import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import typer

from adit import ADIT

app = typer.Typer(help="ADIT CLI: run with direct arguments or a config file.")


def _load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if not config_path:
        return {}

    if not config_path.exists():
        raise typer.BadParameter(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))

    if suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:
            raise typer.BadParameter("YAML config requires PyYAML installed.") from exc
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    raise typer.BadParameter("Config must be JSON or YAML (.json/.yml/.yaml).")


def _load_json_dict(path: Path, field_name: str) -> Dict[str, Any]:
    if not path.exists():
        raise typer.BadParameter(f"{field_name} file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise typer.BadParameter(f"{field_name} must be a JSON object/dict.")
    return data


def _parse_l1(l1_papers: Optional[str], l1_file: Optional[Path]) -> List[str]:
    if l1_papers:
        parsed = [item.strip() for item in l1_papers.split(",") if item.strip()]
        if parsed:
            return parsed

    if l1_file:
        if not l1_file.exists():
            raise typer.BadParameter(f"L1 file not found: {l1_file}")
        parsed = [
            line.strip()
            for line in l1_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if parsed:
            return parsed

    raise typer.BadParameter("Provide L1 papers via --l1-papers or --l1-file (or config values).")


def _resolve_labels(labels_data: Any, features: pd.DataFrame) -> List[int]:
    if isinstance(labels_data, list):
        if len(labels_data) != len(features):
            raise typer.BadParameter("labels list length must match extracted feature rows.")
        return [int(v) for v in labels_data]

    if isinstance(labels_data, dict):
        return [int(labels_data.get(pid, 0)) for pid in features["paper_id"]]

    raise typer.BadParameter("labels_data must be either a JSON list or dict.")


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, help="Path to JSON/YAML config file."),
    theory_name: Optional[str] = typer.Option(
        None, help="Theory name, e.g. 'Technology Acceptance Model'."
    ),
    acronym: Optional[str] = typer.Option(None, help="Optional explicit acronym, e.g. 'TAM'."),
    l1_papers: Optional[str] = typer.Option(
        None, help="Comma-separated L1 paper IDs/titles (alternative to --l1-file)."
    ),
    l1_file: Optional[Path] = typer.Option(None, help="Path to newline-separated L1 papers file."),
    citation_data: Optional[Path] = typer.Option(None, help="Path to citation JSON dict."),
    papers_data: Optional[Path] = typer.Option(None, help="Path to paper metadata JSON dict."),
    labels_data: Optional[Path] = typer.Option(
        None, help="Optional labels JSON (dict by paper_id, or list aligned to extracted features)."
    ),
    output_features: Optional[Path] = typer.Option(
        None, help="Optional CSV path for extracted features."
    ),
    output_predictions: Optional[Path] = typer.Option(
        None, help="Optional CSV path for predictions."
    ),
) -> None:
    """Run ADIT using CLI values and/or a config file."""
    cfg = _load_config(config)

    theory_name = theory_name or cfg.get("theory_name")
    acronym = acronym or cfg.get("acronym")

    l1_cfg = cfg.get("l1_papers")
    l1_cfg_str = ",".join(l1_cfg) if isinstance(l1_cfg, list) else None
    l1_file = l1_file or (Path(cfg["l1_file"]) if cfg.get("l1_file") else None)
    l1 = _parse_l1(l1_papers or l1_cfg_str, l1_file)

    citation_data_path = citation_data or (
        Path(cfg["citation_data"]) if cfg.get("citation_data") else None
    )
    papers_data_path = papers_data or (Path(cfg["papers_data"]) if cfg.get("papers_data") else None)
    labels_data_path = labels_data or (Path(cfg["labels_data"]) if cfg.get("labels_data") else None)

    output_features = output_features or (
        Path(cfg["output_features"]) if cfg.get("output_features") else None
    )
    output_predictions = output_predictions or (
        Path(cfg["output_predictions"]) if cfg.get("output_predictions") else None
    )

    if not theory_name:
        raise typer.BadParameter("theory_name is required (CLI arg or config).")
    if not citation_data_path:
        raise typer.BadParameter("citation_data path is required (CLI arg or config).")
    if not papers_data_path:
        raise typer.BadParameter("papers_data path is required (CLI arg or config).")

    citation_dict = _load_json_dict(citation_data_path, "citation_data")
    papers_dict = _load_json_dict(papers_data_path, "papers_data")

    adit = ADIT(theory_name=theory_name, l1_papers=l1, acronym=acronym)
    adit.build_ecosystem(citation_dict)
    features = adit.extract_features(papers_dict)

    typer.echo(f"Extracted {len(features)} L2 feature rows.")
    typer.echo(f"Using theory='{theory_name}' acronym='{adit.acronym}'")

    if output_features:
        output_features.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_features, index=False)
        typer.echo(f"Saved features to {output_features}")

    if not labels_data_path:
        typer.echo("No labels_data provided; skipped training/prediction.")
        return

    labels_raw = json.loads(labels_data_path.read_text(encoding="utf-8"))
    labels = _resolve_labels(labels_raw, features)
    adit.train_classifier(features, labels)
    predictions = adit.predict_subscription(features)

    predictions_df = pd.DataFrame({"paper_id": features["paper_id"], "prediction": predictions})
    typer.echo(f"Generated {len(predictions_df)} predictions.")

    if output_predictions:
        output_predictions.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(output_predictions, index=False)
        typer.echo(f"Saved predictions to {output_predictions}")
    else:
        typer.echo(predictions_df.to_string(index=False))


if __name__ == "__main__":
    app()
