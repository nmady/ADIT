import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import typer

from adit import ADIT
from citation_ingestion import ingest_from_internet

app = typer.Typer(help="ADIT CLI: run with direct arguments or a config file.")

CONFIG_OPTION = typer.Option(None, help="Path to JSON/YAML config file.")
THEORY_NAME_OPTION = typer.Option(None, help="Theory name, e.g. 'Technology Acceptance Model'.")
ACRONYM_OPTION = typer.Option(None, help="Optional explicit acronym, e.g. 'TAM'.")
L1_PAPERS_OPTION = typer.Option(
    None, help="Comma-separated L1 paper IDs/titles (alternative to --l1-file)."
)
L1_FILE_OPTION = typer.Option(None, help="Path to newline-separated L1 papers file.")
CITATION_DATA_OPTION = typer.Option(None, help="Path to citation JSON dict.")
PAPERS_DATA_OPTION = typer.Option(None, help="Path to paper metadata JSON dict.")
LABELS_DATA_OPTION = typer.Option(
    None, help="Optional labels JSON (dict by paper_id, or list aligned to extracted features)."
)
ONLINE_OPTION = typer.Option(
    False,
    "--online",
    help="Fetch citation data and metadata from internet providers instead of local JSON files.",
)
SOURCES_OPTION = typer.Option(
    None,
    help="Comma-separated providers for online mode: openalex,semantic_scholar,crossref,core.",
)
DEPTH_OPTION = typer.Option("l2l3", help="Online expansion depth: l2 or l2l3.")
KEY_CONSTRUCTS_OPTION = typer.Option(
    None,
    help="Optional comma-separated key constructs to improve online retrieval relevance.",
)
CACHE_DIR_OPTION = typer.Option(None, help="Optional cache directory for online ingestion.")
CHECKPOINT_DIR_OPTION = typer.Option(
    None,
    help=(
        "Optional checkpoint directory for incremental online-ingestion progress snapshots. "
        "Defaults to <cache-dir>/checkpoints."
    ),
)
CHECKPOINT_STALENESS_OPTION = typer.Option(
    None,
    help=(
        "Optional checkpoint staleness window in seconds. "
        "Stale resume state older than this window is ignored."
    ),
)
REFRESH_CACHE_OPTION = typer.Option(
    False,
    help="Ignore ingestion cache and force fresh network retrieval in online mode.",
)
RESET_CHECKPOINTS_OPTION = typer.Option(
    False,
    help="Clear existing checkpoint state for this request before online ingestion starts.",
)
MAX_L2_OPTION = typer.Option(
    200,
    help="Maximum L2 papers to retrieve per provider in online mode.",
)
MAX_L3_OPTION = typer.Option(
    None,
    help=(
        "Optional per-provider cap on L3 reference edges in online mode. "
        "By default, all available L3 references are retrieved."
    ),
)
MAX_WORKERS_OPTION = typer.Option(
    None,
    "--max-workers",
    "-j",
    help=(
        "Optional provider-level parallel worker count for online ingestion. "
        "When omitted, ingestion runs sequentially."
    ),
)
SAVE_INGESTED_CITATION_OPTION = typer.Option(
    None,
    help="Optional output path to persist online-ingested citation_data JSON.",
)
SAVE_INGESTED_PAPERS_OPTION = typer.Option(
    None,
    help="Optional output path to persist online-ingested papers_data JSON.",
)
OUTPUT_FEATURES_OPTION = typer.Option(None, help="Optional CSV path for extracted features.")
OUTPUT_PREDICTIONS_OPTION = typer.Option(None, help="Optional CSV path for predictions.")
ONLY_INGEST_OPTION = typer.Option(
    False,
    help="Load or fetch citation/paper inputs and exit before feature extraction and ML training.",
)
EXHAUSTIVE_OPTION = typer.Option(
    True,
    "--exhaustive/--no-exhaustive",
    help=(
        "When enabled (default), providers that support cited-by traversal paginate "
        "until all citers are fetched. Use --no-exhaustive to cap retrieval at max-l2 "
        "results per L1/provider (faster but incomplete)."
    ),
)
VERBOSE_OPTION = typer.Option(
    False,
    "--verbose",
    "-v",
    help="Print detailed diagnostics to stderr (retry countdowns, per-seed internals).",
)
QUIET_OPTION = typer.Option(
    False,
    "--quiet",
    "-q",
    help="Suppress ingestion progress output on stderr.",
)


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


def _parse_key_constructs(raw_constructs: Any) -> List[str]:
    if isinstance(raw_constructs, list):
        return [str(item).strip() for item in raw_constructs if str(item).strip()]
    if raw_constructs is None:
        return []
    return [item.strip() for item in str(raw_constructs).split(",") if item.strip()]


def _resolve_cli_inputs(
    cfg: Dict[str, Any],
    theory_name: Optional[str],
    acronym: Optional[str],
    l1_papers: Optional[str],
    l1_file: Optional[Path],
    citation_data: Optional[Path],
    papers_data: Optional[Path],
    labels_data: Optional[Path],
    sources: Optional[str],
    depth: str,
    key_constructs: Optional[str],
    cache_dir: Optional[Path],
    checkpoint_dir: Optional[Path],
    checkpoint_staleness_seconds: Optional[int],
    refresh_cache: bool,
    reset_checkpoints: bool,
    max_l2: int,
    max_l3: Optional[int],
    max_workers: Optional[int],
    save_ingested_citation_data: Optional[Path],
    save_ingested_papers_data: Optional[Path],
    output_features: Optional[Path],
    output_predictions: Optional[Path],
    online: bool,
    only_ingest: bool,
    exhaustive: bool,
    verbose: bool,
    quiet: bool,
) -> Dict[str, Any]:
    l1_cfg = cfg.get("l1_papers")
    l1_cfg_str = ",".join(l1_cfg) if isinstance(l1_cfg, list) else None
    resolved_l1_file = l1_file or (Path(cfg["l1_file"]) if cfg.get("l1_file") else None)

    resolved_checkpoint_staleness = (
        int(cfg["checkpoint_staleness_seconds"])
        if cfg.get("checkpoint_staleness_seconds") is not None
        else checkpoint_staleness_seconds
    )
    if resolved_checkpoint_staleness is not None and resolved_checkpoint_staleness <= 0:
        raise typer.BadParameter("checkpoint_staleness_seconds must be a positive integer.")

    resolved_max_workers = (
        int(cfg["max_workers"]) if cfg.get("max_workers") is not None else max_workers
    )
    if resolved_max_workers is not None and resolved_max_workers <= 0:
        raise typer.BadParameter("max_workers must be a positive integer.")

    return {
        "theory_name": theory_name or cfg.get("theory_name"),
        "acronym": acronym or cfg.get("acronym"),
        "l1": _parse_l1(l1_papers or l1_cfg_str, resolved_l1_file),
        "citation_data_path": citation_data
        or (Path(cfg["citation_data"]) if cfg.get("citation_data") else None),
        "papers_data_path": papers_data
        or (Path(cfg["papers_data"]) if cfg.get("papers_data") else None),
        "labels_data_path": labels_data
        or (Path(cfg["labels_data"]) if cfg.get("labels_data") else None),
        "output_features": output_features
        or (Path(cfg["output_features"]) if cfg.get("output_features") else None),
        "output_predictions": output_predictions
        or (Path(cfg["output_predictions"]) if cfg.get("output_predictions") else None),
        "online": bool(online or cfg.get("online", False)),
        "sources": sources or cfg.get("sources"),
        "depth": (depth or cfg.get("depth") or "l2l3").lower(),
        "key_constructs": key_constructs or cfg.get("key_constructs"),
        "cache_dir": cache_dir or (Path(cfg["cache_dir"]) if cfg.get("cache_dir") else None),
        "checkpoint_dir": checkpoint_dir
        or (Path(cfg["checkpoint_dir"]) if cfg.get("checkpoint_dir") else None),
        "checkpoint_staleness_seconds": resolved_checkpoint_staleness,
        "refresh_cache": bool(refresh_cache or cfg.get("refresh_cache", False)),
        "reset_checkpoints": bool(reset_checkpoints or cfg.get("reset_checkpoints", False)),
        "max_l2": int(cfg.get("max_l2", max_l2)),
        "max_l3": int(cfg["max_l3"]) if cfg.get("max_l3") is not None else max_l3,
        "max_workers": resolved_max_workers,
        "save_ingested_citation_data": save_ingested_citation_data
        or (
            Path(cfg["save_ingested_citation_data"])
            if cfg.get("save_ingested_citation_data")
            else None
        ),
        "save_ingested_papers_data": save_ingested_papers_data
        or (
            Path(cfg["save_ingested_papers_data"]) if cfg.get("save_ingested_papers_data") else None
        ),
        "only_ingest": bool(only_ingest or cfg.get("only_ingest", False)),
        "exhaustive": bool(exhaustive if exhaustive is not None else cfg.get("exhaustive", True)),
        "verbose": bool(verbose or cfg.get("verbose", False)),
        "quiet": bool(quiet or cfg.get("quiet", False)),
    }


def _persist_json(path: Path, payload: Dict[str, Any], label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    typer.echo(f"Saved {label} to {path}")


def _load_pipeline_inputs(params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if params["online"]:
        raw_sources = params.get("sources") or "openalex,semantic_scholar,crossref,core"
        if isinstance(raw_sources, list):
            selected_sources = [s.strip() for s in raw_sources if s and s.strip()]
        else:
            selected_sources = [
                item.strip() for item in str(raw_sources).split(",") if item.strip()
            ]

        if "core" in [source.lower() for source in selected_sources] and not os.getenv(
            "CORE_API_KEY"
        ):
            typer.echo(
                "Warning: CORE selected without CORE_API_KEY. Continuing in unauthenticated mode "
                "with stricter rate limits and reduced full-text access.",
            )

        constructs = _parse_key_constructs(params.get("key_constructs"))

        if params["depth"] not in {"l2", "l2l3"}:
            raise typer.BadParameter("depth must be either 'l2' or 'l2l3' in online mode.")

        ingestion = ingest_from_internet(
            theory_name=params["theory_name"],
            l1_papers=params["l1"],
            key_constructs=constructs,
            sources=selected_sources,
            depth=params["depth"],
            cache_dir=params["cache_dir"],
            checkpoint_dir=params.get("checkpoint_dir"),
            checkpoint_staleness_seconds=params.get("checkpoint_staleness_seconds"),
            refresh=params["refresh_cache"],
            reset_checkpoints=params.get("reset_checkpoints", False),
            max_l2=params["max_l2"],
            max_l3=params["max_l3"],
            max_workers=params.get("max_workers"),
            exhaustive=params.get("exhaustive", True),
            verbose=params.get("verbose", False),
            quiet=params.get("quiet", False),
        )
        typer.echo(
            "Online ingestion complete: "
            f"papers={ingestion.metadata.get('paper_count', 0)} "
            f"edges={ingestion.metadata.get('edge_count', 0)} "
            f"sources={','.join(selected_sources)}"
        )

        if params["save_ingested_citation_data"]:
            _persist_json(
                params["save_ingested_citation_data"],
                ingestion.citation_data,
                "ingested citation_data",
            )
        if params["save_ingested_papers_data"]:
            _persist_json(
                params["save_ingested_papers_data"],
                ingestion.papers_data,
                "ingested papers_data",
            )
        return ingestion.citation_data, ingestion.papers_data

    if not params["citation_data_path"]:
        raise typer.BadParameter("citation_data path is required (CLI arg or config).")
    if not params["papers_data_path"]:
        raise typer.BadParameter("papers_data path is required (CLI arg or config).")

    return (
        _load_json_dict(params["citation_data_path"], "citation_data"),
        _load_json_dict(params["papers_data_path"], "papers_data"),
    )


@app.command()
def run(
    config: Optional[Path] = CONFIG_OPTION,
    theory_name: Optional[str] = THEORY_NAME_OPTION,
    acronym: Optional[str] = ACRONYM_OPTION,
    l1_papers: Optional[str] = L1_PAPERS_OPTION,
    l1_file: Optional[Path] = L1_FILE_OPTION,
    citation_data: Optional[Path] = CITATION_DATA_OPTION,
    papers_data: Optional[Path] = PAPERS_DATA_OPTION,
    labels_data: Optional[Path] = LABELS_DATA_OPTION,
    online: bool = ONLINE_OPTION,
    sources: Optional[str] = SOURCES_OPTION,
    depth: str = DEPTH_OPTION,
    key_constructs: Optional[str] = KEY_CONSTRUCTS_OPTION,
    cache_dir: Optional[Path] = CACHE_DIR_OPTION,
    checkpoint_dir: Optional[Path] = CHECKPOINT_DIR_OPTION,
    checkpoint_staleness_seconds: Optional[int] = CHECKPOINT_STALENESS_OPTION,
    refresh_cache: bool = REFRESH_CACHE_OPTION,
    reset_checkpoints: bool = RESET_CHECKPOINTS_OPTION,
    max_l2: int = MAX_L2_OPTION,
    max_l3: Optional[int] = MAX_L3_OPTION,
    max_workers: Optional[int] = MAX_WORKERS_OPTION,
    save_ingested_citation_data: Optional[Path] = SAVE_INGESTED_CITATION_OPTION,
    save_ingested_papers_data: Optional[Path] = SAVE_INGESTED_PAPERS_OPTION,
    output_features: Optional[Path] = OUTPUT_FEATURES_OPTION,
    output_predictions: Optional[Path] = OUTPUT_PREDICTIONS_OPTION,
    only_ingest: bool = ONLY_INGEST_OPTION,
    exhaustive: bool = EXHAUSTIVE_OPTION,
    verbose: bool = VERBOSE_OPTION,
    quiet: bool = QUIET_OPTION,
) -> None:
    """Run ADIT using CLI values and/or a config file."""
    cfg = _load_config(config)

    params = _resolve_cli_inputs(
        cfg,
        theory_name,
        acronym,
        l1_papers,
        l1_file,
        citation_data,
        papers_data,
        labels_data,
        sources,
        depth,
        key_constructs,
        cache_dir,
        checkpoint_dir,
        checkpoint_staleness_seconds,
        refresh_cache,
        reset_checkpoints,
        max_l2,
        max_l3,
        max_workers,
        save_ingested_citation_data,
        save_ingested_papers_data,
        output_features,
        output_predictions,
        online,
        only_ingest,
        exhaustive,
        verbose,
        quiet,
    )

    if not params["theory_name"]:
        raise typer.BadParameter("theory_name is required (CLI arg or config).")

    citation_dict, papers_dict = _load_pipeline_inputs(params)
    constructs = _parse_key_constructs(params.get("key_constructs"))

    if params["only_ingest"]:
        typer.echo(
            "Ingestion-only mode enabled: skipped feature extraction and ML training/prediction."
        )
        return

    adit = ADIT(
        theory_name=params["theory_name"],
        l1_papers=params["l1"],
        acronym=params["acronym"],
        key_constructs=constructs,
    )
    adit.build_ecosystem(citation_dict)
    features = adit.extract_features(papers_dict)

    typer.echo(f"Extracted {len(features)} L2 feature rows.")
    typer.echo(f"Using theory='{params['theory_name']}' acronym='{adit.acronym}'")

    if params["output_features"]:
        params["output_features"].parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(params["output_features"], index=False)
        typer.echo(f"Saved features to {params['output_features']}")

    if not params["labels_data_path"]:
        typer.echo("No labels_data provided; skipped training/prediction.")
        return

    labels_raw = json.loads(params["labels_data_path"].read_text(encoding="utf-8"))
    labels = _resolve_labels(labels_raw, features)
    adit.train_classifier(features, labels)
    predictions = adit.predict_subscription(features)

    predictions_df = pd.DataFrame({"paper_id": features["paper_id"], "prediction": predictions})
    typer.echo(f"Generated {len(predictions_df)} predictions.")

    if params["output_predictions"]:
        params["output_predictions"].parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(params["output_predictions"], index=False)
        typer.echo(f"Saved predictions to {params['output_predictions']}")
    else:
        typer.echo(predictions_df.to_string(index=False))


if __name__ == "__main__":
    app()
