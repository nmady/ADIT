# ADIT Adaptation: Automated Detection of Implicit Theories

This project implements an adaptation of the ADIT (Automated Detection of Implicit Theories) system described in Larsen et al. (2014) "Theory Identity: A Machine-Learning Approach" and related work.

## Overview

ADIT uses machine learning to identify papers that subscribe to a specific theory by analyzing citation networks and article features. This adaptation modernizes the approach with contemporary NLP techniques and graph analysis.

## Key Components

- **Theory Ecosystem Construction**: Builds multi-level citation networks (L1: originating papers, L2: citing papers, L3: cited papers).
- **Feature Extraction**: Uses text embeddings, citation metrics, and structural features.
- **Machine Learning Classification**: Predicts which papers contribute to the theory using modern ML models.

## Installation

1. Set up Python environment (already configured).
2. Install dependencies: `pip install -r requirements.txt` (or via tool).

## Usage

### Typer CLI

The project includes a Typer-based interface in `cli.py`.

Run with direct command-line arguments:

```bash
python cli.py \
	--theory-name "Technology Acceptance Model" \
	--acronym TAM \
	--l1-papers "TAM1,TAM2" \
	--citation-data citation_data.json \
	--papers-data papers_data.json \
	--labels-data labels_data.json \
	--output-features outputs/features.csv \
	--output-predictions outputs/predictions.csv
```

Run with a config file instead:

```bash
python cli.py --config config.yml
```

Run with internet ingestion instead of local `citation_data` / `papers_data` files:

```bash
python cli.py \
	--online \
	--theory-name "Technology Acceptance Model" \
	--acronym TAM \
	--l1-papers "10.2307/249008,10.1287/mnsc.35.8.982" \
	--sources "openalex,semantic_scholar,crossref" \
	--depth l2l3 \
	--key-constructs "usefulness,ease of use,behavioral intention" \
	--cache-dir .cache/adit_ingestion \
	--only-ingest \
	--save-ingested-citation-data outputs/citation_data.json \
	--save-ingested-papers-data outputs/papers_data.json \
	--output-features outputs/features.csv
```

Example `config.yml`:

```yaml
theory_name: Technology Acceptance Model
acronym: TAM
l1_papers: [TAM1, TAM2]
citation_data: citation_data.json
papers_data: papers_data.json
labels_data: labels_data.json
output_features: outputs/features.csv
output_predictions: outputs/predictions.csv
```

Example online-ingestion config:

See `examples/online-ingestion.config.yml` for a ready-to-edit copy.

```yaml
theory_name: Technology Acceptance Model
acronym: TAM
l1_papers:
	- 10.2307/249008
	- 10.1287/mnsc.35.8.982
online: true
sources:
	- openalex
	- semantic_scholar
	- crossref
depth: l2l3
key_constructs:
	- usefulness
	- ease of use
	- behavioral intention
cache_dir: .cache/adit_ingestion
save_ingested_citation_data: outputs/citation_data.json
save_ingested_papers_data: outputs/papers_data.json
output_features: outputs/features.csv
```

If `labels_data` is omitted, the CLI extracts features and skips training/prediction.

### Online Ingestion Notes

- Online mode is opt-in via `--online`; without it, the CLI still expects local `citation_data` and `papers_data` JSON files.
- Supported v1 providers are `openalex`, `semantic_scholar`, and `crossref`.
- Results from multiple providers are normalized and deduplicated before building the citation ecosystem.
- `--depth l2` retrieves direct citers of the L1 papers; `--depth l2l3` also retrieves references from L2 papers to populate L3.
- `--cache-dir` stores cached ingestion results so repeated runs can reuse prior retrieval work.
- `--refresh-cache` forces a fresh internet retrieval instead of reusing cached results.
- `--only-ingest` runs ingestion and exits before feature extraction/training.
- `--save-ingested-citation-data` and `--save-ingested-papers-data` let you persist normalized outputs for offline replay.
- Tests do not rely on live provider calls; the internet ingestion path is covered with mocked fixtures.

### `citation_data.json` Format

`citation_data.json` is a JSON object (dictionary) where:

- Each key is a citing paper ID.
- Each value is a JSON array of cited paper IDs.
- IDs are treated as opaque strings (for example DOI strings, local IDs like `TAM1`, or provider IDs like `openalex:W1708393801`).

Shape:

```json
{
	"<citing_paper_id>": ["<cited_paper_id>", "<cited_paper_id>", "..."],
	"<another_citing_paper_id>": []
}
```

Minimal offline example:

```json
{
	"PaperA": ["TAM1", "PaperX"],
	"PaperB": ["TAM2"],
	"PaperC": []
}
```

Provider-ingested example (what online mode writes):

```json
{
	"openalex:W1708393801": [
		"openalex:W134333820",
		"openalex:W1483247443"
	],
	"openalex:W2337589995": []
}
```

Behavior and expectations:

- Edge direction is `citing -> cited`.
- Empty lists are valid and mean "paper known, no references captured."
- Cited IDs do not need to exist as top-level keys; ADIT creates nodes for them when building the graph.
- The file must be a JSON object, not a list.
- When running offline mode, this file is required via `--citation-data` (or `citation_data` in config).

### `papers_data.json` Format

`papers_data.json` is a JSON object (dictionary) keyed by paper ID, where each value is a metadata object.

Shape:

```json
{
	"<paper_id>": {
		"title": "<string>",
		"abstract": "<string>",
		"keywords": "<string>",
		"citations": 0,
		"year": null
	}
}
```

If year is unknown, prefer `null`:

```json
{
	"<paper_id>": {
		"title": "<string>",
		"abstract": "<string>",
		"keywords": "<string>",
		"citations": 0,
		"year": null
	}
}
```

Minimal offline example:

```json
{
	"TAM1": {
		"title": "Technology Acceptance Model",
		"abstract": "Original TAM paper.",
		"keywords": "TAM, acceptance",
		"citations": 1000,
		"year": 1989
	},
	"PaperA": {
		"title": "Extension of Technology Acceptance Model",
		"abstract": "This paper extends TAM in a new context.",
		"keywords": "TAM, technology acceptance",
		"citations": 50,
		"year": 2015
	}
}
```

Behavior and expectations:

- Keys should use the same ID namespace as `citation_data.json` for correct graph-to-metadata joins.
- Missing fields are tolerated; ADIT applies defaults during feature extraction:
	- `title`, `abstract`, `keywords` default to empty string.
	- `citations` defaults to `0`.
	- `year` is treated as unknown when missing/invalid (`null` recommended in JSON).
- Publication-year normalization is computed only from known years present in the dataset.
- For papers with unknown year, `pub_year` is emitted as missing (`NaN`) in extracted features.
- During classifier train/predict, missing numeric feature values (including `pub_year`) are imputed with the training-set median.
- `keywords` is currently treated as a plain string and searched with substring checks.
- Extra fields are allowed and ignored by the current feature extractor.
- Online-ingested entries may include `source_ids`, a provider-to-provider-id map used for provenance and debugging.
- When running offline mode, this file is required via `--papers-data` (or `papers_data` in config).
- In online mode, this file is generated when using `--save-ingested-papers-data`.

### Ingestion Metadata

Online ingestion returns a `metadata` object alongside `citation_data` and `papers_data`.
This includes `fetch_stats`:

```json
{
	"fetch_stats": {
		"total_requests": 42,
		"total_failures": 3,
		"per_provider_failures": {
			"openalex": 1,
			"semantic_scholar": 2,
			"crossref": 0
		}
	}
}
```

- `total_requests` counts attempted HTTP fetches during the run.
- `total_failures` counts failed HTTP fetches.
- `per_provider_failures` breaks failures down by provider.

## Config reference

A compact reference for keys accepted in config files (and equivalent CLI flags). Types shown as (type, default).

- **theory_name**: (string, required) — Human-readable theory name used for keyword matching and acronym derivation. CLI: `--theory-name`.
- **acronym**: (string, optional) — Explicit acronym to search for (e.g., `TAM`). If omitted, derived from `theory_name`.
- **l1_papers**: (list|string, required) — L1 seed papers. In YAML provide a list, or use comma-separated string on CLI. Example: `[TAM1, TAM2]` or `--l1-papers "TAM1,TAM2"`.
- **l1_file**: (path, optional) — Path to newline-separated file of L1 paper IDs (alternative to `l1_papers`).
- **citation_data** / **citation_data**: (path, optional) — Offline `citation_data.json` path. Required when `online: false`.
- **papers_data**: (path, optional) — Offline `papers_data.json` path. Required when `online: false`.
- **labels_data**: (path, optional) — Labels file (JSON list aligned to features or dict by paper_id) used for training.
- **online**: (bool, default: false) — When true, fetch citation and paper metadata from online providers instead of local JSON.
- **sources**: (list|string, default: [openalex, semantic_scholar, crossref]) — Providers to query in online mode. Accepts YAML lists or comma-separated strings on CLI.
- **depth**: (string, default: l2l3) — Ingestion expansion depth: `l2` or `l2l3`.
- **key_constructs**: (list|string, optional) — Additional keywords to bias provider search relevance. Accepts YAML lists or comma-separated strings.
- **cache_dir**: (path, default: .cache/adit_ingestion) — Directory to cache provider responses and merged ingestion payloads.
- **refresh_cache**: (bool, default: false) — If true, ignore cached ingestion and force fresh retrieval.
- **max_l2**: (int, default: 200) — Per-provider cap on L2 candidates requested from each source (effective per-provider limit may be lower due to provider caps).
- **max_l3**: (int or null, default: null/unlimited) — Optional per-provider cap on L3 reference edges retrieved when expanding L2 → L3. Omit or set to null for exhaustive L3 retrieval.
- **save_ingested_citation_data**: (path, optional) — Persist normalized `citation_data.json` produced by online ingestion.
- **save_ingested_papers_data**: (path, optional) — Persist normalized `papers_data.json` produced by online ingestion.
- **output_features**: (path, optional) — CSV path to write extracted features.
- **output_predictions**: (path, optional) — CSV path to write model predictions.

Notes:
- Many config keys accept either YAML lists (recommended for examples) or comma-separated CLI strings — the CLI normalizes both formats.
- `max_l2`/`max_l3` are applied per-provider; the final L2/L3 coverage is the union of provider outputs after deduplication.
- Use `year: null` in `papers_data.json` to indicate unknown publication year; ADIT emits `pub_year` as `NaN` and imputes numeric features at training time.

## Adaptation Notes

- Augments traditional ML with transformer-based embeddings for text analysis.
- Uses NetworkX for graph analysis.
- Extensible to other theories and datasets.

## Data Sources

Current internet-ingestion support includes:
- OpenAlex
- Semantic Scholar
- Crossref

Planned future adapters may include:
- Dimensions
- Web of Science
- Scopus

For this demo, synthetic data is used.