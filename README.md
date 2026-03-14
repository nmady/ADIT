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
- `--save-ingested-citation-data` and `--save-ingested-papers-data` let you persist normalized outputs for offline replay.
- Tests do not rely on live provider calls; the internet ingestion path is covered with mocked fixtures.

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