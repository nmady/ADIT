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

If `labels_data` is omitted, the CLI extracts features and skips training/prediction.

## Adaptation Notes

- Augments traditional ML with transformer-based embeddings for text analysis.
- Uses NetworkX for graph analysis.
- Extensible to other theories and datasets.

## Data Sources

Real citation data can be obtained from:
- Google Scholar
- Web of Science
- Scopus
- IEEE Xplore

For this demo, synthetic data is used.