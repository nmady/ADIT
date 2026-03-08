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

1. Define the theory's originating papers.
2. Collect citation data (manually or via APIs).
3. Run the ecosystem builder.
4. Train the classifier on labeled data.
5. Predict subscribing papers.

## Adaptation Notes

- Replaces traditional ML with transformer-based embeddings for text analysis.
- Uses NetworkX for graph analysis.
- Extensible to other theories and datasets.

## Data Sources

Real citation data can be obtained from:
- Google Scholar
- Web of Science
- Scopus
- IEEE Xplore

For this demo, synthetic data is used.