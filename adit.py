import logging

import networkx as nx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Public helper and stopword list for acronym derivation. Exported so tests
# and lightweight test doubles can reuse the exact same logic without
# duplicating definitions.
ACRONYM_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "to",
    "from",
    "by",
    "with",
    "for",
    "as",
    "about",
    "and",
    "or",
    "but",
    "is",
    "are",
}


def derive_acronym(theory_name: str) -> str:
    """Derive a deterministic acronym from a theory name.

    Strategy: ignore common minor words (stopwords) when deriving the
    acronym. If removing stopwords leaves no words, fall back to using the
    original words so the output is never empty.
    """
    words = theory_name.split()
    content_words = [w for w in words if w.lower() not in ACRONYM_STOPWORDS]
    selected_words = content_words or words
    return "".join(w[0] for w in selected_words).lower()


class ADIT:
    pass

    def __init__(self, theory_name, l1_papers, transformer=None, acronym=None):
        """
        Initialize ADIT for a specific theory.

        :param theory_name: Name of the theory (e.g., 'TAM')
        :param l1_papers: List of originating paper IDs or titles
        :param acronym: Optional explicit acronym (e.g., 'TAM'). If omitted, derived from theory_name.
        """
        self.theory_name = theory_name
        self.acronym = acronym.lower() if acronym else derive_acronym(theory_name)
        self.l1_papers = l1_papers
        self.ecosystem = nx.DiGraph()
        # Dependency injection: accept an optional transformer for easier testing
        self.transformer = transformer or SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # For text embeddings
        self.classifier = RandomForestClassifier()

    def build_ecosystem(self, citation_data):
        """
        Build a citation ecosystem graph with three levels:
        - L1: Foundational theory papers
        - L2: Papers that cite at least one L1 paper
        - L3: Papers cited by L2 papers that are not already in L1/L2

        :param citation_data: Dict of {citing_paper: [cited_papers]}
        """
        # Add foundational theory papers.
        for paper in self.l1_papers:
            self.ecosystem.add_node(paper, level="L1")

        # First pass: classify citing papers as L2 when they cite any L1 paper.
        for citing, cited_list in citation_data.items():
            cites_l1 = any(cited in self.l1_papers for cited in cited_list)
            if cites_l1:
                self.ecosystem.add_node(citing, level="L2")

            for cited in cited_list:
                if cited not in self.ecosystem:
                    self.ecosystem.add_node(cited, level="Unknown")
                self.ecosystem.add_edge(citing, cited)

        # Second pass: promote unknown nodes cited by L2 papers to L3.
        for node, data in list(self.ecosystem.nodes(data=True)):
            if data.get("level") != "Unknown":
                continue

            predecessors = list(self.ecosystem.predecessors(node))
            cited_by_l2 = any(self.ecosystem.nodes[p].get("level") == "L2" for p in predecessors)
            if cited_by_l2:
                self.ecosystem.nodes[node]["level"] = "L3"

    def compute_eigenfactor(self):
        """
        Compute article-level Eigenfactor scores using modified PageRank.
        Based on Larsen et al. (2014): Modified for citation networks where
        citations flow backward in time. Uses PageRank with citation direction
        (citing -> cited) and handles temporal acyclic nature.

        :return: Dict of Eigenfactor scores for each node
        """
        try:
            # Use PageRank with citation direction (citing -> cited)
            # In citation networks, if A cites B, B gets importance from A
            eigenfactor_scores = nx.pagerank(self.ecosystem, alpha=0.85, max_iter=100)
        except Exception:
            # If PageRank fails (e.g., convergence error or empty graph), fall back to uniform scores
            logging.warning(
                "PageRank computation failed, using uniform scores as fallback", exc_info=True
            )
            eigenfactor_scores = dict.fromkeys(self.ecosystem.nodes(), 1.0)
        return eigenfactor_scores

    def extract_features(self, papers_data):
        """
        Extract features for L2 papers combining hand-designed and modern NLP features.
        Hand-designed features from Larsen et al. (2014, 2019):
        - Eigenfactor_Eco: Article-level Eigenfactor (prestige-based importance)
        - Betweenness centrality: Bridge importance in citation network
        - Theory-Attribution Ratio (TAR): Citations to other L2 papers
        - Citation count: Number of citations
        - Publication Year (dynamically normalized to [0, 1] based on data range)
        - Word count in abstract
        - Theory name/acronym presence in title/keywords/abstract (binary flags)
        - Key construct presence in title/abstract (binary flags)

        Modern NLP features:
        - Semantic similarity: Cosine similarity via sentence transformers

        :param papers_data: Dict with paper info: {paper_id: {'title': str, 'abstract': str,
                                                          'keywords': str, 'citations': int,
                                                          'year': int}}
        :return: DataFrame with features
        """
        features = []
        concat_L1_abs = " ".join(
            [papers_data.get(p, {}).get("abstract", "") for p in self.l1_papers]
        )
        theory_emb = self.transformer.encode(concat_L1_abs)

        # Compute network centrality measures
        try:
            betweenness_scores = nx.betweenness_centrality(self.ecosystem)
        except Exception:
            # If betweenness computation fails (resource limit or other issue),
            # assign zero betweenness so downstream feature pipeline continues.
            logging.warning(
                "Betweenness centrality computation failed, using zero scores as fallback",
                exc_info=True,
            )
            betweenness_scores = dict.fromkeys(self.ecosystem.nodes(), 0.0)

        eigenfactor_scores = self.compute_eigenfactor()

        # Determine min and max year for dynamic normalization
        years = [papers_data.get(p, {}).get("year", 2010) for p in papers_data]
        min_year = min(years) if years else 2010
        max_year = max(years) if years else 2010
        year_range = max_year - min_year if max_year > min_year else 1

        for node, data in self.ecosystem.nodes(data=True):
            if data.get("level") == "L2":
                paper_info = papers_data.get(node, {})
                title = paper_info.get("title", "").lower()
                abstract = paper_info.get("abstract", "").lower()
                keywords = paper_info.get("keywords", "").lower()
                citation_count = paper_info.get("citations", 0)
                year = paper_info.get("year", 2010)

                # 1. Network feature: Eigenfactor_Eco (article-level Eigenfactor)
                eigenfactor = eigenfactor_scores.get(node, 0.0)

                # 2. Network feature: Betweenness centrality (bridge importance)
                betweenness = betweenness_scores.get(node, 0.0)

                # 2. Theory-Attribution Ratio (TAR): fraction of references to L2 papers
                l2_papers_cited = 0
                total_refs = len(list(self.ecosystem.successors(node)))
                for ref in self.ecosystem.successors(node):
                    ref_data = self.ecosystem.nodes[ref]
                    if ref_data.get("level") == "L2":
                        l2_papers_cited += eigenfactor_scores.get(ref, 0.0)
                tar = (l2_papers_cited / max(total_refs, 1)) if total_refs > 0 else 0.0

                # 3. Citation count

                # 4. Publication year (normalized dynamically to [0, 1])
                pub_year_norm = (year - min_year) / year_range

                # 5. Abstract word count
                word_count = len(abstract.split())

                # 6-8. Theory name in title/keywords/abstract (binary)
                theory_name = self.theory_name.lower()
                theory_in_title = int(theory_name in title)
                theory_in_keywords = int(theory_name in keywords)
                theory_in_abstract = int(theory_name in abstract)

                # 9-11. Theory acronym in title/keywords/abstract (binary)
                acronym = self.acronym
                acronym_in_title = int(acronym in title)
                acronym_in_keywords = int(acronym in keywords)
                acronym_in_abstract = int(acronym in abstract)

                # 12. Key constructs - individual binary flags (example: TAM uses "usefulness", "ease of use")
                key_constructs = [
                    "usefulness",
                    "ease of use",
                    "acceptance",
                    "intention",
                    "attitude",
                ]
                construct_features = {}
                for construct in key_constructs:
                    feature_name = f"has_{construct.replace(' ', '_')}"
                    construct_features[feature_name] = int(
                        construct in title or construct in abstract
                    )

                # 13. Semantic similarity (modern NLP via embeddings)
                paper_emb = self.transformer.encode(abstract)
                norm_theory = np.linalg.norm(theory_emb)
                norm_paper = np.linalg.norm(paper_emb)
                if norm_theory > 0 and norm_paper > 0:
                    semantic_similarity = np.dot(theory_emb, paper_emb) / (norm_theory * norm_paper)
                else:
                    semantic_similarity = 0.0

                # 14-15. Network structure: in_degree and out_degree
                in_degree = self.ecosystem.in_degree(node)
                out_degree = self.ecosystem.out_degree(node)

                features.append(
                    {
                        "paper_id": node,
                        "eigenfactor": eigenfactor,
                        "betweenness": betweenness,
                        "theory_attribution_ratio": tar,
                        "citation_count": citation_count,
                        "pub_year": pub_year_norm,
                        "abstract_word_count": word_count,
                        "theory_in_title": theory_in_title,
                        "theory_in_keywords": theory_in_keywords,
                        "theory_in_abstract": theory_in_abstract,
                        "acronym_in_title": acronym_in_title,
                        "acronym_in_keywords": acronym_in_keywords,
                        "acronym_in_abstract": acronym_in_abstract,
                        **construct_features,  # Unpack individual construct flags
                        "semantic_similarity": semantic_similarity,
                        "in_degree": in_degree,
                        "out_degree": out_degree,
                    }
                )

        return pd.DataFrame(features)

    def train_classifier(self, features_df, labels):
        """
        Train the ML classifier using all hand-designed and modern NLP features.

        :param features_df: DataFrame with features
        :param labels: List of labels (1: subscribes, 0: not)
        """
        feature_cols = [col for col in features_df.columns if col != "paper_id"]
        X = features_df[feature_cols]
        y = labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("\nFeature Importance (top 5):")
        feature_importance = (
            pd.DataFrame(
                {"feature": feature_cols, "importance": self.classifier.feature_importances_}
            )
            .sort_values("importance", ascending=False)
            .head(5)
        )
        print(feature_importance)

    def predict_subscription(self, features_df):
        """
        Predict which papers subscribe to the theory using trained classifier.

        :param features_df: DataFrame with features
        :return: Predictions (1: subscribes, 0: does not)
        """
        feature_cols = [col for col in features_df.columns if col != "paper_id"]
        X = features_df[feature_cols]
        return self.classifier.predict(X)


# Example usage with mock data
if __name__ == "__main__":
    # Mock L1 papers (TAM originating papers)
    l1 = ["TAM1", "TAM2"]

    adit = ADIT("TAM", l1)

    # Mock citation data: {citing_paper: [cited_papers]}
    citation_data = {
        "PaperA": ["TAM1"],
        "PaperB": ["TAM2"],
        "PaperC": ["TAM1", "Other"],
        "PaperD": ["Unrelated"],
        "PaperE": ["TAM1", "TAM2"],
    }

    adit.build_ecosystem(citation_data)

    # Mock paper data with all relevant fields
    papers_data = {
        "PaperA": {
            "title": "Extension of Technology Acceptance Model",
            "abstract": "This paper extends TAM with new constructs for mobile adoption.",
            "keywords": "TAM, technology acceptance, mobile",
            "citations": 50,
            "year": 2015,
        },
        "PaperB": {
            "title": "Empirical test of TAM",
            "abstract": "We test TAM in a new context with emphasis on ease of use and usefulness.",
            "keywords": "TAM, acceptance, empirical study",
            "citations": 30,
            "year": 2012,
        },
        "PaperC": {
            "title": "Unrelated topic in information systems",
            "abstract": "This paper studies something unrelated to technology acceptance.",
            "keywords": "information systems, management",
            "citations": 10,
            "year": 2010,
        },
        "PaperD": {
            "title": "AI and machine learning applications",
            "abstract": "Methods for applying machine learning to various domains.",
            "keywords": "machine learning, AI",
            "citations": 5,
            "year": 2018,
        },
        "PaperE": {
            "title": "TAM in healthcare: Ease of use and behavioral intention",
            "abstract": "Applies TAM to understand healthcare technology adoption with focus on usefulness.",
            "keywords": "TAM, acceptance, healthcare, behavioral intention",
            "citations": 25,
            "year": 2014,
        },
        "TAM1": {
            "title": "Technology Acceptance Model",
            "abstract": "Original TAM paper proposing model of technology acceptance.",
            "keywords": "TAM, acceptance, technology",
            "citations": 1000,
            "year": 1989,
        },
        "TAM2": {
            "title": "Technology Acceptance Model extension",
            "abstract": "Extension of TAM with more constructs.",
            "keywords": "TAM, acceptance",
            "citations": 500,
            "year": 1992,
        },
    }

    # Extract features for all L2 papers
    features = adit.extract_features(papers_data)
    print("=== Extracted Features ===")
    print(features)
    print(f"\nFeature columns: {list(features.columns)}")

    # Mock labels keyed by paper_id (1: subscribes to TAM, 0: does not)
    label_map = {
        "PaperA": 1,  # extends TAM
        "PaperB": 1,  # tests TAM
        "PaperC": 0,  # unrelated
        "PaperD": 0,  # unrelated (may not be in extracted L2 set)
        "PaperE": 1,  # applies TAM
    }
    labels = [label_map.get(paper_id, 0) for paper_id in features["paper_id"]]

    print("\n=== Training Classifier ===")
    adit.train_classifier(features, labels)

    print("\n=== Predictions ===")
    predictions = adit.predict_subscription(features)
    for i, paper_id in enumerate(features["paper_id"]):
        print(f"{paper_id}: {'Subscribes' if predictions[i] == 1 else 'Does not subscribe'}")
