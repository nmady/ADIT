import networkx as nx
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class ADIT:
    def __init__(self, theory_name, l1_papers):
        """
        Initialize ADIT for a specific theory.

        :param theory_name: Name of the theory (e.g., 'TAM')
        :param l1_papers: List of originating paper IDs or titles
        """
        self.theory_name = theory_name
        self.l1_papers = l1_papers
        self.ecosystem = nx.DiGraph()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # For text embeddings
        self.classifier = RandomForestClassifier()

    def build_ecosystem(self, citation_data):
        """
        Build the theory ecosystem from citation data.

        :param citation_data: Dict or DataFrame with paper citations
        """
        # Add L1 papers
        for paper in self.l1_papers:
            self.ecosystem.add_node(paper, level='L1')

        # Assuming citation_data is a dict: {citing_paper: [cited_papers]}
        # Build L2: papers citing L1
        l2_papers = set()
        for citing, cited_list in citation_data.items():
            if any(cited in self.l1_papers for cited in cited_list):
                l2_papers.add(citing)
                self.ecosystem.add_node(citing, level='L2')
                for cited in cited_list:
                    if cited in self.l1_papers:
                        self.ecosystem.add_edge(citing, cited)

        # L3: papers cited by L2 (simplified)
        for l2_paper in l2_papers:
            if l2_paper in citation_data:
                for cited in citation_data[l2_paper]:
                    if cited not in self.l1_papers and cited not in l2_papers:
                        self.ecosystem.add_node(cited, level='L3')
                        self.ecosystem.add_edge(l2_paper, cited)

    def extract_features(self, papers_data):
        """
        Extract features for L2 papers combining hand-designed and modern NLP features.
        
        Hand-designed features from Larsen et al. (2014, 2019):
        - Eigenfactor_Eco: Network importance in theory ecosystem
        - Theory-Attribution Ratio (TAR): Citations to other L2 papers
        - Impact: Citation count
        - Publication Year
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
        theory_desc = " ".join([papers_data.get(p, {}).get('abstract', '') for p in self.l1_papers])
        theory_emb = self.model.encode(theory_desc)
        
        # Compute Eigenfactor approximation using betweenness centrality
        try:
            eigenfactor_scores = nx.betweenness_centrality(self.ecosystem)
        except:
            eigenfactor_scores = {node: 1.0 for node in self.ecosystem.nodes()}

        for node, data in self.ecosystem.nodes(data=True):
            if data.get('level') == 'L2':
                paper_info = papers_data.get(node, {})
                title = paper_info.get('title', '').lower()
                abstract = paper_info.get('abstract', '').lower()
                keywords = paper_info.get('keywords', '').lower()
                citations = paper_info.get('citations', 0)
                year = paper_info.get('year', 2010)

                # 1. Network feature: Eigenfactor_Eco (approximated via betweenness centrality)
                eigenfactor = eigenfactor_scores.get(node, 0.0)

                # 2. Theory-Attribution Ratio (TAR): fraction of references to L2 papers
                l2_papers_cited = 0
                total_refs = len(list(self.ecosystem.successors(node)))
                for ref in self.ecosystem.successors(node):
                    ref_data = self.ecosystem.nodes[ref]
                    if ref_data.get('level') == 'L2':
                        l2_papers_cited += eigenfactor_scores.get(ref, 0.0)
                tar = (l2_papers_cited / max(total_refs, 1)) if total_refs > 0 else 0.0

                # 3. Impact (citation count)
                impact = citations

                # 4. Publication year (normalized)
                pub_year_norm = (year - 2000) / 30.0  # Normalize to roughly [0, 1]

                # 5. Abstract word count
                word_count = len(abstract.split())

                # 6-8. Theory name in title/keywords/abstract (binary)
                theory_name = self.theory_name.lower()
                theory_in_title = int(theory_name in title)
                theory_in_keywords = int(theory_name in keywords)
                theory_in_abstract = int(theory_name in abstract)

                # 9-11. Theory acronym in title/keywords/abstract (binary)
                acronym = ''.join([w[0] for w in self.theory_name.split()]).lower()
                acronym_in_title = int(acronym in title)
                acronym_in_keywords = int(acronym in keywords)
                acronym_in_abstract = int(acronym in abstract)

                # 12. Key constructs count (example: TAM uses "usefulness", "ease of use")
                key_constructs = ['usefulness', 'ease of use', 'acceptance', 'intention', 'attitude']
                constructs_count = sum(1 for construct in key_constructs 
                                     if construct in title or construct in abstract)

                # 13. Semantic similarity (modern NLP via embeddings)
                paper_emb = self.model.encode(abstract)
                norm_theory = np.linalg.norm(theory_emb)
                norm_paper = np.linalg.norm(paper_emb)
                if norm_theory > 0 and norm_paper > 0:
                    semantic_similarity = np.dot(theory_emb, paper_emb) / (norm_theory * norm_paper)
                else:
                    semantic_similarity = 0.0

                # 14-15. Network structure: in_degree and out_degree
                in_degree = self.ecosystem.in_degree(node)
                out_degree = self.ecosystem.out_degree(node)

                features.append({
                    'paper_id': node,
                    'eigenfactor': eigenfactor,
                    'theory_attribution_ratio': tar,
                    'impact': impact,
                    'pub_year': pub_year_norm,
                    'abstract_word_count': word_count,
                    'theory_in_title': theory_in_title,
                    'theory_in_keywords': theory_in_keywords,
                    'theory_in_abstract': theory_in_abstract,
                    'acronym_in_title': acronym_in_title,
                    'acronym_in_keywords': acronym_in_keywords,
                    'acronym_in_abstract': acronym_in_abstract,
                    'key_constructs_count': constructs_count,
                    'semantic_similarity': semantic_similarity,
                    'in_degree': in_degree,
                    'out_degree': out_degree
                })

        return pd.DataFrame(features)

    def train_classifier(self, features_df, labels):
        """
        Train the ML classifier using all hand-designed and modern NLP features.

        :param features_df: DataFrame with features
        :param labels: List of labels (1: subscribes, 0: not)
        """
        feature_cols = [col for col in features_df.columns if col != 'paper_id']
        X = features_df[feature_cols]
        y = labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"\nFeature Importance (top 5):")
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False).head(5)
        print(feature_importance)

    def predict_subscription(self, features_df):
        """
        Predict which papers subscribe to the theory using trained classifier.

        :param features_df: DataFrame with features
        :return: Predictions (1: subscribes, 0: does not)
        """
        feature_cols = [col for col in features_df.columns if col != 'paper_id']
        X = features_df[feature_cols]
        return self.classifier.predict(X)

# Example usage with mock data
if __name__ == "__main__":
    # Mock L1 papers (TAM originating papers)
    l1 = ['TAM1', 'TAM2']

    adit = ADIT('TAM', l1)

    # Mock citation data: {citing_paper: [cited_papers]}
    citation_data = {
        'PaperA': ['TAM1'],
        'PaperB': ['TAM2'],
        'PaperC': ['TAM1', 'Other'],
        'PaperD': ['Unrelated'],
        'PaperE': ['TAM1', 'TAM2']
    }

    adit.build_ecosystem(citation_data)

    # Mock paper data with all relevant fields
    papers_data = {
        'PaperA': {
            'title': 'Extension of Technology Acceptance Model',
            'abstract': 'This paper extends TAM with new constructs for mobile adoption.',
            'keywords': 'TAM, technology acceptance, mobile',
            'citations': 50,
            'year': 2015
        },
        'PaperB': {
            'title': 'Empirical test of TAM',
            'abstract': 'We test TAM in a new context with emphasis on ease of use and usefulness.',
            'keywords': 'TAM, acceptance, empirical study',
            'citations': 30,
            'year': 2012
        },
        'PaperC': {
            'title': 'Unrelated topic in information systems',
            'abstract': 'This paper studies something unrelated to technology acceptance.',
            'keywords': 'information systems, management',
            'citations': 10,
            'year': 2010
        },
        'PaperD': {
            'title': 'AI and machine learning applications',
            'abstract': 'Methods for applying machine learning to various domains.',
            'keywords': 'machine learning, AI',
            'citations': 5,
            'year': 2018
        },
        'PaperE': {
            'title': 'TAM in healthcare: Ease of use and behavioral intention',
            'abstract': 'Applies TAM to understand healthcare technology adoption with focus on usefulness.',
            'keywords': 'TAM, acceptance, healthcare, behavioral intention',
            'citations': 25,
            'year': 2014
        },
        'TAM1': {
            'title': 'Technology Acceptance Model',
            'abstract': 'Original TAM paper proposing model of technology acceptance.',
            'keywords': 'TAM, acceptance, technology',
            'citations': 1000,
            'year': 1989
        },
        'TAM2': {
            'title': 'Technology Acceptance Model extension',
            'abstract': 'Extension of TAM with more constructs.',
            'keywords': 'TAM, acceptance',
            'citations': 500,
            'year': 1992
        }
    }

    # Extract features for all L2 papers
    features = adit.extract_features(papers_data)
    print("=== Extracted Features ===")
    print(features)
    print(f"\nFeature columns: {list(features.columns)}")

    # Mock labels (1: subscribes to TAM, 0: does not)
    # PaperA: 1 (extends TAM), PaperB: 1 (tests TAM), PaperC: 0 (unrelated), 
    # PaperD: 0 (unrelated), PaperE: 1 (applies TAM)
    labels = [1, 1, 0, 0, 1]

    print("\n=== Training Classifier ===")
    adit.train_classifier(features, labels)

    print("\n=== Predictions ===")
    predictions = adit.predict_subscription(features)
    for i, paper_id in enumerate(features['paper_id']):
        print(f"{paper_id}: {'Subscribes' if predictions[i] == 1 else 'Does not subscribe'}")