# src/reranking.py
"""
Reranking Module using Cross-Encoder

Cross-encoders score query-document pairs more accurately than bi-encoders
by processing them together. This module reranks initial retrieval results
to improve relevance.
"""

from typing import List, Tuple
from sentence_transformers import CrossEncoder
import numpy as np


class DocumentReranker:
    """
    Implements cross-encoder based reranking of retrieved documents.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder model for reranking.

        Args:
            model_name: Name of the cross-encoder model from sentence-transformers
                       Default is a lightweight model trained on MS MARCO dataset
        """
        self.model_name = model_name
        self.cross_encoder = None
        self._load_model()

    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            self.cross_encoder = CrossEncoder(self.model_name, max_length=512)
            print(f"Reranker loaded: {self.model_name}")
        except Exception as e:
            print(f"Warning: Failed to load cross-encoder: {e}")
            print("Reranking will be skipped in advanced RAG.")
            self.cross_encoder = None

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> Tuple[List[str], List[float]]:
        """
        Rerank documents using cross-encoder scoring.

        Args:
            query: The search query
            documents: List of retrieved documents to rerank
            top_k: Number of top documents to return (None = return all)

        Returns:
            Tuple of (reranked_documents, scores)
        """
        if self.cross_encoder is None:
            # If model failed to load, return original order
            return documents, [1.0] * len(documents)

        if len(documents) == 0:
            return [], []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score all pairs
        scores = self.cross_encoder.predict(pairs)

        # Sort by score (descending)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Extract top_k if specified
        if top_k is not None:
            doc_score_pairs = doc_score_pairs[:top_k]

        reranked_docs = [doc for doc, _ in doc_score_pairs]
        reranked_scores = [score for _, score in doc_score_pairs]

        return reranked_docs, reranked_scores

    def rerank_with_diversity(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        diversity_weight: float = 0.3
    ) -> Tuple[List[str], List[float]]:
        """
        Rerank with diversity penalty using MMR-like approach.

        Args:
            query: The search query
            documents: List of retrieved documents
            top_k: Number of documents to return
            diversity_weight: Weight for diversity (0 = pure relevance, 1 = pure diversity)

        Returns:
            Tuple of (reranked_documents, scores)
        """
        if self.cross_encoder is None or len(documents) == 0:
            return documents[:top_k] if top_k else documents, []

        # Score all documents
        pairs = [[query, doc] for doc in documents]
        relevance_scores = self.cross_encoder.predict(pairs)

        # Maximal Marginal Relevance (MMR) selection
        selected_docs = []
        selected_scores = []
        remaining_indices = list(range(len(documents)))

        for _ in range(min(top_k, len(documents))):
            if not remaining_indices:
                break

            # Calculate MMR score for each remaining document
            mmr_scores = []
            for idx in remaining_indices:
                relevance = relevance_scores[idx]

                # Calculate diversity penalty (simple word overlap)
                diversity_penalty = 0
                if selected_docs:
                    overlaps = [
                        self._word_overlap(documents[idx], sel_doc)
                        for sel_doc in selected_docs
                    ]
                    diversity_penalty = max(overlaps) if overlaps else 0

                mmr = (1 - diversity_weight) * relevance - diversity_weight * diversity_penalty
                mmr_scores.append((idx, mmr))

            # Select document with highest MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_docs.append(documents[best_idx])
            selected_scores.append(relevance_scores[best_idx])
            remaining_indices.remove(best_idx)

        return selected_docs, selected_scores

    def _word_overlap(self, doc1: str, doc2: str) -> float:
        """
        Calculate simple word overlap between two documents.

        Args:
            doc1, doc2: Documents to compare

        Returns:
            Overlap score [0, 1]
        """
        words1 = set(doc1.lower().split())
        words2 = set(doc2.lower().split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))

        return overlap / total if total > 0 else 0.0
