# src/query_rewriting.py
"""
Query Rewriting Module using HyDE (Hypothetical Document Embeddings)

HyDE generates a hypothetical answer to the query, then uses that answer
for retrieval instead of the original query. This helps bridge the semantic
gap between questions and documents.
"""

from typing import List
from transformers import pipeline


class QueryRewriter:
    """
    Implements HyDE-based query rewriting.
    Generates hypothetical document from the query for better retrieval.
    """

    def __init__(self, llm_pipeline):
        """
        Initialize with an LLM pipeline for hypothesis generation.

        Args:
            llm_pipeline: A HuggingFace text generation pipeline
        """
        self.llm_pipeline = llm_pipeline

        # Template for generating hypothetical document
        self.hyde_template = (
            "Write a detailed paragraph that would answer the following question. "
            "Be specific and factual.\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    def rewrite_query(self, query: str, method: str = "hyde") -> str:
        """
        Rewrite query using specified method.

        Args:
            query: Original user query
            method: Rewriting method ("hyde", "expansion", or "original")

        Returns:
            Rewritten query string
        """
        if method == "original":
            return query

        elif method == "hyde":
            return self._hyde_rewrite(query)

        elif method == "expansion":
            return self._query_expansion(query)

        else:
            raise ValueError(f"Unknown rewriting method: {method}")

    def _hyde_rewrite(self, query: str) -> str:
        """
        HyDE: Generate hypothetical document from query.

        Args:
            query: Original query

        Returns:
            Hypothetical document text
        """
        prompt = self.hyde_template.format(question=query)

        response = self.llm_pipeline(
            prompt,
            max_new_tokens=80,
            do_sample=False,
            num_beams=2,
            no_repeat_ngram_size=3
        )

        hypothesis = response[0]["generated_text"].strip()

        # Combine original query with hypothesis for better retrieval
        # This balances between question semantics and answer semantics
        combined = f"{query} {hypothesis}"

        return combined

    def _query_expansion(self, query: str) -> str:
        """
        Simple query expansion by generating related terms.

        Args:
            query: Original query

        Returns:
            Expanded query
        """
        prompt = f"Rephrase this question in a different way: {query}\n\nRephrased:"

        response = self.llm_pipeline(
            prompt,
            max_new_tokens=40,
            do_sample=False,
            num_beams=2
        )

        expansion = response[0]["generated_text"].strip()

        # Combine original and expansion
        return f"{query} {expansion}"
