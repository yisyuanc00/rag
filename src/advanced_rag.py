# src/advanced_rag.py
"""
Advanced RAG Implementation with Query Rewriting and Reranking

This module extends the naive RAG with two enhancements:
1. Query Rewriting using HyDE (Hypothetical Document Embeddings)
2. Reranking using Cross-Encoder scoring
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Dict, Optional
from sklearn.decomposition import PCA

from src.utils import (
    setup_milvus_collection,
    retrieve_from_milvus,
    calculate_f1_em,
    LLM_MODEL_NAME,
    COLLECTION_NAME
)
from src.query_rewriting import QueryRewriter
from src.reranking import DocumentReranker


# --- 1. LLM Setup (Load once) ---
def load_llm(model_name: str = LLM_MODEL_NAME):
    """Loads the LLM model using pipeline."""
    from transformers import pipeline
    qa_pipeline = pipeline("text2text-generation", model=model_name, device='cpu')
    print(f"LLM loaded: {model_name}")
    return qa_pipeline

# Load the LLM globally on script start
LLM_PIPELINE = load_llm()

# Initialize query rewriter and reranker
QUERY_REWRITER = QueryRewriter(LLM_PIPELINE)
RERANKER = DocumentReranker()


# --- 2. Prompt Templates (same as naive RAG) ---

CLEAR_PROMPT = (
    """
    You are a helpful assistant that answers questions ONLY using the provided passages.
    If any part is unsupported, respond with exactly: "Unable to verify from provided passages."
    Use plain, neutral language.
    Do not add extra background, examples, or definitions unless they appear word-for-word in the passages.
    Keep answers concise and factual, ≤60 words.
    """
)

FEWSHOTS_PROMPT = (
    """
    You are a helpful assistant that answers questions ONLY using the provided passages.
    If any part is unsupported, respond with exactly: "Unable to verify from provided passages."
    Use plain, neutral language.
    Do not add extra background, examples, or definitions unless they appear word-for-word in the passages.
    Keep answers concise and factual, ≤60 words.

    Follow the examples exactly.

    Example 1
    Q: In what year was Company A founded?
    Passages:
    [1] Company A was founded in 2004.
    A:
    Founded in 2004. [1]

    Example 2
    Q: Who is the CEO of Company B?
    Passages:
    [1] Company B was founded in 1999.
    [2] Its current CEO is Jane Smith.
    A:
    Jane Smith is the CEO. [2]

    Example 3
    Q: What is Company C's market share in 2023?
    Passages:
    [P1] Company C is a technology company headquartered in London.
    A:
    Unable to verify from provided passages."""
)


# --- 3. Advanced Answer Generation ---
def generate_answer(
    retrieved_context: List[str],
    question: str,
    prompt_template: str
) -> str:
    """
    Generates an answer using the LLM pipeline.
    """
    # Build context string
    context_list_with_citation = [
        f"[{i+1}] {chunk[:600]}"
        for i, chunk in enumerate(retrieved_context)
    ]
    context_str = "\n\n".join(context_list_with_citation)

    # Format the final prompt
    prompt = f"""{prompt_template}\n\nContext: {context_str}: \n\nQuestion: {question} """

    # LLM Inference using pipeline
    response = LLM_PIPELINE(
        prompt,
        max_new_tokens=60,
        do_sample=False,
        num_beams=3,
        no_repeat_ngram_size=3
    )

    answer = response[0]["generated_text"]
    return answer.strip()


# --- 4. Advanced Experiment Runner with Query Rewriting and Reranking ---
def run_advanced_experiment(
    texts_for_index: List[str],
    questions: List[str],
    ground_truths: List[str],
    embedding_model_name: str,
    target_dimension: int,
    initial_k: int,
    final_k: int,
    prompt_strategy_name: str,
    prompt_template: str,
    use_query_rewriting: bool = True,
    use_reranking: bool = True,
    query_rewrite_method: str = "hyde"
) -> Dict:
    """
    Executes an advanced RAG experiment with query rewriting and reranking.

    Args:
        texts_for_index: Documents to index
        questions: List of questions
        ground_truths: List of ground truth answers
        embedding_model_name: Name of embedding model
        target_dimension: Target embedding dimension
        initial_k: Number of documents to retrieve initially (before reranking)
        final_k: Number of documents to use for generation (after reranking)
        prompt_strategy_name: Name of prompt strategy
        prompt_template: Prompt template string
        use_query_rewriting: Whether to use query rewriting
        use_reranking: Whether to use reranking
        query_rewrite_method: Method for query rewriting ("hyde" or "expansion")

    Returns:
        Dictionary with experiment results and metrics
    """
    print("\n" + "="*70)
    print(f"--- Running ADVANCED RAG Experiment ---")
    print(f"  Model: {embedding_model_name}, Dimension: {target_dimension}")
    print(f"  Initial Retrieval K: {initial_k}, Final K: {final_k}")
    print(f"  Prompt Strategy: {prompt_strategy_name}")
    print(f"  Query Rewriting: {use_query_rewriting} ({query_rewrite_method if use_query_rewriting else 'N/A'})")
    print(f"  Reranking: {use_reranking}")
    print("="*70)

    # 1. Indexing and Embedding (Milvus setup with PCA)
    milvus_client, embed_model, texts, pca_transformer = setup_milvus_collection(
        texts_for_index,
        model_name=embedding_model_name,
        target_dimension=target_dimension
    )

    generated_answers = []

    # 2. Retrieval and Generation Loop with Enhancements
    for i, question in enumerate(questions):
        # 2a. Query Rewriting (if enabled)
        if use_query_rewriting:
            rewritten_query = QUERY_REWRITER.rewrite_query(question, method=query_rewrite_method)
        else:
            rewritten_query = question

        # 2b. Initial Retrieval (retrieve more documents than needed for reranking)
        retrieved_context = retrieve_from_milvus(
            milvus_client,
            embed_model,
            rewritten_query,
            initial_k,
            pca_transformer
        )

        # 2c. Reranking (if enabled)
        if use_reranking and len(retrieved_context) > 0:
            # Rerank and select top final_k
            reranked_docs, scores = RERANKER.rerank(
                question,  # Use original question for reranking
                retrieved_context,
                top_k=final_k
            )
            final_context = reranked_docs
        else:
            # Just take top final_k from initial retrieval
            final_context = retrieved_context[:final_k]

        # 2d. Generation
        answer = generate_answer(
            final_context,
            question,
            prompt_template
        )
        generated_answers.append(answer)

        if (i + 1) % 50 == 0 and i < len(questions) - 1:
            print(f"  Processed {i + 1}/{len(questions)} questions.")

    # 3. Evaluation
    metrics = calculate_f1_em(generated_answers, ground_truths)

    result = {
        "embedding_model": embedding_model_name,
        "target_dimension": target_dimension,
        "initial_k": initial_k,
        "final_k": final_k,
        "prompt_strategy_type": prompt_strategy_name,
        "query_rewriting": use_query_rewriting,
        "query_rewrite_method": query_rewrite_method if use_query_rewriting else "none",
        "reranking": use_reranking,
        "f1": metrics["f1"],
        "exact_match": metrics["exact_match"],
    }

    print("\n--- Advanced Experiment Results ---")
    print(f"  F1 Score: {result['f1']:.4f}")
    print(f"  Exact Match: {result['exact_match']:.4f}")
    print("-----------------------------------")

    # Clean up Milvus collection after the experiment run
    milvus_client.drop_collection(COLLECTION_NAME)

    return result
