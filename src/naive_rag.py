# src/naive_rag.py

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Dict
from sklearn.decomposition import PCA
from src.utils import (
    setup_milvus_collection,
    retrieve_from_milvus,
    calculate_f1_em,
    LLM_MODEL_NAME,
    COLLECTION_NAME
)

# --- 1. LLM Setup (Load once) ---
def load_llm(model_name: str = LLM_MODEL_NAME):
    """Loads the LLM model using pipeline."""
    from transformers import pipeline
    qa_pipeline = pipeline("text2text-generation", model=model_name, device = 'cpu')
    print(f"LLM loaded: {model_name}")
    return qa_pipeline

# Load the LLM globally on script start
LLM_PIPELINE = load_llm()


# --- 2. Prompt Templates for Experimentation ---

# 1. Clear Instruction Prompt
CLEAR_PROMPT = (
    """
    You are a helpful assistant that answers questions ONLY using the provided passages.
    If any part is unsupported, respond with exactly: "Unable to verify from provided passages."
    Use plain, neutral language.
    Do not add extra background, examples, or definitions unless they appear word-for-word in the passages.
    Keep answers concise and factual, ≤60 words.
    """
)

# 2. Few-Shots Prompt
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


# --- 3. Answer Generation ---
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
    # prompt_clear = f"""{prompt_clear}\n\nContext: {context}: \n\nQuestion: {query} """
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


# --- 4. Core Experiment Runner ---
def run_experiment(
    texts_for_index: List[str],
    questions: List[str],
    ground_truths: List[str],
    embedding_model_name: str,
    target_dimension: int,
    top_k: int,
    prompt_strategy_name: str,
    prompt_template: str
) -> Dict:
    """
    Executes a single RAG experiment configuration (Embedding Dim, Top-K, Prompt) and returns metrics.
    """
    print("\n" + "="*50)
    print(f"--- Running Experiment Configuration ---")
    print(f"  Model: {embedding_model_name}, Target Dimension: {target_dimension}")
    print(f"  Retrieval K: {top_k}, Prompt Strategy: {prompt_strategy_name}")
    print("="*50)

    # 1. Indexing and Embedding (Milvus setup with PCA)
    milvus_client, embed_model, texts, pca_transformer = setup_milvus_collection(
        texts_for_index,
        model_name=embedding_model_name,
        target_dimension=target_dimension
    )

    generated_answers = []

    # 2. Retrieval and Generation Loop
    for i, question in enumerate(questions):
        # 2a. Retrieval
        retrieved_context = retrieve_from_milvus(
            milvus_client,
            embed_model,
            question,
            top_k,
            pca_transformer
        )

        # 2b. Generation
        answer = generate_answer(
            retrieved_context,
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
        "top_k": top_k,
        "prompt_strategy_type": prompt_strategy_name,
        "f1": metrics["f1"],
        "exact_match": metrics["exact_match"],
    }

    print("\n--- Experiment Results ---")
    print(f"  F1 Score: {result['f1']:.4f}")
    print(f"  Exact Match: {result['exact_match']:.4f}")
    print("--------------------------")

    # Clean up Milvus collection after the experiment run
    milvus_client.drop_collection(COLLECTION_NAME)

    return result