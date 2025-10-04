# test_script.py
# This script runs a single configuration (384D, K=3, CLEAR_INSTRUCTION)
# and prints detailed output for the 101st question for verification.

import pandas as pd
import os
import sys

# Import core functions and configurations
from src.utils import prepare_experiment_data, EMBEDDING_MODEL_NAME
from src.naive_rag import run_experiment, CLEAR_PROMPT, load_llm, LLM_PIPELINE


def main():
    # Load LLM to initialize global variables
    _ = load_llm()
    print("--- Starting Single Test Run ---")

    # Disable tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 1. Data Loading and Preparation (Must be called first)
    texts_for_index, questions, ground_truths = prepare_experiment_data()

    if len(questions) == 0:
        print("Data preparation failed. Exiting.")
        return

    print(f"Loaded {len(questions)} questions and {len(texts_for_index)} documents.")

    # --- 2. Define Single Experiment Parameters ---

    target_dimension = 384
    top_k = 1
    prompt_strategy_name = "CLEAR_INSTRUCTION"
    prompt_template = CLEAR_PROMPT

    # --- 3. Execute Core RAG with Verification Output ---
    print("\n" + "="*50)
    print(f"--- Running Test Configuration ---")
    print(f"  Model: {EMBEDDING_MODEL_NAME}, Target Dimension: {target_dimension}")
    print(f"  Retrieval K: {top_k}, Prompt Strategy: {prompt_strategy_name}")
    print("==================================================")

    # NOTE: We are intentionally not defining run_experiment here,
    # instead we will call the one imported from naive_rag.py,
    # which already contains the verification logic for question 101.

    result = run_experiment(
        texts_for_index=texts_for_index,
        questions=questions,
        ground_truths=ground_truths,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        target_dimension=target_dimension,
        top_k=top_k,
        prompt_strategy_name=prompt_strategy_name,
        prompt_template=prompt_template,
    )

    print("\n--- Test Run Complete ---")

if __name__ == "__main__":
    main()