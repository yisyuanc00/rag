# run_advanced_experiments.py
"""
Run experiments comparing Naive RAG vs Advanced RAG

This script tests the impact of:
1. Query Rewriting (HyDE)
2. Reranking (Cross-Encoder)
Both individually and combined
"""

import pandas as pd
import os
import sys

from src.utils import prepare_experiment_data, EMBEDDING_MODEL_NAME
from src.advanced_rag import run_advanced_experiment, CLEAR_PROMPT, FEWSHOTS_PROMPT


# Ensure the results directory exists
if not os.path.exists("results"):
    os.makedirs("results")


def main():
    # --- 1. Data Loading and Preparation ---
    print("=" * 80)
    print("ADVANCED RAG EXPERIMENTATION")
    print("=" * 80)

    # Disable tokenizer parallelism to prevent issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Use 100 random questions for faster experiments (minimum requirement)
    texts_for_index, questions, ground_truths = prepare_experiment_data(sample_size=100, random_seed=42)

    if len(questions) == 0:
        print("Data preparation failed or resulted in zero questions. Exiting.")
        return

    print(f"\nLoaded {len(questions)} questions and {len(texts_for_index)} documents.")

    # --- 2. Define Experiment Parameters ---
    # Test all combinations like in naive RAG (Step 4):
    # - 2 embedding dimensions (384D, 256D)
    # - 3 retrieval K values (1, 3, 5)
    # - 2 prompt strategies (CLEAR_INSTRUCTION, FEW_SHOTS)
    # Total: 2 × 3 × 2 = 12 configurations

    embedding_dimensions = [384, 256]
    top_k_values = [1, 3, 5]
    prompt_strategies = [
        {"name": "CLEAR_INSTRUCTION", "template": CLEAR_PROMPT},
        {"name": "FEW_SHOTS", "template": FEWSHOTS_PROMPT}
    ]

    # --- 3. Generate All Advanced RAG Configurations ---
    # All configurations use both enhancements (Query Rewriting + Reranking)

    advanced_configs = []
    for dim in embedding_dimensions:
        for final_k in top_k_values:
            for prompt in prompt_strategies:
                config = {
                    "name": f"Advanced RAG (dim={dim}, K={final_k}, prompt={prompt['name']})",
                    "dimension": dim,
                    "initial_k": 10,  # Always retrieve 10 for reranking
                    "final_k": final_k,
                    "query_rewriting": True,
                    "reranking": True,
                    "query_rewrite_method": "hyde",
                    "prompt_name": prompt["name"],
                    "prompt_template": prompt["template"]
                }
                advanced_configs.append(config)

    # --- 4. Execute All Configurations ---
    all_results = []
    total_experiments = len(advanced_configs)

    for idx, config in enumerate(advanced_configs, 1):
        print(f"\n{'#'*80}")
        print(f"Experiment {idx}/{total_experiments}: {config['name']}")
        print(f"{'#'*80}")

        result = run_advanced_experiment(
            texts_for_index=texts_for_index,
            questions=questions,
            ground_truths=ground_truths,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            target_dimension=config["dimension"],
            initial_k=config["initial_k"],
            final_k=config["final_k"],
            prompt_strategy_name=config["prompt_name"],
            prompt_template=config["prompt_template"],
            use_query_rewriting=config["query_rewriting"],
            use_reranking=config["reranking"],
            query_rewrite_method=config["query_rewrite_method"]
        )

        # Add configuration name to result
        result["config_name"] = config["name"]

        all_results.append(result)

    # --- 5. Output Results Table ---
    results_df = pd.DataFrame(all_results)

    # Reorder columns for better readability
    column_order = [
        "config_name",
        "query_rewriting",
        "query_rewrite_method",
        "reranking",
        "initial_k",
        "final_k",
        "f1",
        "exact_match",
        "embedding_model",
        "target_dimension",
        "prompt_strategy_type"
    ]
    results_df = results_df[column_order]

    csv_path = "results/advanced_rag_analysis.csv"
    results_df.to_csv(csv_path, index=False)

    print("\n" + "#"*80)
    print("ADVANCED RAG EXPERIMENTATION COMPLETE!")
    print(f"Results saved to: {csv_path}")
    print("#"*80)

    # Print summary
    print("\n--- RESULTS SUMMARY ---")
    print(results_df[["config_name", "f1", "exact_match"]].to_string(index=False))
    print("\n")

    # Find best configuration
    best_idx = results_df["f1"].idxmax()
    best_config = results_df.loc[best_idx]
    print(f"Best Configuration: {best_config['config_name']}")
    print(f"  F1 Score: {best_config['f1']:.4f}")
    print(f"  Exact Match: {best_config['exact_match']:.4f}")
    print(f"  Improvement over baseline: {best_config['f1'] - results_df.loc[0, 'f1']:.4f} F1 points")


if __name__ == "__main__":
    main()
