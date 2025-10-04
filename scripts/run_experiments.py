# run_experiments.py

import pandas as pd
import os
import sys

from src.utils import prepare_experiment_data, EMBEDDING_MODEL_NAME
from src.naive_rag import run_experiment, CLEAR_PROMPT, FEWSHOTS_PROMPT


# Ensure the results directory exists
if not os.path.exists("results"):
    os.makedirs("results")

def main():
    # --- 1. Data Loading and Preparation ---
    print("--- Starting Full Data Preparation (Loading, Cleaning, Chunking) ---")

    # Disable tokenizer parallelism to prevent issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Use 100 random questions for faster experiments (minimum requirement)
    texts_for_index, questions, ground_truths = prepare_experiment_data(sample_size=100, random_seed=42)

    if len(questions) == 0:
        print("Data preparation failed or resulted in zero questions. Exiting.")
        return

    print(f"Loaded {len(questions)} questions and {len(texts_for_index)} documents.")

    # --- 2. Define Experiment Parameters ---

    # 2.1. Embedding Model and Dimension Variations
    embedding_params = [
        {"model_name": EMBEDDING_MODEL_NAME, "dimension": 384}, # Native Dimension
        {"model_name": EMBEDDING_MODEL_NAME, "dimension": 256}, # PCA Reduced Dimension
    ]

    # 2.2. Retrieval K Variations
    top_k_variations = [1, 3, 5]

    # 2.3. Prompt Strategies
    prompt_strategies = [
        {"name": "CLEAR_INSTRUCTION", "template": CLEAR_PROMPT},
        {"name": "FEW_SHOTS", "template": FEWSHOTS_PROMPT},
    ]

    # --- 3. Execute All Combinations ---
    all_results = []
    total_experiments = len(embedding_params) * len(top_k_variations) * len(prompt_strategies)
    current_experiment = 1

    for embed_p in embedding_params:
        for k in top_k_variations:
            for prompt_s in prompt_strategies:
                print(f"\n[--- Starting Experiment {current_experiment}/{total_experiments} ---]")

                # Call the core RAG execution function
                result = run_experiment(
                    texts_for_index=texts_for_index,
                    questions=questions,
                    ground_truths=ground_truths,
                    embedding_model_name=embed_p["model_name"],
                    target_dimension=embed_p["dimension"],
                    top_k=k,
                    prompt_strategy_name=prompt_s["name"],
                    prompt_template=prompt_s["template"],
                )
                all_results.append(result)
                current_experiment += 1

    # --- 4. Output Results Table ---
    results_df = pd.DataFrame(all_results)

    csv_path = "results/comparison_analysis.csv"
    results_df.to_csv(csv_path, index=False)

    print("\n" + "#"*70)
    print("Step 4 Experimentation Complete!")
    print(f"Results saved to: {csv_path}")
    print("#"*70)

if __name__ == "__main__":
    main()