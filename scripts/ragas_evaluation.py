# run_custom_evaluation.py
"""
Advanced Evaluation with RAGAs

Evaluates both naive and advanced RAG systems using RAGAs framework.
Provides metrics beyond token matching: faithfulness, relevancy, precision, recall.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

import json
import pandas as pd
from typing import List, Dict, Tuple
from openai import OpenAI
from src.utils import prepare_experiment_data, EMBEDDING_MODEL_NAME, setup_milvus_collection, retrieve_from_milvus, COLLECTION_NAME
from src.naive_rag import generate_answer, CLEAR_PROMPT
from src.advanced_rag import QUERY_REWRITER, RERANKER
from tqdm import tqdm
import time

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


EVALUATION_PROMPT = """You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.

Given the following:
- Question: {question}
- Generated Answer: {answer}
- Retrieved Context: {context}
- Ground Truth Answer: {ground_truth}

Please evaluate the RAG system on these 4 metrics (rate each 0.0 to 1.0):

1. **Faithfulness**: Is the generated answer factually grounded in the retrieved context?
   - 1.0 = All claims in answer are supported by context
   - 0.0 = Answer contains unsupported claims

2. **Answer Relevancy**: Does the generated answer directly address the question?
   - 1.0 = Completely relevant and on-topic
   - 0.0 = Irrelevant or off-topic

3. **Context Recall**: Does the retrieved context contain the information needed to answer the question?
   - 1.0 = Context has all necessary information
   - 0.0 = Context missing key information

4. **Context Precision**: Are the most relevant parts of the context ranked/presented first?
   - 1.0 = Most relevant info appears early in context
   - 0.0 = Relevant info buried or missing

Return ONLY a valid JSON object with these exact keys:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0, "context_precision": 0.0}}

Do not include any explanation, just the JSON."""


def evaluate_single_question(question: str, answer: str, context: str, ground_truth: str, retry_delay: int = 2) -> Dict[str, float]:
    """
    Evaluate a single question using OpenAI API.
    Returns all 4 metrics in one API call.
    """
    prompt = EVALUATION_PROMPT.format(
        question=question,
        answer=answer,
        context=context,
        ground_truth=ground_truth
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise RAG evaluation assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200,
                timeout=30
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON
            if result_text.startswith("```json"):
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif result_text.startswith("```"):
                result_text = result_text.split("```")[1].split("```")[0].strip()

            scores = json.loads(result_text)

            # Validate keys
            required_keys = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
            if all(k in scores for k in required_keys):
                return scores
            else:
                print(f"  Warning: Missing keys in response, retrying...")

        except json.JSONDecodeError as e:
            print(f"  JSON decode error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except Exception as e:
            print(f"  API error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    # Return zeros if all retries failed
    print(f"  Failed to evaluate after {max_retries} attempts, returning zeros")
    return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0, "context_precision": 0.0}


def collect_and_evaluate_naive(texts_for_index, questions, ground_truths, sample_size=100):
    """Run naive RAG and evaluate."""
    print("\n" + "="*70)
    print("NAIVE RAG - COLLECT & EVALUATE")
    print("="*70)

    from src.naive_rag import LLM_PIPELINE, generate_answer

    # Setup Milvus
    milvus_client, embed_model, texts, pca_transformer = setup_milvus_collection(
        texts_for_index,
        model_name=EMBEDDING_MODEL_NAME,
        target_dimension=256
    )

    questions_sample = questions[:sample_size]
    ground_truths_sample = ground_truths[:sample_size]

    all_scores = []

    print(f"\nProcessing {len(questions_sample)} questions...")

    for i, (question, ground_truth) in enumerate(tqdm(zip(questions_sample, ground_truths_sample), total=len(questions_sample))):
        # Retrieve
        retrieved_context = retrieve_from_milvus(
            milvus_client,
            embed_model,
            question,
            top_k=3,
            pca_transformer=pca_transformer
        )

        # Generate
        answer = generate_answer(retrieved_context, question, CLEAR_PROMPT)

        # Evaluate
        context_str = "\n\n".join(retrieved_context)
        scores = evaluate_single_question(question, answer, context_str, ground_truth)
        scores["question_id"] = i
        all_scores.append(scores)

        # Rate limiting - wait 1 second between calls
        if i < len(questions_sample) - 1:
            time.sleep(1)

    # Cleanup
    milvus_client.drop_collection(COLLECTION_NAME)

    # Calculate averages
    avg_scores = {
        "faithfulness": sum(s["faithfulness"] for s in all_scores) / len(all_scores),
        "answer_relevancy": sum(s["answer_relevancy"] for s in all_scores) / len(all_scores),
        "context_recall": sum(s["context_recall"] for s in all_scores) / len(all_scores),
        "context_precision": sum(s["context_precision"] for s in all_scores) / len(all_scores),
    }

    print(f"\nNaive RAG Results:")
    for metric, score in avg_scores.items():
        print(f"  {metric:20s}: {score:.4f}")

    return avg_scores, all_scores


def collect_and_evaluate_advanced(texts_for_index, questions, ground_truths, sample_size=100):
    """Run advanced RAG and evaluate."""
    print("\n" + "="*70)
    print("ADVANCED RAG - COLLECT & EVALUATE")
    print("="*70)

    from src.advanced_rag import generate_answer

    # Setup Milvus
    milvus_client, embed_model, texts, pca_transformer = setup_milvus_collection(
        texts_for_index,
        model_name=EMBEDDING_MODEL_NAME,
        target_dimension=256
    )

    questions_sample = questions[:sample_size]
    ground_truths_sample = ground_truths[:sample_size]

    all_scores = []

    print(f"\nProcessing {len(questions_sample)} questions with enhancements...")

    for i, (question, ground_truth) in enumerate(tqdm(zip(questions_sample, ground_truths_sample), total=len(questions_sample))):
        # Query rewriting
        rewritten_query = QUERY_REWRITER.rewrite_query(question, method="hyde")

        # Initial retrieval
        retrieved_context = retrieve_from_milvus(
            milvus_client,
            embed_model,
            rewritten_query,
            top_k=10,
            pca_transformer=pca_transformer
        )

        # Reranking
        reranked_docs, scores = RERANKER.rerank(
            question,
            retrieved_context,
            top_k=1
        )

        # Generate
        answer = generate_answer(reranked_docs, question, CLEAR_PROMPT)

        # Evaluate
        context_str = "\n\n".join(reranked_docs)
        scores = evaluate_single_question(question, answer, context_str, ground_truth)
        scores["question_id"] = i
        all_scores.append(scores)

        # Rate limiting - wait 1 second between calls
        if i < len(questions_sample) - 1:
            time.sleep(1)

    # Cleanup
    milvus_client.drop_collection(COLLECTION_NAME)

    # Calculate averages
    avg_scores = {
        "faithfulness": sum(s["faithfulness"] for s in all_scores) / len(all_scores),
        "answer_relevancy": sum(s["answer_relevancy"] for s in all_scores) / len(all_scores),
        "context_recall": sum(s["context_recall"] for s in all_scores) / len(all_scores),
        "context_precision": sum(s["context_precision"] for s in all_scores) / len(all_scores),
    }

    print(f"\nAdvanced RAG Results:")
    for metric, score in avg_scores.items():
        print(f"  {metric:20s}: {score:.4f}")

    return avg_scores, all_scores


def main():
    print("\n" + "#"*70)
    print("CUSTOM RAG EVALUATION - SINGLE API CALL APPROACH")
    print("#"*70)

    # Load data
    print("\nLoading data...")
    texts_for_index, questions, ground_truths = prepare_experiment_data(sample_size=100, random_seed=42)
    print(f"Loaded {len(questions)} questions and {len(texts_for_index)} documents.")

    EVAL_SAMPLE_SIZE = 100
    print(f"\nEvaluating {EVAL_SAMPLE_SIZE} questions")
    print(f"Total API calls: {EVAL_SAMPLE_SIZE * 2} (1 per question, 2 systems)")
    print(f"Estimated time: ~3-5 minutes (with 1s rate limiting)")

    # Evaluate naive RAG
    naive_avg, naive_details = collect_and_evaluate_naive(
        texts_for_index, questions, ground_truths, sample_size=EVAL_SAMPLE_SIZE
    )

    # Evaluate advanced RAG
    advanced_avg, advanced_details = collect_and_evaluate_advanced(
        texts_for_index, questions, ground_truths, sample_size=EVAL_SAMPLE_SIZE
    )

    # Compare
    print("\n" + "#"*70)
    print("COMPARISON RESULTS")
    print("#"*70)

    comparison_df = pd.DataFrame({
        "Metric": ["faithfulness", "answer_relevancy", "context_recall", "context_precision"],
        "Naive RAG": [naive_avg[m] for m in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]],
        "Advanced RAG": [advanced_avg[m] for m in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]],
    })

    comparison_df["Improvement"] = comparison_df["Advanced RAG"] - comparison_df["Naive RAG"]
    comparison_df["Improvement %"] = (comparison_df["Improvement"] / comparison_df["Naive RAG"] * 100).round(2)

    print(comparison_df.to_string(index=False))

    # Save results
    comparison_df.to_csv("results/ragas_comparison.csv", index=False)

    # Save detailed scores
    pd.DataFrame(naive_details).to_csv("results/ragas_naive.csv", index=False)
    pd.DataFrame(advanced_details).to_csv("results/ragas_advanced.csv", index=False)

    print(f"\nResults saved to:")
    print(f"  - results/ragas_comparison.csv")
    print(f"  - results/ragas_naive.csv")
    print(f"  - results/ragas_advanced.csv")


if __name__ == "__main__":
    main()
