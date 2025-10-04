# src/utils.py

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType
from typing import List, Dict, Tuple, Optional
from evaluate import load
from sklearn.decomposition import PCA
import pandas as pd
from datasets import load_dataset

# --- Global Configurations ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"

# --- Milvus Configurations ---
MILVUS_URI = "rag_wikipedia_mini.db"
COLLECTION_NAME = "rag_mini"
VECTOR_FIELD_NAME = "embedding"
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "IP"             # Inner Product / Cosine Similarity
SEARCH_PARAMS = {"metric_type": METRIC_TYPE, "params": {"nprobe": 32}}


# --- 1. Full Data Preparation (Loading, Cleaning, Chunking) ---
def prepare_experiment_data(sample_size: Optional[int] = None, random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Loads raw data, applies cleaning, chunking, and prepares questions/answers.

    Args:
        sample_size: If provided, randomly sample this many questions (default: None = use all)
        random_seed: Random seed for reproducible sampling (default: 42)

    Returns:
        texts_for_index: List of cleaned and chunked passages.
        questions: List of cleaned questions.
        answers: List of ground truth answers.
    """
    print("--- 1. Loading and Cleaning Passages ---")

    # Use the same data loading method as in .ipynb
    try:
        passages = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet")
    except Exception as e:
        print(f"Error loading passages dataset: {e}")
        print("Please ensure you have internet access and the 'datasets' library installed.")
        return [], [], []


    # --- Cleaning Passages ---
    s = passages["passage"].dropna().astype(str)

    # Normalize whitespace: convert NBSP, collapse all whitespace, and trim
    s = (
        s.str.replace("\u00A0", " ", regex=False)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )

    # Filter obvious noise: keep >=5 words
    s = s[s.str.split().str.len().ge(5)]

    # Drop exact duplicates
    s = s.drop_duplicates()
    texts = s.tolist()

    print(f"Passages after cleaning/filtering: {len(texts)}")

    # --- Word-Based Chunking ---
    MAX_LEN = 300      # trigger threshold (words)
    CHUNK = 220        # chunk size (words)
    OVERLAP = 50       # overlap (words)

    texts_for_index = []
    for t in texts:
        words = t.split()
        if len(words) <= MAX_LEN:
            texts_for_index.append(t)
        else:
            step = max(CHUNK - OVERLAP, 1)
            for i in range(0, len(words), step):
                piece = " ".join(words[i:i+CHUNK])
                if not piece: break
                texts_for_index.append(piece)
                if i + CHUNK >= len(words): break

    print(f"Passages after optional chunking: {len(texts_for_index)}")


    # --- 2. Loading and Cleaning Questions ---
    print("\n--- 2. Loading and Cleaning Questions ---")
    # Use the same data loading method as in .ipynb
    queries = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")

    qdf = queries.dropna(subset=["question"]).copy()

    # Normalize whitespace for questions
    qdf["question"] = (
        qdf["question"].astype(str)
        .str.replace("\u00A0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Filter short/low-signal questions
    qdf = qdf[qdf["question"].str.len().ge(5) & qdf["question"].str.split().str.len().ge(3)]

    # Drop exact duplicates
    qdf = qdf.drop_duplicates(subset=["question"]).reset_index(drop=True)

    questions = qdf["question"].tolist()
    answers = qdf["answer"].tolist()

    # Apply random sampling if requested
    if sample_size is not None and sample_size < len(questions):
        import random
        random.seed(random_seed)
        indices = random.sample(range(len(questions)), sample_size)
        indices.sort()  # Keep in original order for consistency
        questions = [questions[i] for i in indices]
        answers = [answers[i] for i in indices]
        print(f"Randomly sampled {sample_size} questions (seed={random_seed})")

    print(f"Usable questions for evaluation: {len(questions)}")

    return texts_for_index, questions, answers


# --- 3. Milvus Index and Embedding Setup with PCA ---
# ... (The rest of utils.py remains the same) ...

def setup_milvus_collection(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL_NAME,
    target_dimension: int = None
) -> Tuple[MilvusClient, SentenceTransformer, List[str], Optional[PCA]]:
    """
    Encodes texts, optionally applies PCA, sets up the Milvus collection, and inserts data.
    """
    print(f"Loading embedding model: {model_name}")
    device = "cpu"
    embedding_model = SentenceTransformer(model_name, device=device)

    native_dim = embedding_model.get_sentence_embedding_dimension()
    dimension = target_dimension if target_dimension else native_dim
    pca_transformer = None

    # 1. Encode all texts (to native dimension) with normalization for IP metric
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True  # Critical: Required for IP metric to work as cosine similarity
    )

    # 2. Apply PCA if needed
    if dimension < native_dim:
        print(f"--- PCA WARNING ---: Applying PCA to reduce dimension from {native_dim} to {dimension}.")
        pca_transformer = PCA(n_components=dimension)
        embeddings = pca_transformer.fit_transform(embeddings)

        # Re-normalize after PCA for IP metric (PCA output is not normalized)
        from sklearn.preprocessing import normalize
        embeddings = normalize(embeddings, norm='l2', axis=1).astype('float32')
        print(f"Re-normalized embeddings after PCA for IP metric.")
    elif dimension > native_dim:
        print(f"--- DIMENSION ERROR ---: Target dim ({dimension}) > native dim ({native_dim}). Using native dim.")
        dimension = native_dim
        embeddings = embeddings.astype('float32')
    else:
        # No PCA needed, dimension == native_dim
        embeddings = embeddings.astype('float32')

    # 3. Milvus Setup
    client = MilvusClient(uri=MILVUS_URI)
    if client.has_collection(collection_name=COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)

    # Explicit schema definition
    schema = client.create_schema(auto_id=False, description="RAG mini wikipedia passages")
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(field_name="passage", datatype=DataType.VARCHAR, max_length=4096)
    schema.add_field(field_name=VECTOR_FIELD_NAME, datatype=DataType.FLOAT_VECTOR, dim=dimension)

    # Define the vector index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=VECTOR_FIELD_NAME,
        index_type=INDEX_TYPE,
        metric_type=METRIC_TYPE
    )

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)

    # 4. Insertion
    data_to_insert = [
        {"id": int(i), "passage": text, VECTOR_FIELD_NAME: embedding.tolist()}
        for i, (text, embedding) in enumerate(zip(texts, embeddings))
    ]

    client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    client.flush(collection_name=COLLECTION_NAME)
    client.load_collection(COLLECTION_NAME)

    print(f"Milvus setup complete with dimension: {dimension}")
    return client, embedding_model, texts, pca_transformer


# --- 4. Retrieval from Milvus with PCA ---
def retrieve_from_milvus(
    client: MilvusClient,
    embedding_model: SentenceTransformer,
    query: str,
    top_k: int,
    pca_transformer: Optional[PCA] = None
) -> List[str]:
    """
    Retrieves the top_k most relevant documents, applying PCA and searching the 'embedding' field.
    """
    # 1. Encode the query (to native dimension) with normalization for IP metric
    query_vector = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True  # Critical: Must match indexing normalization
    )

    # 2. Apply PCA transformation if the transformer is present
    if pca_transformer:
        search_vector = pca_transformer.transform(query_vector)

        # Re-normalize after PCA for IP metric
        from sklearn.preprocessing import normalize
        search_vector = normalize(search_vector, norm='l2', axis=1)
    else:
        search_vector = query_vector

    search_data = search_vector.tolist()

    # 3. Perform search
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=search_data,
        anns_field=VECTOR_FIELD_NAME,
        limit=top_k,
        output_fields=["passage"],
        search_params=SEARCH_PARAMS
    )

    # 4. Extract the texts
    retrieved_texts = [hit["entity"]["passage"] for hit in res[0]]

    return retrieved_texts


# --- 5. Evaluation Metrics ---
def calculate_f1_em(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculates F1 Score and Exact Match (EM) using the SQuAD evaluation script.
    """
    squad_metric = load("squad")
    formatted_preds = [{"prediction_text": pred, "id": str(i)} for i, pred in enumerate(predictions)]
    formatted_refs = [{"answers": {"answer_start": [0], "text": [ref]}, "id": str(i)} for i, ref in enumerate(references)]
    results = squad_metric.compute(predictions=formatted_preds, references=formatted_refs)
    return {"f1": results.get("f1", 0.0), "exact_match": results.get("exact_match", 0.0)}