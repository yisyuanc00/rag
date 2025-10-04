# RAG System Implementation

**Dataset**: RAG-mini-Wikipedia (HuggingFace rag-datasets)
**Objective**: Build, evaluate, and enhance a Retrieval-Augmented Generation system

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Setup Instructions](#setup-instructions)
4. [Usage Guide](#usage-guide)
5. [Results Summary](#results-summary)
6. [Project Structure](#project-structure)
7. [Documentation](#documentation)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project implements a complete RAG pipeline from naive baseline to advanced enhancements, with comprehensive evaluation using both traditional metrics (F1, EM) and modern LLM-based assessment (RAGAs).

### Key Components

- **Naive RAG**: Baseline system with bi-encoder retrieval + small LLM generation
- **Advanced RAG**: Enhanced with Query Rewriting (HyDE) + Cross-Encoder Reranking
- **Evaluation**: Multi-metric assessment (SQuAD, RAGAs)
- **Experiments**: Systematic parameter exploration (12 configurations)

### Technologies

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: Milvus Lite (local SQLite-based)
- **LLM**: FLAN-T5-small (60M parameters, CPU-friendly)
- **Evaluation**: HuggingFace SQuAD + RAGAs framework

---

## System Architecture

```
┌─────────────┐
│   Question  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│  Query Rewriting (Optional) │  ← Advanced RAG Feature #1
│      HyDE / Expansion       │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│   Embedding Model           │
│ (sentence-transformers)     │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│   Vector Search             │
│   (Milvus + IVF_FLAT)       │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Reranking (Optional)       │  ← Advanced RAG Feature #2
│   Cross-Encoder Scoring     │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│   Context Assembly          │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│   LLM Generation            │
│   (FLAN-T5-small)           │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────┐
│   Answer    │
└─────────────┘
```

---

## Setup Instructions

### Prerequisites

- Python 3.12+ (tested on 3.12)
- macOS, Linux, or Windows
- 8GB+ RAM (for embeddings + LLM)
- OpenAI API key (for RAGAs evaluation, Step 6 only)

### Installation

1. **Clone Repository**

```bash
cd "Applications of NL(X) and LLM/95820_hw2"
```

2. **Create Virtual Environment**

```bash
python3 -m venv ~/venvs/rag-pip
source ~/venvs/rag-pip/bin/activate  # macOS/Linux
# or
~/venvs/rag-pip/Scripts/activate  # Windows
```

3. **Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencies include**:
- PyTorch 2.5.1 (CPU version)
- transformers 4.56.2
- sentence-transformers 5.1.1
- pymilvus 2.4+ (with milvus-lite)
- ragas 0.3.5
- evaluate, datasets, pandas, scikit-learn

4. **Configure OpenAI API Key** (for RAGAs only)

```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

*Or skip this if only running Steps 2-5 (SQuAD metrics don't need API key)*

5. **Verify Installation**

```bash
python -m src.test  # Quick test of naive RAG
python test_advanced_features.py  # Test advanced features
```

---

## 📖 Usage Guide

### Step 2-4: Naive RAG Experiments

Run all 12 naive RAG configurations:

```bash
python scripts/run_experiments.py
```

**Parameters tested**:
- Embedding dimensions: 384 (native), 256 (PCA-reduced)
- Retrieval strategies: Top-1, Top-3, Top-5
- Prompt strategies: CLEAR_INSTRUCTION, FEW_SHOTS

**Runtime**: ~60-120 minutes on CPU
**Output**: `results/comparison_analysis.csv`

### Step 5: Advanced RAG Experiments

Run enhanced RAG with query rewriting + reranking:

```bash
python scripts/run_advanced_experiments.py
```

**Configurations tested**:
1. Baseline (no enhancements)
2. Query Rewriting only
3. Reranking only
4. Both enhancements (Top-1 final)
5. Both enhancements (Top-3 final)

**Runtime**: ~90-150 minutes on CPU (cross-encoder is slower)
**Output**: `results/advanced_rag_analysis.csv`

### Step 6: RAGAs Evaluation

Compare naive vs. advanced RAG using LLM-as-judge metrics:

```bash
python scripts/ragas_evaluation.py
```

**Requires**: OpenAI API key in `.env`
**Sample size**: 100 questions (adjustable in script)
**Runtime**: ~15-30 minutes
**Cost**: ~$0.30-0.80 (GPT-3.5-turbo)
**Output**: `results/ragas_comparison.csv`


---

## Results Summary

### Naive RAG (Step 3-4)

**Best Configuration**: 384D + Top-1 + CLEAR_INSTRUCTION

| Configuration | F1 Score | Exact Match |
|--------------|----------|-------------|
| **Best** (384D, K=1, CLEAR) | **39.07%** | **31.08%** |
| 384D, K=1, FEW_SHOTS | 32.32% | 24.21% |
| 384D, K=3, CLEAR | 32.87% | 23.45% |
| 384D, K=5, CLEAR | 32.97% | 23.12% |
| 256D, K=1, CLEAR | 38.92% | 30.97% |

**Key Findings**:
- Top-1 outperforms Top-3/Top-5 (less context dilution)
- CLEAR prompt beats FEW_SHOTS (+6.75 F1 points)
- PCA dimension reduction (384 to 256) has minimal impact (-0.15 F1)

### Advanced RAG (Step 5)

**Expected Improvements** (based on feature testing):
- Query Rewriting (HyDE): +2-5% F1 (better retrieval recall)
- Cross-Encoder Reranking: +3-8% F1 (better context precision)
- Combined: +5-12% F1 (synergistic effect)

*Full results available after running `scripts/run_advanced_experiments.py`*

### RAGAs Evaluation (Step 6)

**Metrics** (0-1 scale):
- **Faithfulness**: Answer grounded in context
- **Answer Relevancy**: Answer addresses question
- **Context Precision**: Relevant passages ranked high
- **Context Recall**: Retrieved context contains answer info

*Results available after running `scripts/ragas_evaluation.py` with OpenAI API key*

---

## Project Structure

```
95820_hw2/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .env                            # API keys (gitignored)
├── .gitignore                      # Git exclusions
│
├── src/                            # Core modules
│   ├── utils.py                    # Data prep, Milvus setup, metrics
│   ├── naive_rag.py                # Baseline RAG implementation
│   ├── advanced_rag.py             # Enhanced RAG with features
│   ├── query_rewriting.py          # HyDE query rewriting
│   ├── reranking.py                # Cross-encoder reranking
│   ├── evaluation.py               # RAGAs evaluation module
│   └── test.py                     # Unit tests
│
├── scripts/                        # Executable experiment scripts
│   ├── run_experiments.py          # Naive RAG experiments (12 configs)
│   ├── run_advanced_experiments.py # Advanced RAG experiments (12 configs)
│   └── ragas_evaluation.py         # RAGAs evaluation comparison
│
├── notebooks/                      # Jupyter notebooks
│   └── rag-Starter-Code.ipynb      # Initial exploration notebook
│
├── data/                           # Database files (gitignored)
│   └── *.db                        # Milvus Lite databases
│
├── results/                        # Experiment outputs
│   ├── comparison_analysis.csv     # Naive RAG results (12 configs)
│   ├── advanced_rag_analysis.csv   # Advanced RAG results (12 configs)
│   └── ragas_comparison.csv        # RAGAs metric comparison
│
├── logs/                           # Execution logs (gitignored)
│   └── *.log                       # Experiment run logs
│
├── deliverables/                   # Final submission documents
│   ├── Final_Technical_Report.md   # Complete technical report
│   └── ai_usage_log.md             # AI assistance documentation
│
└── docs/                           # Additional documentation
    └── PROJECT_STRUCTURE.md        # Detailed structure guide
```

See `docs/PROJECT_STRUCTURE.md` for detailed file descriptions.

---

### Code Documentation

All modules include:
- Docstrings for functions and classes
- Inline comments explaining complex logic
- Type hints for parameters and returns
- Error handling with descriptive messages

---

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'src'**

```bash
# Solution: Run scripts from project root
python scripts/run_experiments.py  # Correct
# Not: cd scripts && python run_experiments.py  # Wrong (imports won't work)
```

**2. Milvus database locked error**

```
ConnectionConfigException: Open local milvus failed
```

**Solution**: Another process is using the database

```bash
# Find and kill the process
ps aux | grep python
kill <PID>

# Or delete the database file
rm rag_wikipedia_mini.db
rm .rag_wikipedia_mini.db.lock
```

**3. Out of memory errors**

**Solutions**:
- Reduce batch size in embeddings (edit `src/utils.py:136`)
- Use smaller sample for RAGAs (edit `EVAL_SAMPLE_SIZE` in `run_ragas_evaluation.py`)
- Close other applications

**4. RAGAs evaluation fails**

```
OpenAI API error / No API key found
```

**Solution**: Check `.env` file

```bash
cat .env  # Should show OPENAI_API_KEY=...
```

If missing:

```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

**5. Slow performance**

- **Embeddings**: Bottleneck is CPU inference (~1-5 it/s normal)
- **LLM Generation**: FLAN-T5-small is optimized for CPU
- **Cross-Encoder**: Slowest component (~30-60s per 100 questions)

**Speed ups**:
- Use GPU if available (edit `device='cuda'` in code)
- Reduce test set size for iteration
- Use smaller embedding batches

---




Expected output:
- Query Rewriting Test Passed
- Reranking Test Passed
- Full Pipeline Test Completed

---

## Performance Benchmarks

### System Requirements

- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Optimal**: 16GB RAM, GPU (CUDA/MPS)

### Runtime Benchmarks (CPU, M1 Mac)

| Task | Dataset Size | Runtime |
|------|--------------|---------|
| Data preprocessing | 3026 docs | ~10s |
| Embedding generation | 3026 docs | ~25s |
| Milvus indexing | 3026 docs | ~5s |
| Question answering | 917 questions | ~45-60 min |
| Full naive experiments | 12 configs | ~60-120 min |
| Advanced RAG experiments | 5 configs | ~90-150 min |
| RAGAs evaluation | 100 questions | ~15-30 min |

---


### Reproducibility Standards

- All experiments use fixed random seeds where applicable
- Dependencies locked in `requirements.txt`
- Configuration parameters documented
- Results saved to CSV for verification

---

## License & Attribution

**Dataset**: [RAG-mini-Wikipedia](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) (HuggingFace)
**Models**:
- sentence-transformers (Apache 2.0)
- FLAN-T5 (Apache 2.0)
- cross-encoder/ms-marco-MiniLM-L-6-v2 (Apache 2.0)

**RAGAs Framework**: [explodinggradients/ragas](https://github.com/explodinggradients/ragas)

---

## Contact & Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review documentation in `docs/`
3. Check code comments and docstrings

**Last Updated**: October 2025
**Version**: 1.0
