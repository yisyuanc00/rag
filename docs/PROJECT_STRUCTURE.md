# Project Structure

This document describes the organization of the RAG System Implementation project.

## Directory Layout

```
95820_hw2/
├── README.md                        # Main project documentation
├── requirements.txt                 # Python dependencies
├── .env                            # Environment variables (API keys)
├── .gitignore                      # Git exclusion rules
│
├── src/                            # Source code modules
│   ├── utils.py                    # Data preparation, Milvus setup, metrics
│   ├── naive_rag.py                # Baseline RAG implementation
│   ├── advanced_rag.py             # Enhanced RAG with query rewriting & reranking
│   ├── query_rewriting.py          # HyDE query rewriting module
│   ├── reranking.py                # Cross-encoder reranking module
│   ├── evaluation.py               # RAGAs evaluation framework
│   └── test.py                     # Unit tests
│
├── scripts/                        # Executable experiment scripts
│   ├── run_experiments.py          # Naive RAG experiments (12 configs)
│   ├── run_advanced_experiments.py # Advanced RAG experiments (12 configs)
│   └── ragas_evaluation.py         # RAGAs evaluation comparison
│
├── notebooks/                      # Jupyter notebooks for exploration
│   └── rag-Starter-Code.ipynb      # Initial development notebook
│
├── data/                           # Database files (gitignored)
│   ├── rag_wikipedia_mini.db       # Milvus Lite database (auto-generated)
│   └── *.db                        # Other database snapshots
│
├── results/                        # Experiment output files
│   ├── comparison_analysis.csv     # Naive RAG results (12 configs)
│   ├── advanced_rag_analysis.csv   # Advanced RAG results (12 configs)
│   ├── ragas_comparison.csv        # RAGAs metric comparison
│   ├── ragas_naive.csv             # Detailed naive RAG scores
│   └── ragas_advanced.csv          # Detailed advanced RAG scores
│
├── logs/                           # Execution logs (gitignored)
│   ├── run_advanced_experiments.log
│   ├── ragas_evaluation.log
│   └── run_experiments_100q.log
│
├── deliverables/                   # Final submission documents
│   ├── Final_Technical_Report.md   # Comprehensive technical report
│   └── ai_usage_log.md             # AI assistance documentation
│
└── docs/                           # Additional documentation
    └── PROJECT_STRUCTURE.md        # This file
```

## File Descriptions

### Root Directory

- **README.md**: Comprehensive project overview, setup instructions, usage guide
- **requirements.txt**: All Python dependencies with version pinning
- **.env**: API keys (OpenAI for RAGAs evaluation)
- **.gitignore**: Excludes generated files, databases, logs, virtual environments

### src/ - Core Modules

All core RAG system logic is modular and reusable:

- **utils.py** (265 lines):
  - Dataset loading and preprocessing
  - Milvus collection setup with PCA support
  - Retrieval functions
  - F1/EM metric calculation

- **naive_rag.py** (180 lines):
  - LLM pipeline initialization
  - Prompt templates (CLEAR_INSTRUCTION, FEW_SHOTS)
  - Answer generation logic
  - Experiment runner for baseline configurations

- **advanced_rag.py** (239 lines):
  - Extends naive RAG with enhancements
  - Integrates query rewriting and reranking
  - Advanced experiment runner with ablation support

- **query_rewriting.py** (111 lines):
  - HyDE (Hypothetical Document Embeddings) implementation
  - Query expansion method
  - LLM-based hypothesis generation

- **reranking.py** (165 lines):
  - Cross-encoder document reranking
  - MMR-based diversity reranking (optional)
  - Semantic relevance scoring

- **evaluation.py** (243 lines):
  - RAGAs framework integration
  - Metric computation (faithfulness, relevancy, precision, recall)
  - System comparison utilities

- **test.py** (93 lines):
  - Quick sanity tests for naive RAG
  - Single-query validation

### scripts/ - Execution Scripts

Standalone scripts for running experiments (should be run from project root):

- **run_experiments.py**:
  - Tests all 12 naive RAG configurations
  - Runtime: ~60-90 minutes on CPU
  - Generates comparison_analysis.csv

- **run_advanced_experiments.py**:
  - Tests all 12 advanced RAG configurations
  - Runtime: ~2-3 hours on CPU
  - Generates advanced_rag_analysis.csv

- **ragas_evaluation.py**:
  - RAGAs evaluation using OpenAI API
  - Compares naive vs advanced systems
  - Sample size: 100 questions (configurable)
  - Generates comparison metrics

### notebooks/ - Exploratory Analysis

- **rag-Starter-Code.ipynb**:
  - Initial prototype and exploration
  - Dataset analysis
  - Early experiments before code modularization

### data/ - Generated Databases

Contains Milvus Lite database files (SQLite-based):

- Auto-generated during experiments
- Excluded from git (can be large)
- Deleted and rebuilt between experiments

### results/ - Experiment Outputs

All CSV files with experiment results:

- **comparison_analysis.csv**: Naive RAG performance across 12 configs
- **advanced_rag_analysis.csv**: Advanced RAG performance across 12 configs
- **ragas_*.csv**: RAGAs evaluation metrics and detailed scores

### logs/ - Execution Logs

Terminal output from experiment runs:

- Useful for debugging
- Track experiment progress
- Excluded from git

### deliverables/ - Submission Documents

Final documents for assignment submission:

- **Final_Technical_Report.md**: Complete technical report (~4,800 words)
- **ai_usage_log.md**: Comprehensive AI tool usage documentation

### docs/ - Documentation

Additional project documentation:

- This structure guide
- Future: architecture diagrams, API docs, etc.

## Usage Patterns

### Running Experiments

```bash
# From project root (all scripts must be run from project root)
python scripts/run_experiments.py          # Naive RAG
python scripts/run_advanced_experiments.py # Advanced RAG
python scripts/ragas_evaluation.py         # RAGAs evaluation
```

### Importing Modules

```python
# From other scripts
from src.utils import prepare_experiment_data, setup_milvus_collection
from src.naive_rag import generate_answer, CLEAR_PROMPT
from src.advanced_rag import run_advanced_experiment
from src.query_rewriting import QueryRewriter
from src.reranking import DocumentReranker
from src.evaluation import evaluate_with_ragas
```

### Adding New Experiments

1. Create script in `scripts/`
2. Import necessary modules from `src/`
3. Save results to `results/`
4. Logs automatically go to `logs/`

## Design Principles

1. **Modularity**: Each component is independently testable
2. **Separation of Concerns**: Data prep, retrieval, generation, evaluation are separate
3. **Reproducibility**: Fixed seeds, deterministic generation, version pinning
4. **Clarity**: Clear naming, comprehensive docstrings, type hints
5. **Maintainability**: DRY principle, no code duplication

## File Sizes (Approximate)

```
Source Code:     ~2,500 lines (Python)
Documentation:   ~8,000 words (Markdown)
Results:         ~10 KB (CSV files)
Logs:            ~170 KB (text)
Database:        ~18 MB (Milvus Lite)
Notebook:        ~90 KB (ipynb)
```

## Dependencies

See `requirements.txt` for complete list. Key dependencies:

- PyTorch 2.5.1 (CPU)
- Transformers 4.56.2
- Sentence-Transformers 5.1.1
- PyMilvus 2.4+ with Milvus Lite
- RAGAs 0.3.5
- Pandas, NumPy, Scikit-learn

