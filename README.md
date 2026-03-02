# Energy-Aware Requirements Dependency Classification

A comprehensive system for classifying and analyzing dependencies between software requirements while measuring energy consumption and latency metrics of the classification pipeline.

## Project Overview

This project implements an intelligent requirements dependency classification system that leverages:
- **Large Language Models (LLMs)** for requirement classification
- **Vector Databases** (Milvus) for semantic similarity-based retrieval
- **Graph Databases** (Neo4j) for entity-based candidate retrieval
- **Energy Tracking** (CodeCarbon) to measure carbon emissions and power consumption
- **Multiple Datasets** from various software projects

The system aims to identify relationships between requirements (e.g., dependencies, conflicts) while being energy-aware and optimizing for computational efficiency.

## Repository Structure

```
Energy-aware-Requirements-Dependency-Classification/
├── App/                              # Main application code
│   ├── engine.py                    # Core classification engine and orchestration
│   ├── llm.py                       # LLM client and interactions
│   ├── graphdb.py                   # Neo4j graph database retriever
│   ├── vectordb.py                  # Milvus vector database retriever
│   ├── prompt.py                    # Prompt templates (zero-shot, few-shot, CoT)
│   ├── utils.py                     # Utility functions
│   ├── retrievalEval_energy.py      # Energy consumption evaluation for retrieval
│   ├── retrievalEval_Recall.py      # Recall metrics evaluation for retrieval
│   ├── experiment.ipynb             # Main experiment notebook
│   └── __init__.py
├── Data/                            # Datasets
│   ├── cdn_pairs.csv                # CDN project requirement pairs
│   ├── open_coss_clean_pairs.csv    # OpenCOSS project requirement pairs
│   ├── pure_clean_pairs.csv         # Pure project requirement pairs
│   ├── uav_clean_pairs.csv          # UAV project requirement pairs
│   ├── world_vista_clean_pairs.csv  # World Vista project requirement pairs
│   └── conflictrequirements/        # Conflict requirement datasets
│       ├── cdn_creq.csv
│       ├── open_coss_creq.csv
│       ├── pure_creq.csv
│       ├── uav_creq.csv
│       └── world_vista_creq.csv
├── dataModeling/                    # Data processing and modeling
│   ├── main.ipynb                   # Data processing notebook
│   ├── unique_requirements_with_entities.csv     # Requirements with extracted entities
│   ├── improved_unique_requirements_with_entities.csv
│   ├── original_unique_requirements_with_entities.csv
│   ├── entity_extraction_energy_latency.csv      # Energy/latency for entity extraction
│   └── emissions.csv                # Carbon emissions data
├── scripts/                         # Analysis and evaluation scripts
│   ├── evaluation.ipynb             # General evaluation notebook
│   ├── modelcamparison.ipynb        # Model comparison analysis
│   ├── RetrievalEnergyPipeline_evaluation.ipynb # Detailed energy analysis
│   ├── retrievalpipeline_evaluation_recall@k.ipynb # Recall@K analysis
│   └── update_result_with_groundTruth.ipynb     # Ground truth update script
├── results/                         # Output results
│   └── Result.csv                   # Classification results
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Key Components

### 1. **Engine (engine.py)**
- Orchestrates the classification pipeline
- Integrates retrieval mechanisms (vector and graph-based)
- Parses LLM responses
- Tracks energy consumption during inference
- Supports different prompting strategies: zero-shot, few-shot, chain-of-thought (CoT)

### 2. **LLM Client (llm.py)**
- Interfaces with local LLM endpoints (default: Mistral-7B-Instruct)
- Measures energy consumption using CodeCarbon
- Tracks latency and power consumption metrics
- Supports configurable temperature for generation

### 3. **Vector Database Retriever (vectordb.py)**
- Uses Milvus for semantic similarity search
- Employs SentenceTransformer models for embedding
- Returns top-k similar requirements as candidates
- Enables efficient semantic retrieval at scale

### 4. **Graph Database Retriever (graphdb.py)**
- Neo4j-based entity-aware retrieval
- Extracts candidates through entity matching (actors, actions, objects, attributes)
- Finds requirements sharing similar semantic entities
- Provides structured, interpretable retrieval paths

### 5. **Data Processing (dataModeling/)**
- Entity extraction from requirements (actors, actions, objects, conditions)
- Named entity recognition using spaCy
- Creation of structured requirement datasets
- Energy and latency tracking for preprocessing steps


## Installation

### Prerequisites
- Python 3.8+
- Neo4j instance running (for graph-based retrieval)
- Milvus instance running (for vector-based retrieval)
- Local LLM endpoint (e.g., LM Studio with Mistral-7B-Instruct)

### Setup


2. **Create a virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model** (required for entity extraction)
```bash
python -m spacy download en_core_web_sm
```

## Configuration

Environment variables required:
- `NEO4J_URI`: Neo4j connection URI (e.g., `bolt://localhost:7687`)
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `ZILLIZ_URI`: Milvus/Zilliz connection URI (optional, for cloud Milvus)
- `ZILLIZ_TOKEN`: Milvus/Zilliz authentication token (optional)

Local Milvus defaults:
- Host: `localhost`
- Port: `19530`

LLM endpoint defaults:
- URL: `http://localhost:1234/v1/chat/completions`
- Model: `mistral-7b-instruct`

## Usage

### Running the Main Experiment
Open and execute the Jupyter notebook:
```bash
jupyter notebook App/experiment.ipynb
```

### Data Modeling & Entity Extraction
Process requirements and extract entities:
```bash
jupyter notebook dataModeling/main.ipynb
```

### Evaluation Scripts
- **Energy Analysis**: `scripts/RetrievalEnergyPipeline_evaluation.ipynb`
- **Recall Metrics**: `scripts/retrievalpipeline_evaluation_recall@k.ipynb`
- **Model Comparison**: `scripts/modelcamparison.ipynb`
- **General Evaluation**: `scripts/evaluation.ipynb`

### Programmatic Usage Example

```python
from App.engine import Engine
from App.llm import LLMClient
from App.graphdb import Neo4jRetriever
from App.vectordb import MilvusRetriever

# Initialize components
llm = LLMClient(model="mistral-7b-instruct", temperature=0.0)
graph_retriever = Neo4jRetriever(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password"
)
vector_retriever = MilvusRetriever(
    host="localhost",
    port="19530",
    collection_name="requirements",
    model_name="all-MiniLM-L6-v2"
)

# Create engine
engine = Engine(
    llm_client=llm,
    graph_retriever=graph_retriever,
    vector_retriever=vector_retriever
)

# Classify requirement pair
result = engine.classify_pair(
    text1="Requirement A",
    text2="Requirement B",
    source="project_name"
)
```

## Dependencies

Key libraries included:
- **codecarbon**: Carbon emissions and energy tracking
- **neo4j**: Graph database connectivity
- **pymilvus**: Vector database connectivity
- **sentence-transformers**: Semantic embeddings
- **spacy**: Named entity recognition and linguistic processing
- **pandas**: Data manipulation and analysis
- **requests**: HTTP client for LLM API calls
- **transformers**: Hugging Face transformers library
- **jupyter**: Interactive notebook environment

See `requirements.txt` for the complete dependency list.

## Datasets

The project includes requirement pairs from 5 different software projects:

| Dataset | File | Description |
|---------|------|-------------|
| **CDN** | `cdn_pairs.csv` | Content Delivery Network requirements |
| **OpenCOSS** | `open_coss_clean_pairs.csv` | Open source software requirements |
| **Pure** | `pure_clean_pairs.csv` | Pure software project requirements |
| **UAV** | `uav_clean_pairs.csv` | Unmanned Aerial Vehicle system requirements |
| **World Vista** | `world_vista_clean_pairs.csv` | World Vista project requirements |

Additionally, conflict requirement datasets are available in `Data/conflictrequirements/` for specialized analysis.

## Results and Outputs

Results are saved to:
- `results/Result.csv` - Classification results with labels and confidence scores
- `dataModeling/emissions.csv` - Energy consumption metrics
- `dataModeling/entity_extraction_energy_latency.csv` - Extraction process metrics

## Citation

If you use this project in your research, please cite:

```bibtex
@repository{shreyaspatil2024energy,
  title={Energy-aware Requirements Dependency Classification},
  author={Patil, Shreyas},
  year={2024},
  url={https://github.com/ShreyasP20/Energy-aware-Requirements-Dependency-Classification}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests to improve the project.

## Acknowledgments

- CodeCarbon for energy tracking capabilities
- Neo4j and Milvus communities for excellent database solutions
- Hugging Face for transformer models and the transformers library
- The requirements engineering research community
