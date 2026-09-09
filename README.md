# NHS Performance RAG Agent

**Tools:** Python · FAISS · Sentence Transformers · Groq (Llama 3.1 8B Instant) · Streamlit  
**Data:** NHS England A&E and RTT Waiting List — January 2025  
**Domain:** Healthcare analytics · Natural language querying over structured NHS data  

---

## Overview

A retrieval-augmented generation (RAG) system for natural language lookup of NHS England performance records. Users can ask questions about A&E breach rates, RTT waiting times, and trust-level performance without manually searching spreadsheets. The system retrieves semantically relevant records and provides them as context to an LLM for grounded answer generation.

The project demonstrates a full NLP pipeline: structured data ingestion, text chunk generation, semantic embedding, FAISS vector indexing, and LLM-powered answer generation via the Groq API.

---

## Architecture
```
NHS CSV Data
     │
     ▼
Data Processor          ← Converts rows to natural language chunks
     │
     ▼
Sentence Transformer    ← Embeds chunks (all-MiniLM-L6-v2)
     │
     ▼
FAISS Vector Index      ← Cosine-similarity retrieval over embedded records
     │
     ▼
Groq LLM (Llama 3.1 8B Instant)      ← Generates answer from retrieved context
     │
     ▼
Streamlit UI            ← User query interface
```

---

## Data

| Source | Description | Chunks |
|---|---|---|
| NHS A&E Monthly — Jan 2025 | Trust-level attendances, 4hr breaches, 12hr waits | 124 |
| NHS RTT Waiting List — Jan 2025 | Incomplete pathways by provider and treatment function | 4,130 |
| **Total** | | **4,254** |

Each row is converted to a natural language chunk — for example:

> *"In January 2025, CALDERDALE AND HUDDERSFIELD NHS FOUNDATION TRUST (code: RWY) recorded 14,485 Type 1 A&E attendances. 4,954 patients waited over 4 hours (34.2% breach rate). 33 patients waited 12 or more hours from decision to admit."*

This representation enables semantic retrieval of records relevant to trust- and metric-specific queries.

---

## Repository Structure
```
├── src/
│   ├── data_processor.py      # Converts NHS CSVs to text chunks
│   ├── vector_store.py        # Builds and saves FAISS index
│   └── rag_pipeline.py        # Retrieval + Groq LLM answer generation
├── app.py                     # Streamlit query interface
├── data/
│   ├── chunks.json            # Generated text chunks
│   ├── faiss.index            # FAISS vector index
│   └── metadata.json          # Chunk metadata for retrieval
├── .env.example               # Environment variable template
├── .gitignore                 # Excludes .env and data files
└── README.md
```

---

## Example Queries

| Query | Example response |
|---|---|
| What was the A&E breach rate for Calderdale and Huddersfield in January 2025? | Reports the trust's Type 1 A&E breach rate from the retrieved record |
| How many patients at Mid Cheshire Hospitals were waiting over 52 weeks for Cardiology? | Reports the over-52-week waiting count from the relevant RTT record |
| What was the RTT position for Cardiology at a specific NHS trust? | Retrieves the relevant provider and treatment-function record |

---

## Technical Stack

| Component | Tool |
|---|---|
| Data processing | Python · pandas |
| Text embedding | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | FAISS (faiss-cpu) |
| LLM | Groq API · Llama 3.1 8B Instant |
| UI | Streamlit |
| Environment | python-dotenv |

---

## How to Run

**Prerequisites:** Python 3.10+, a free [Groq API key](https://console.groq.com), and the NHS data files.
```bash
# 1. Clone the repository
git clone https://github.com/amdrwn/nhs-rag-agent
cd nhs-rag-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the environment template and add your Groq API key
cp .env.example .env

# 4. Add NHS data files to data/ directory
# Download from NHS England open data portal:
# - A&E: https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/
# - RTT: https://www.england.nhs.uk/statistics/statistical-work-areas/rtt-waiting-times/

# 5. Process data and build vector store
python src/data_processor.py
python src/vector_store.py

# 6. Run the app
python -m streamlit run app.py
```

**Note:** The NHS data files are not included in this repository. Download them from the NHS England open data portal using the links above.

---

## Limitations

- Data covers January 2025 only; queries about other periods are unsupported.
- Retrieval operates over individual trust/treatment records rather than executing structured queries over the full dataset.
- Aggregate, ranking, and comparison questions (e.g. national totals or "which trust had the highest rate?") cannot be answered reliably using top-k semantic retrieval alone.
- Answer quality depends on the relevant record being retrieved and included in the LLM context.
- Generated answers should be checked against the displayed source chunks for high-stakes use.

## Future Improvements

- Add structured query/aggregation support for ranking, comparison and national-level questions
- Add support for multiple months and time-series analysis
- Implement retrieval evaluation metrics such as precision@k and recall@k
- Explore hybrid semantic and metadata-based retrieval
- Add caching for faster query responses
