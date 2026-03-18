# NHS Performance RAG Agent

**Tools:** Python · FAISS · Sentence Transformers · Groq (Llama 3) · Streamlit
**Data:** NHS England A&E and RTT Waiting List — January 2025  
**Domain:** Healthcare analytics · Natural language querying over structured NHS data  

---

## Overview

A retrieval-augmented generation (RAG) system that enables natural language querying over NHS England performance data. Users can ask questions about A&E breach rates, RTT waiting times, and trust-level performance without requiring SQL or manual exploration of spreadsheets, and the system retrieves the most relevant data chunks and generates a grounded, cited answer. This approach enables faster, more accessible analysis of NHS performance data, reducing reliance on manual reporting and enabling non-technical users to explore system-wide bottlenecks.

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
FAISS Vector Index      ← Stores and retrieves by semantic similarity
     │
     ▼
Groq LLM (Llama 3)      ← Generates answer from retrieved context
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

This format allows the embedding model to match semantic queries to specific trusts and metrics.

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

| Query | Answer |
|---|---|
| Which trust had the highest A&E breach rate in January 2025? | EAST SUFFOLK AND NORTH ESSEX NHS FOUNDATION TRUST (code: RDE) with a breach rate of 52.7% |
| How many patients were waiting over 52 weeks for Cardiology? | 29 patients at The Queen Elizabeth Hospital, King's Lynn; 552 at Mid Cheshire Hospitals NHS Foundation Trust |
| Who are you? | I am an NHS data analyst |

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

- Data covers January 2025 only — queries about other periods will not return results
- The system retrieves the top 5 most semantically similar chunks; complex aggregate queries (e.g. national totals) may return partial results
- Answer quality depends on whether the relevant trust/metric appears in the top 5 retrieved chunks

## Future Improvements
- Add support for multi-month or time-series analysis
- Implement retrieval evaluation metrics (e.g. precision@k)
- Improve chunking with aggregation at trust or regional level
- Add caching for faster query responses