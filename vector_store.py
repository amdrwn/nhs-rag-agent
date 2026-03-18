import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def build_vector_store(chunks_path: str, index_path: str, metadata_path: str):
    with open(chunks_path, "r") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]
    metadata = [chunk["metadata"] for chunk in chunks]

    print(f"Encoding {len(texts)} chunks...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    with open(metadata_path, "w") as f:
        json.dump({"texts": texts, "metadata": metadata}, f)

    print(f"Vector store built: {index.ntotal} vectors")
    print(f"Index saved to {index_path}")

if __name__ == "__main__":
    build_vector_store(
        chunks_path="data/chunks.json",
        index_path="data/faiss.index",
        metadata_path="data/metadata.json"
    )
