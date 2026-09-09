import json
import faiss
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class NHSRagAgent:
    def __init__(self, index_path: str, metadata_path: str):
        print("Loading vector store...")
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, "r") as f:
            data = json.load(f)
        
        self.texts = data["texts"]
        self.metadata = data["metadata"]
        
        print("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("Connecting to Groq...")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        print("Ready.")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "score": float(scores[0][i])
            })
        return results

    def answer(self, query: str, top_k: int = 5) -> dict:
        chunks = self.retrieve(query, top_k)
        context = "\n\n".join([c["text"] for c in chunks])
        
        prompt = (
            "You are an NHS data analyst. Answer the question using only the "
            "context provided below. Do not use outside knowledge or infer values "
            "that are not present in the context. If the retrieved context is "
            "insufficient to answer the question, say so clearly. Be concise and specific.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        
        return {
            "query": query,
            "answer": response.choices[0].message.content,
            "sources": chunks
        }

if __name__ == "__main__":
    agent = NHSRagAgent(
        index_path="data/faiss.index",
        metadata_path="data/metadata.json"
    )
    
    test_queries = [
        "What was the A&E breach rate for Calderdale and Huddersfield in January 2025?",
        "How many patients at Mid Cheshire Hospitals were waiting over 52 weeks for Cardiology?",
    ]
    
    for query in test_queries:
        print(f"\nQ: {query}")
        result = agent.answer(query)
        print(f"A: {result['answer']}")
        print(f"Sources: {[s['metadata']['org_name'] for s in result['sources']]}")
