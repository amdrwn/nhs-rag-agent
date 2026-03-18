import streamlit as st
from src.rag_pipeline import NHSRagAgent

st.set_page_config(
    page_title="NHS RAG Agent",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 NHS Performance RAG Agent")
st.markdown("Ask questions about NHS A&E and RTT waiting list performance in January 2025.")

@st.cache_resource
def load_agent():
    return NHSRagAgent(
        index_path="data/faiss.index",
        metadata_path="data/metadata.json"
    )

agent = load_agent()

query = st.text_input("Ask a question:", placeholder="e.g. Which trust had the highest A&E breach rate?")

if query:
    with st.spinner("Searching..."):
        result = agent.answer(query)

    st.markdown("### Answer")
    st.write(result["answer"])

    st.markdown("### Source Chunks")
    for i, source in enumerate(result["sources"]):
        with st.expander(f"Source {i+1} — {source['metadata']['org_name']}"):
            st.write(source["text"])
            st.json(source["metadata"])
