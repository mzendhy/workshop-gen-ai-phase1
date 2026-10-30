import os
from dotenv import load_dotenv

# Import module LlamaIndex
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. Load Environment Variables dari file .env
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 2. Konfigurasi Global Settings menggunakan OpenAILike
Settings.llm = OpenAILike(
    model="meta-llama/llama-3.1-70b-instruct",
    api_key=OPENROUTER_API_KEY,
    api_base="https://openrouter.ai/api/v1",
    is_chat_model=True,
    temperature=0.1,
    max_tokens=1024,
)

# Untuk embedding tetap gunakan OpenAIEmbedding dengan api_base OpenRouter
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    api_base="https://openrouter.ai/api/v1",
)

# 3. Menyiapkan Dokumen Sample
documents = [
    Document(text="HNSW (Hierarchical Navigable Small World) is the graph algorithm powering most modern vector search."),
    Document(text="IVF (Inverted File Index) partitions vectors into clusters for fast approximate search."),
    Document(text="PQ (Product Quantisation) compresses vectors to reduce memory usage at the cost of recall."),
    Document(text="Exact nearest-neighbour search guarantees correctness but is O(n) - unusable at scale."),
]

# 4. Membuat VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)

# 5. Membuat Retriever (top 2 dokumen)
retriever = VectorIndexRetriever(index=index, similarity_top_k=2)

# 6. Menjalankan Custom Query Pipeline
response_modes = [ResponseMode.COMPACT, ResponseMode.TREE_SUMMARIZE]

print("=== RUNNING LLAMAINDEX CUSTOM QUERY PIPELINE (OPENROUTER) ===\n")

for mode in response_modes:
    synthesizer = get_response_synthesizer(response_mode=mode)
    engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)
    
    query_str = "What algorithms are used in vector search?"
    resp = engine.query(query_str)
    
    print(f"--- Response Mode: {mode} ---")
    print(f"Result: {str(resp).strip()}\n")