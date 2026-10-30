import os
from dotenv import load_dotenv
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. Load Environment Variables (.env)
load_dotenv()

# Mengambil API Key OpenRouter dari file .env
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY tidak ditemukan di file .env!")

# 2. Konfigurasi LLM & Embedding Global Menggunakan OpenRouter
# LlamaIndex menggunakan interface OpenAI dengan mengganti base_url ke OpenRouter
Settings.llm = OpenAI(
    model="meta-llama/llama-3.1-70b-instruct",  # atau "openai/gpt-4o-mini"
    api_key=openrouter_api_key,
    api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_tokens=1024,
)

# Untuk embedding, kita bisa menggunakan text-embedding-3-small via OpenRouter
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=openrouter_api_key,
    api_base="https://openrouter.ai/api/v1",
)

# 3. Membuat Document dari Teks
documents = [
    Document(
        text="""LlamaIndex is a data framework for LLM applications.
It provides tools for ingesting, structuring, and querying data
using large language models. The core abstraction is the Index,
which organises documents into a structure optimised for retrieval.""",
        metadata={"source": "llamaindex_overview", "category": "framework"},
    ),
    Document(
        text="""Vector store indexes store document embeddings in a vector database.
They support fast semantic search using approximate nearest-neighbour algorithms.
LlamaIndex integrates with ChromaDB, Pinecone, Weaviate, and many other backends.""",
        metadata={"source": "vector_store_guide", "category": "retrieval"},
    ),
    Document(
        text="""The query engine is the highest-level abstraction in LlamaIndex.
It takes a natural language question, retrieves relevant nodes,
and uses an LLM to synthesise a coherent answer from the retrieved context.""",
        metadata={"source": "query_engine_docs", "category": "querying"},
    ),
]

# 4. Parsing Documents menjadi Nodes (Chunks)
parser = SentenceSplitter(chunk_size=128, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(documents)

# 5. Output Hasil Node Parsing
print(f"Jumlah Documents: {len(documents)}, Jumlah Nodes: {len(nodes)}\n")
print("--- Detail Nodes ---")
for n in nodes:
    print(f"Node ID : {n.node_id[:8]}...")
    print(f"Length  : {len(n.text)} chars")
    print(f"Source  : {n.metadata.get('source')}")
    print(f"Text    : {n.text[:60]}...")
    print("-" * 40)