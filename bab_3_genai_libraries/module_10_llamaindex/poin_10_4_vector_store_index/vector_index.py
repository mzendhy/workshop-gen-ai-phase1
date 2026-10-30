import os
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

# 1. Load Environment Variables
load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# 2. Konfigurasi LLM menggunakan OpenAILike
Settings.llm = OpenAILike(
    model="meta-llama/llama-3.1-70b-instruct",
    api_key=openrouter_api_key,
    api_base="https://openrouter.ai/api/v1",
    is_chat_model=True,
    context_window=128000,  # Menyediakan batas context window secara eksplisit
    temperature=0.1,
    max_tokens=1024,
)

# 3. Konfigurasi Embedding Model
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=openrouter_api_key,
    api_base="https://openrouter.ai/api/v1",
)

# 4. Buat Sample Dokumen
documents = [
    Document(
        text="RAG combines retrieval with generation to answer questions from external knowledge."
    ),
    Document(
        text="Fine-tuning trains a model on new data to improve performance on a specific task."
    ),
    Document(
        text="Prompt engineering optimises the input to an LLM to get better outputs without any training."
    ),
    Document(
        text="Vector databases store high-dimensional embeddings and support fast approximate nearest-neighbour search."
    ),
    Document(
        text="LangChain provides abstractions for building chains, agents, and memory-enabled LLM applications."
    ),
]

# 5. Build Vector Store Index
print("Sedang membuat VectorStoreIndex...")
index = VectorStoreIndex.from_documents(documents)

# 6. Buat Query Engine & Eksekusi Query
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query(
    "What is the difference between RAG and fine-tuning?"
)

# 7. Tampilkan Hasil
print("\n=== ANSWER ===")
print(response)

print("\n=== SOURCE NODES ===")
for node in response.source_nodes:
    print(f"- Score {node.score:.4f}: {node.text[:60]}...")