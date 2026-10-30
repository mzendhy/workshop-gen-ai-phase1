import os
from dotenv import load_dotenv
from llama_index.core import Document, SummaryIndex, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. Load environment variables dari file .env di root project
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY tidak ditemukan di .env!")

# 2. Global Settings: Konfigurasi LlamaIndex agar menggunakan OpenRouter API
Settings.llm = OpenAILike(
    model="meta-llama/llama-3.1-70b-instruct",  # Bisa juga diganti ke "openai/gpt-4o-mini"
    api_key=OPENROUTER_API_KEY,
    api_base="https://openrouter.ai/api/v1",
    is_chat_model=True,
    temperature=0.1,
    max_tokens=1024,
)

# Kebutuhan Embedding tetap diarahkan via OpenRouter
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    api_base="https://openrouter.ai/api/v1",
)

# 3. Simulasi Dokumen Panjang / Berbagai Bagian (Sesuai Materi Modul 10.6 PDF)
sections = [
    Document(
        text="Introduction: This report analyses the performance of five LLMs on coding, reasoning, and language tasks."
    ),
    Document(
        text="Section 1 - Coding: Claude Sonnet 4.5 scored highest on HumanEval with 94.2%. GPT-4o followed at 90.1%."
    ),
    Document(
        text="Section 2 - Reasoning: GPT-4o led on GSM8K with 97.1%. Claude Sonnet 4.5 achieved 96.8%."
    ),
    Document(
        text="Section 3 - Language: Both models performed similarly on MMLU, within 1% of each other."
    ),
    Document(
        text="Conclusion: For coding tasks, prefer Claude Sonnet 4.5. For balanced performance, GPT-4o is the safer choice."
    ),
]

# 4. Buat SummaryIndex dari dokumen
index = SummaryIndex.from_documents(sections)

# 5. Inisialisasi Query Engine
# Berbeda dengan VectorStoreIndex yang hanya mengambil top-k chunk terdekat,
# SummaryIndex membaca SEMUA node/chunk untuk melakukan rangkuman menyeluruh (document-wide synthesis).
query_engine = index.as_query_engine()

# 6. Eksekusi Query Rangkuman
response = query_engine.query("Which model should I use for a coding project?")

# 7. Tampilkan Hasil Output
print("=== Summary Index Query Response ===")
print(response)