import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load Environment Variables (.env)
load_dotenv()

# 2. Inisialisasi LLM via OpenRouter
# Kamu bisa memilih model meta-llama/llama-3.1-70b-instruct atau openai/gpt-4o-mini
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-70b-instruct", # Atau "openai/gpt-4o-mini"
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7
)

# 3. Buat / Muat Dokumen Mentah (Sesuai Poin 9.6 PDF)
raw_docs = [
    Document(
        page_content="""LangChain provides a standard interface for interacting with
large language models and related tools. It simplifies the development of
context-aware reasoning applications by combining prompts, models, memory,
and retrieval in composable pipelines. LCEL makes these pipelines
declarative and easy to debug.""",
        metadata={"source": "langchain_docs.txt", "page": 1}
    ),
    Document(
        page_content="""Retrieval-Augmented Generation (RAG) is a pattern that combines
a retriever with a generator. The retriever finds relevant documents from a
knowledge base. Those documents are injected into the generator's context window.
This allows models to answer questions about information they were never trained on.""",
        metadata={"source": "rag_guide.txt", "page": 1}
    ),
]

# 4. Split Dokumen Menggunakan RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(raw_docs)

print(f"=== HASIL TEXT SPLITTING ===")
print(f"Original docs: {len(raw_docs)} | Generated chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i} ({len(chunk.page_content)} chars) | Source: {chunk.metadata['source']}")
    print(f"Content: {chunk.page_content}\n" + "-"*40)

# 5. Opsional: Mengirimkan salah satu Chunk ke LLM OpenRouter untuk Dirangkum
print("\n=== MENGIRIM CHUNK KE MODEL OPENROUTER ===")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Rangkum teks berikut dalam 1 kalimat singkat Bahasa Indonesia."),
    ("human", "{text}")
])

chain = prompt | llm | StrOutputParser()

# Rangkum chunk pertama
sample_summary = chain.invoke({"text": chunks[0].page_content})
print("Rangkuman Chunk 0:")
print(sample_summary)