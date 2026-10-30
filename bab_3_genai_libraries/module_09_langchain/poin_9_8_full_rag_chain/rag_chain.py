import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load Environment Variables (OpenRouter)
load_dotenv()

# Gunakan key OpenRouter dari file .env
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
    "OPENAI_API_KEY"
)
openrouter_base_url = "https://openrouter.ai/api/v1"

# 2. Inisialisasi Embeddings via OpenRouter
# Catatan: Untuk embeddings di OpenRouter/OpenAI kompatibel, kita pakai text-embedding-3-small atau sejenisnya
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openrouter_api_key,
    openai_api_base=openrouter_base_url,
)

# 3. Inisialisasi Chat Model (Pilih salah satu model OpenRouter kamu)
# Contoh menggunakan Llama 3.1 70B Instruct atau GPT-4o Mini
MODEL_NAME = "meta-llama/llama-3.1-70b-instruct"
# MODEL_NAME = "openai/gpt-4o-mini"

llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=openrouter_api_key,
    openai_api_base=openrouter_base_url,
    temperature=0.7,
)

# 4. Dummy Documents & Chunking
docs = [
    Document(
        page_content="RAG stands for Retrieval-Augmented Generation. It connects LLMs to external knowledge bases."
    ),
    Document(
        page_content="The retrieval step uses embeddings and vector search to find relevant documents for a given query."
    ),
    Document(
        page_content="The generation step takes the retrieved context and the user question to produce a grounded answer."
    ),
    Document(
        page_content="Vector databases like ChromaDB, Pinecone, and Weaviate store embeddings for fast similarity search."
    ),
    Document(
        page_content="Fine-tuning bakes knowledge into model weights. RAG retrieves knowledge at inference time. Both have trade-offs."
    ),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents(docs)

# 5. Embed & Store in Chroma Vectorstore (In-Memory)
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. RAG Prompt Template
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain the answer, say "I don't have that information."

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


# 7. Build Full RAG Chain menggunakan LCEL (LangChain Expression Language)
parser = StrOutputParser()

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | parser
)

# 8. Testing Query
if __name__ == "__main__":
    questions = [
        "What does RAG stand for?",
        "How is RAG different from fine-tuning?",
        "What is the capital of Indonesia?",  # Tidak ada di konteks
    ]

    for q in questions:
        print(f"Q: {q}")
        print(f"A: {rag_chain.invoke(q)}\n")