import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 1. Load environment variables dari .env
load_dotenv()

# Mengambil API key dari OPENROUTER_API_KEY atau OPENAI_API_KEY di .env
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

# 2. Inisialisasi ChatOpenAI yang diarahkan ke OpenRouter
# Kamu bisa pakai model: "meta-llama/llama-3.1-70b-instruct" atau "openai/gpt-4o-mini"
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-70b-instruct",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=1024,
)

# 3. Menyiapkan pesan (SystemMessage & HumanMessage)
messages = [
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is LangChain Expression Language (LCEL)?"),
]

print("=== 1. Invoke (Blocking) ===")
response = llm.invoke(messages)
print(response.content)
print("\n" + "=" * 40 + "\n")

print("=== 2. Stream (Token by Token) ===")
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
print("\n\n" + "=" * 40 + "\n")

print("=== 3. Batch (Parallel Inputs) ===")
batch_responses = llm.batch(
    [
        [HumanMessage(content="What is RAG in 1 sentence?")],
        [HumanMessage(content="What is a vector database in 1 sentence?")],
    ]
)

for idx, r in enumerate(batch_responses, 1):
    print(f"Response {idx}: {r.content}")