import os
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. Load Environment Variables (.env)
load_dotenv()

# 2. Inisialisasi LLM via OpenRouter
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-70b-instruct",  # Atau "openai/gpt-4o-mini"
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7
)

parser = StrOutputParser()

# 3. Buat Prompt Template dengan MessagesPlaceholder untuk Memory
prompt = ChatPromptTemplate.from_messages([
    ("system", "Kamu adalah asisten AI yang ramah dan membantu. Jawab dengan singkat dan jelas dalam Bahasa Indonesia."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | parser

# 4. In-Memory Session Store
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 5. Bungkus Chain dengan Memory Management
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "user-session-1"}}

# --- DEMO PERCAKAPAN BERTAHAP ---

# Turn 1: Mengenalkan diri
print("=== TURN 1 ===")
q1 = "Halo, nama aku Mayzendhy. Aku lagi belajar modul LangChain."
print(f"User : {q1}")
r1 = chain_with_history.invoke({"input": q1}, config=config)
print(f"AI   : {r1}\n")

# Turn 2: Menguji memori model tentang nama penggunanya
print("=== TURN 2 ===")
q2 = "Siapa nama aku dan apa yang lagi aku pelajari?"
print(f"User : {q2}")
r2 = chain_with_history.invoke({"input": q2}, config=config)
print(f"AI   : {r2}\n")

# Inspect histori pesan yang tersimpan di sesi
history = get_session_history("user-session-1")
print(f"=== STATISTIK MEMORI ===")
print(f"Total pesan tersimpan dalam memori: {len(history.messages)}")