import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# 1. Load Environment Variables (.env)
load_dotenv()

# 2. Inisialisasi ChatModel menggunakan OpenRouter
# Kamu bisa pakai "meta-llama/llama-3.1-70b-instruct" atau "openai/gpt-4o-mini"
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-70b-instruct", 
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=1024
)

print("=== 1. Simple Prompt Template ===")
# Membuat template prompt sederhana
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in {language}."),
    ("human", "{question}"),
])

# Formatting prompt menjadi list messages (untuk dicek)
msgs = prompt.format_messages(
    role="Python tutor",
    language="Indonesian",
    question="Apa itu generator dalam Python?",
)

print("Formatted Messages:")
print(msgs)

# Menjalankan pemanggilan ke LLM
response = llm.invoke(msgs)
print("\nLLM Response:")
print(response.content)

print("\n" + "="*40 + "\n")

print("=== 2. Prompt Template dengan MessagesPlaceholder (History) ===")
# Template prompt yang menyertakan riwayat percakapan (chat history)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Today's context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

formatted_chat = chat_prompt.format_messages(
    context="User is learning about RAG",
    history=[
        HumanMessage(content="What is retrieval?"),
    ],
    input="How is it different from fine-tuning?",
)

print(f"Total messages in formatted prompt: {len(formatted_chat)}")

# Menjalankan pemanggilan ke LLM
chat_response = llm.invoke(formatted_chat)
print("\nLLM Response:")
print(chat_response.content)