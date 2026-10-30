import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# 1. Load Environment Variables (.env)
load_dotenv()

# 2. Inisialisasi ChatModel mengarah ke OpenRouter
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-70b-instruct",
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=1024
)

parser = StrOutputParser()

# 3. Prompt khusus Kode
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Python expert. Provide a complete, working code example."),
    ("human", "{input}"),
])

# 4. Prompt khusus Teori / Penjelasan
prose_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a clear technical writer. Explain conceptually."),
    ("human", "{input}"),
])

code_chain = code_prompt | llm | parser
prose_chain = prose_prompt | llm | parser

# 5. Logika Routing
def route(input_dict: dict) -> str:
    """Mengarahkan ke code_chain atau prose_chain berdasarkan kata kunci."""
    q = input_dict["input"].lower()
    if any(w in q for w in ["write", "code", "implement", "example", "function"]):
        return "code"
    return "prose"

def run_routed(input_dict: dict) -> str:
    choice = route(input_dict)
    if choice == "code":
        return code_chain.invoke(input_dict)
    return prose_chain.invoke(input_dict)

router_chain = RunnableLambda(run_routed)

# 6. Pengujian Router Chain
print("=== Output Request Kode ===")
print(router_chain.invoke({"input": "Write a function to chunk text with overlap."}))

print("\n" + "="*40 + "\n")

print("=== Output Request Konsep ===")
print(router_chain.invoke({"input": "What is attention in transformers?"}))