import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

# 3. Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise the following text in exactly {sentences} sentences."),
    ("human", "{text}"),
])

# 4. Membangun Basic Chain dengan LCEL (| operator)
chain = prompt | llm | StrOutputParser()

# 5. Jalankan Chain
result = chain.invoke({
    "sentences": 2,
    "text": """LangChain is a framework for developing applications powered by large language models.
    It provides abstractions for prompts, models, memory, and retrieval so developers can
    compose complex AI workflows from simple building blocks without writing boilerplate."""
})

print("=== Basic Chain Result ===")
print(result)