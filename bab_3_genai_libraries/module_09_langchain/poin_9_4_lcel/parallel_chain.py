import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

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

# 3. Buat dua prompt yang berbeda
summarise_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise this in one sentence."),
    ("human", "{text}"),
])

keywords_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract 3 keywords from this text. Return them comma-separated."),
    ("human", "{text}"),
])

parser = StrOutputParser()

# 4. Eksekusi secara paralel memakai RunnableParallel
parallel_chain = RunnableParallel(
    summary=summarise_prompt | llm | parser,
    keywords=keywords_prompt | llm | parser,
)

# 5. Jalankan Parallel Chain
result = parallel_chain.invoke({
    "text": "Retrieval-Augmented Generation (RAG) improves LLM answers by fetching relevant documents at inference time."
})

print("=== Parallel Chain Result ===")
print("Summary :", result["summary"])
print("Keywords:", result["keywords"])