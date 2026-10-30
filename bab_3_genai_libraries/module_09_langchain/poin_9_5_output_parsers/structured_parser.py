import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Muat environment variables dari file .env
load_dotenv()

# 2. Inisialisasi ChatOpenAI yang dihubungkan ke OpenRouter
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-70b-instruct",  # Atau "openai/gpt-4o-mini"
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
)

# 3. Definisikan Pydantic Schema untuk Structured Output
class ModelComparison(BaseModel):
    models: List[str] = Field(description="Daftar nama model yang dibandingkan")
    best_for_cost: str = Field(description="Model yang paling hemat biaya")
    best_for_quality: str = Field(description="Model dengan kualitas tertinggi")
    recommendation: str = Field(description="Rekomendasi keseluruhan secara singkat")

# 4. Hubungkan LLM dengan skema output terstruktur
structured_llm = llm.with_structured_output(ModelComparison)

# 5. Buat ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "Anda adalah seorang pakar/analis perbandingan model AI."),
    ("human", "Bandingkan model-model berikut untuk sebuah startup dengan anggaran terbatas: {models}"),
])

# 6. Susun LCEL Chain (Prompt | Structured LLM)
chain = prompt | structured_llm

# 7. Eksekusi (Invoke) Chain
if __name__ == "__main__":
    print("Sedang memproses evaluasi model via OpenRouter...\n")
    
    result: ModelComparison = chain.invoke({
        "models": "gpt-4o, gpt-4o-mini, claude-3.5-sonnet"
    })

    # Output hasil yang sudah terparsing dalam bentuk objek Pydantic
    print(f"Model Terbanding : {result.models}")
    print(f"Terbaik untuk Biaya: {result.best_for_cost}")
    print(f"Terbaik untuk Kualitas: {result.best_for_quality}")
    print(f"Rekomendasi       : {result.recommendation}")