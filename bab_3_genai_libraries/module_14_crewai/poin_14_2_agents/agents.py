import os
from crewai import Agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables dari file .env
load_dotenv()

# 1. Inisialisasi LLM via OpenRouter
# Kamu bisa mengganti model="meta-llama/llama-3.1-70b-instruct" atau "openai/gpt-4o-mini"
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-70b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# 2. Definisikan Agent - AI Research Analyst
researcher = Agent(
    role="AI Research Analyst",
    goal="Mencari informasi yang akurat dan terbaru mengenai topik Generative AI.",
    backstory="""Kamu adalah seorang AI Research Analyst senior berpengalaman 10 tahun 
    yang ahli dalam membaca dan merangkum jurnal akademik, blog teknis, dan catatan rilis AI. 
    Kamu terkenal sangat presisi dan selalu menyertakan sumber.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5
)

# 3. Definisikan Agent - Technical Content Writer
writer = Agent(
    role="Technical Content Writer",
    goal="Menulis konten teknis yang jelas, akurat, dan menarik untuk developer.",
    backstory="""Kamu adalah seorang penulis teknis dengan latar belakang software engineering. 
    Kamu sangat ahli dalam mengubah konsep AI yang rumit menjadi panduan praktis yang terstruktur rapi.""",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# 4. Definisikan Agent - Content Quality Reviewer
reviewer = Agent(
    role="Content Quality Reviewer",
    goal="Memastikan konten akurat, lengkap, dan siap dipublikasikan.",
    backstory="""Kamu adalah editor teliti yang mengulas konten teknis untuk memastikan 
    keakuratan, kejelasan, dan kelengkapan materi sebelum rilis.""",
    llm=llm,
    verbose=True,
    allow_delegation=True
)

if __name__ == "__main__":
    print("--- Agen CrewAI Berhasil Dibuat ---")
    print(f"1. Role: {researcher.role} | Model: {llm.model_name}")
    print(f"2. Role: {writer.role} | Model: {llm.model_name}")
    print(f"3. Role: {reviewer.role} | Model: {llm.model_name}")