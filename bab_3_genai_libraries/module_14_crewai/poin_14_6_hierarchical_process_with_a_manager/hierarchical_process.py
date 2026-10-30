import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM

# 1. Load Environment Variables dari file .env
load_dotenv()

# 2. Inisialisasi LLM via OpenRouter menggunakan LLM bawaan CrewAI
llm = LLM(
    model="openrouter/meta-llama/llama-3.1-70b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# 3. Inisialisasi Manager Agent
manager = Agent(
    role="AI Content Production Manager",
    goal="Coordinate a team to produce high-quality AI content efficiently.",
    backstory="""You are an experienced content production manager who excels at
    breaking down projects, assigning work to the right specialist, and ensuring
    quality standards are met before publication.""",
    llm=llm,
    allow_delegation=True,
    verbose=True,
)

# 4. Inisialisasi Worker Agents
researcher = Agent(
    role="AI Research Analyst",
    goal="Research AI topics thoroughly.",
    backstory="Senior researcher specialising in LLM systems and RAG.",
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Write developer-focused technical content.",
    backstory="Technical writer with Python and ML experience.",
    llm=llm,
    verbose=True,
)

# 5. Inisialisasi Main Task
main_task = Task(
    description="""Produce a complete technical brief on 'Vector Databases for RAG in Production'.
    Include: what they are, how to choose one, and three implementation best practices.""",
    expected_output="A structured technical brief in Markdown, 400-500 words.",
    agent=manager,
)

# 6. Inisialisasi Crew dengan Process Hierarchical
crew = Crew(
    agents=[researcher, writer],  # Manager tidak boleh masuk di list ini
    tasks=[main_task],
    process=Process.hierarchical,
    manager_agent=manager,
    verbose=True,
)

# 7. Eksekusi Crew
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n=== HASIL AKHIR HIERARCHICAL PROCESS ===")
    print(result)