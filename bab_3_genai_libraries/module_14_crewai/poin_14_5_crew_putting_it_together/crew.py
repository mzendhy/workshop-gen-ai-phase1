import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM

load_dotenv()

# Inisialisasi LLM langsung pakai CrewAI LLM (atau cukup passing string "gpt-4o")
llm = LLM(
    model="openrouter/meta-llama/llama-3.1-70b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Dummy / Placeholder Helper Functions
def make_tasks(researcher_agent, writer_agent, reviewer_agent):
    research_task = Task(
        description="Research the latest trends in AI context windows and capabilities.",
        expected_output="A summary report of AI context window trends.",
        agent=researcher_agent
    )
    writing_task = Task(
        description="Write a technical article based on the research findings.",
        expected_output="A well-structured technical article.",
        agent=writer_agent
    )
    review_task = Task(
        description="Review the article for accuracy, clarity, and tone.",
        expected_output="A final polished article ready for publication.",
        agent=reviewer_agent
    )
    return research_task, writing_task, review_task

# Recreate agents
researcher = Agent(
    role="AI Research Analyst",
    goal="Find accurate information on AI topics.",
    backstory="Senior AI research analyst with expertise in LLM systems.",
    llm=llm,
    verbose=False,
    max_iter=5,
)

writer = Agent(
    role="Technical Content Writer",
    goal="Write clear technical content for developers.",
    backstory="Technical writer with a software engineering background.",
    llm=llm,
    verbose=False,
    max_iter=5,
)

reviewer = Agent(
    role="Content Quality Reviewer",
    goal="Ensure content is accurate and publication-ready.",
    backstory="Meticulous editor who reviews technical AI content.",
    llm=llm,
    verbose=False,
    max_iter=5,
)

research_task, writing_task, review_task = make_tasks(researcher, writer, reviewer)

crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,
    verbose=False,
    memory=False,
)

# Kick off the crew
result = crew.kickoff()
print("=== CREW OUTPUT ===")
print(result)

# Access individual task outputs
print("\n=== TASK OUTPUTS ===")
for task in crew.tasks:
    print(f"\n--- {task.agent.role} ---")
    if task.output:
        print(str(task.output)[:200], "...")