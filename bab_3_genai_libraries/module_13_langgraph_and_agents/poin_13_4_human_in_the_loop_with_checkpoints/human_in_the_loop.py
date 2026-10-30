import os
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv

# Import dari LangGraph & LangChain
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load API Key dari file .env di root project
load_dotenv()

# ── 1. State Definition ───────────────────────────────────────────────────────
class ReviewState(TypedDict):
    plan: str
    approved: bool
    result: str
    messages: Annotated[list[str], operator.add]

# ── 2. Configuration for OpenRouter ──────────────────────────────────────────
openrouter_key = os.getenv("OPENROUTER_API_KEY")

# Menggunakan ChatOpenAI yang di-redirect ke OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",  # Bisa diganti ke "meta-llama/llama-3.1-70b-instruct"
    openai_api_key=openrouter_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7
)

# ── 3. Nodes Definition ───────────────────────────────────────────────────────
def plan_node(state: ReviewState) -> dict:
    """Node untuk membuat rencana/plan."""
    # LLM membuat rencana berdasarkan percakapan
    response = llm.invoke([
        SystemMessage(content="You are a planning assistant. Create a brief 3-step execution plan for the user's request."),
        HumanMessage(content="I need to process vector embeddings and summarize the document.")
    ])
    
    plan_text = response.content
    return {
        "plan": plan_text, 
        "messages": [f"Plan created: {plan_text}"]
    }

def execute_node(state: ReviewState) -> dict:
    """Node eksekusi yang hanya berjalan jika approved = True."""
    if not state.get("approved", False):
        return {
            "result": "Execution skipped - plan not approved.", 
            "messages": ["Skipped."]
        }
    
    return {
        "result": "Pipeline executed successfully based on the approved plan.", 
        "messages": ["Executed."]
    }

# ── 4. Graph Assembly & Checkpointer ──────────────────────────────────────────
# MemorySaver digunakan untuk menyimpan status/checkpoint agent di dalam memori
checkpointer = MemorySaver()

graph = StateGraph(ReviewState)
graph.add_node("plan", plan_node)
graph.add_node("execute", execute_node)

graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", END)

# Kunci Human-in-the-Loop: interrupt_before=["execute"]
# Graph akan BERHENTI sementara tepat sebelum masuk ke node "execute"
app = graph.compile(
    checkpointer=checkpointer, 
    interrupt_before=["execute"]
)

# ── 5. Execution Demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "review-session-1"}}

    print("--- STEP 1: Memulai Graph (Membuat Plan) ---")
    # Jalankan graph untuk pertama kali
    state = app.invoke(
        {"plan": "", "approved": False, "result": "", "messages": []},
        config=config
    )

    print("\n[Rencana dari Agent]:")
    print(state["plan"])
    print("\n--> Graph terhenti di sini (Interrupted)! Menunggu persetujuan manusia...")

    # Simulasi Manusia Memberikan Persetujuan (Human Approval)
    input("\nTekan [ENTER] untuk menyetujui plan dan melanjutkan eksekusi...")

    print("\n--- STEP 2: Melanjutkan Graph Setelah Disetujui ---")
    # Update state bahwa manusia sudah menyetujui (approved = True)
    state["approved"] = True

    # Panggil invoke kembali dengan config thread_id yang sama untuk melanjutkannya
    final = app.invoke(state, config=config)

    print("\n[Hasil Akhir Eksekusi]:")
    print(final["result"])
    print("\n[Histori Pesan]:")
    for msg in final["messages"]:
        print(f"- {msg}")