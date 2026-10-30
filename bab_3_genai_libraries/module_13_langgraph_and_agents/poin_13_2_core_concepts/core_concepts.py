from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

# 1. State Definition
class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]
    step_count: int
    final_answer: str

# 2. Node Functions
def think_node(state: AgentState) -> dict:
    thought = f"Step {state['step_count'] + 1}: analysing the problem..."
    return {
        "messages": [thought],
        "step_count": state["step_count"] + 1,
    }

def answer_node(state: AgentState) -> dict:
    return {"final_answer": "Based on my analysis, the answer is X."}

# 3. Conditional Edge Logic
def should_continue(state: AgentState) -> str:
    if state["step_count"] < 2:
        return "think"
    return "answer"

# 4. Graph Construction
graph = StateGraph(AgentState)

graph.add_node("think", think_node)
graph.add_node("answer", answer_node)

graph.add_edge(START, "think")
graph.add_conditional_edges(
    "think", 
    should_continue, 
    {"think": "think", "answer": "answer"}
)
graph.add_edge("answer", END)

# Compile Graph
app = graph.compile()

# 5. Execution
if __name__ == "__main__":
    result = app.invoke({"messages": [], "step_count": 0, "final_answer": ""})
    print("Messages:", result["messages"])
    print("Answer  :", result["final_answer"])