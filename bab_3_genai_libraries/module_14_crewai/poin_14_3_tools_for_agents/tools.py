import os
import json
from typing import Type
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI

# 1. Load Environment Variables dari file .env
load_dotenv()

# Retrieve API Key OpenRouter dari .env
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# 2. Inisialisasi LLM via OpenRouter
# Kamu bisa mengganti model ke "openai/gpt-4o-mini" jika diperlukan
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-70b-instruct",
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7
)

# ─── DEFINISI CUSTOM TOOLS ───────────────────────────────────────────────────

# Tool 1: Token Cost Calculator Input Schema
class TokenCostCalculatorInput(BaseModel):
    model: str = Field(description="LLM model name, e.g. 'gpt-4o'")
    input_tokens: int = Field(description="Number of input tokens")
    output_tokens: int = Field(description="Number of output tokens")

# Tool 1: Token Cost Calculator BaseTool Class
class TokenCostCalculator(BaseTool):
    name: str = "Token Cost Calculator"
    description: str = "Calculate the USD cost of an LLM API call given model name and token counts."
    args_schema: Type[BaseModel] = TokenCostCalculatorInput

    def _run(self, model: str, input_tokens: int, output_tokens: int) -> str:
        pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
            "meta-llama/llama-3.1-70b-instruct": {"input": 0.35, "output": 0.40},
        }
        if model not in pricing:
            return f"Unknown model: {model}"
        
        p = pricing[model]
        cost = (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
        return json.dumps({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        })

# Tool 2: Model Context Lookup Input Schema
class ModelContextLookupInput(BaseModel):
    model: str = Field(description="LLM model name")

# Tool 2: Model Context Lookup BaseTool Class
class ModelContextLookup(BaseTool):
    name: str = "Model Context Window Lookup"
    description: str = "Look up the context window size (in tokens) of an LLM."
    args_schema: Type[BaseModel] = ModelContextLookupInput

    def _run(self, model: str) -> str:
        windows = {
            "gpt-4o": 128_000,
            "gpt-4o-mini": 128_000,
            "claude-sonnet-4-5": 200_000,
            "meta-llama/llama-3.1-70b-instruct": 128_000,
            "gemini-1.5-pro": 1_000_000,
        }
        tokens = windows.get(model, "Unknown")
        return json.dumps({"model": model, "context_window_tokens": tokens})

# Instantiate tools agar siap di-import/digunakan oleh Agent di poin berikutnya
cost_tool = TokenCostCalculator()
context_tool = ModelContextLookup()

if __name__ == "__main__":
    # Test jalankan tool secara mandiri
    print("Testing TokenCostCalculator:")
    print(cost_tool.run(model="meta-llama/llama-3.1-70b-instruct", input_tokens=1000, output_tokens=500))
    
    print("\nTesting ModelContextLookup:")
    print(context_tool.run(model="meta-llama/llama-3.1-70b-instruct"))