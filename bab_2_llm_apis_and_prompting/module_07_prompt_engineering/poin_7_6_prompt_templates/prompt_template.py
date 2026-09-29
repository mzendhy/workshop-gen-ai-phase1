import os
from dataclasses import dataclass, field
from string import Formatter
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables dari .env
load_dotenv()

# 2. Inisialisasi OpenAI Client yang mengarah ke OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# 3. Struktur Dataclass PromptTemplate sesuai materi poin 7.6
@dataclass
class PromptTemplate:
    """A reusable, versioned prompt template with variable validation."""
    name: str
    system: str
    user: str
    version: str = "1.0"
    required_vars: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Auto-detect required variables from both system and user templates
        formatter = Formatter()
        combined = self.system + self.user
        self.required_vars = [
            fname for _, fname, _, _ in formatter.parse(combined)
            if fname is not None
        ]

    def render(self, **kwargs: Any) -> tuple[str, str]:
        """Return (rendered_system, rendered_user). Raises error if required vars are missing."""
        missing = set(self.required_vars) - set(kwargs)
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return self.system.format(**kwargs), self.user.format(**kwargs)


# 4. Definisi Template
QA_TEMPLATE = PromptTemplate(
    name="question_answering",
    version="1.2",
    system="You are a {domain} expert. Answer questions accurately and concisely.",
    user="Question: {question}\n\nContext:\n{context}",
)


# 5. Eksekusi dan Pengujian ke OpenRouter
if __name__ == "__main__":
    # Render template dengan mengisi variabel yang dibutuhkan
    system_rendered, user_rendered = QA_TEMPLATE.render(
        domain="machine learning",
        question="What is Prompt Engineering?",
        context="Prompt engineering is the practice of designing and optimizing inputs to reliably produce desired outputs from LLMs."
    )

    print("--- RENDERED SYSTEM PROMPT ---")
    print(system_rendered)
    print("\n--- RENDERED USER PROMPT ---")
    print(user_rendered)
    print("\n--- SENDING REQUEST TO OPENROUTER ---")

    # Pemanggilan API OpenRouter
    # Kamu bisa mengganti model ke "openai/gpt-4o-mini" jika diperlukan
    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-70b-instruct", 
        messages=[
            {"role": "system", "content": system_rendered},
            {"role": "user", "content": user_rendered},
        ],
        temperature=0.7,
        max_tokens=512,
    )

    print("\n--- RESPONSE FROM MODEL ---")
    print(response.choices[0].message.content)