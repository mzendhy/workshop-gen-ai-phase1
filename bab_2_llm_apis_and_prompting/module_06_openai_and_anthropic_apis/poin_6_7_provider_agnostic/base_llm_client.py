import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables dari file .env
load_dotenv()


# 2. Definisikan Data Structures (Standard Output)
@dataclass
class ChatMessage:
    role: str  # "user", "assistant", atau "system"
    content: str


@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


# 3. Abstract Base Class (Interface dasar untuk semua provider/model)
class BaseLLMClient(ABC):

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        pass


# 4. Implementation Client Menggunakan OpenRouter
class OpenRouterLLMClient(BaseLLMClient):

    def __init__(self, model: str = "meta-llama/llama-3.1-70b-instruct"):
        self.model = model

        # Mengambil API Key OpenRouter dari .env
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY tidak ditemukan di environment variables."
            )

        # Inisialisasi OpenAI Client dengan Custom Base URL OpenRouter
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        api_messages = []

        # Sisipkan System Prompt jika disediakan
        if system:
            api_messages.append({"role": "system", "content": system})

        # Masukkan pesan percakapan
        for m in messages:
            api_messages.append({"role": m.role, "content": m.content})

        # Eksekusi API Call via OpenRouter
        response = self._client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Ambil Token Usage secara aman
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return ChatResponse(
            text=response.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )


# ==========================================
# TEST RUN / CARA PENGGUNAAN
# ==========================================
if __name__ == "__main__":
    print("=== Testing Provider-Agnostic Client via OpenRouter ===\n")

    # Daftar model OpenRouter yang ingin kita uji secara bergantian
    models_to_test = [
        "meta-llama/llama-3.1-70b-instruct",
        "openai/gpt-4o-mini",
    ]

    # Pesan pengujian
    test_messages = [
        ChatMessage(
            role="user",
            content="Jelaskan secara singkat apa itu Provider-Agnostic Abstraction!",
        )
    ]

    for model_name in models_to_test:
        print(f"--- Menggunakan Model: {model_name} ---")
        client: BaseLLMClient = OpenRouterLLMClient(model=model_name)

        res = client.chat(
            messages=test_messages,
            system="Kamu adalah asisten AI yang ramah dan ringkas.",
            max_tokens=256,
            temperature=0.5,
        )

        print(f"Response:\n{res.text}\n")
        print(
            f"Usage -> Input Tokens: {res.input_tokens} | Output Tokens: {res.output_tokens}\n"
        )
        print("-" * 60)