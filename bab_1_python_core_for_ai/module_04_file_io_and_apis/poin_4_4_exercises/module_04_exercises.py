# 4.4 Module 04 Exercises
import json
import threading
import time
from pathlib import Path

def save_conversation(history: list[dict], path: str) -> None:
    Path(path).write_text(json.dumps(history, indent=2), encoding="utf-8")

def load_conversation(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

class ThreadSafeCSVLogger:
    def __init__(self, filename: str):
        self.filename = filename
        self.lock = threading.Lock()
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write("timestamp,model,input_tokens,output_tokens,latency_ms\n")

    def log(self, model: str, in_tok: int, out_tok: int, latency_ms: float):
        with self.lock:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(f"{time.time()},{model},{in_tok},{out_tok},{latency_ms}\n")
