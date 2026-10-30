import os
import torch
from dotenv import load_dotenv
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# 1. Load Environment Variables dari file .env di root
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print(f"OpenRouter API Key Loaded: {'Yes' if OPENROUTER_API_KEY else 'No'}")

# 2. Inisialisasi Model Name
# Untuk fine-tuning LoRA lokal Hugging Face, kita gunakan Llama 3.1 8B atau distilgpt2 sebagai fallback CPU
MODEL_NAME = "distilgpt2"  # Ganti ke "meta-llama/Llama-3.1-8B" jika memiliki GPU/Akses HuggingFace Token

print(f"Loading Model & Tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 3. Konfigurasi LoRA (Low-Rank Adaptation)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                  # Rank
    lora_alpha=32,        # Scaling factor
    target_modules=["c_attn"] if "gpt2" in MODEL_NAME else ["q_proj", "v_proj"], # Adaptive target modules
    lora_dropout=0.1,
    bias="none",
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# 4. Dataset Latihan (Q&A Sample)
training_examples = [
    "Question: What is RAG? Answer: RAG stands for Retrieval-Augmented Generation.",
    "Question: What is fine-tuning? Answer: Fine-tuning trains a pre-trained model on new data.",
    "Question: What is an embedding? Answer: An embedding is a dense vector representing text meaning.",
    "Question: What is a vector database? Answer: A vector database stores embeddings for similarity search.",
    "Question: What is LangChain? Answer: LangChain is a framework for building LLM applications.",
]

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )

dataset = Dataset.from_dict({"text": training_examples})
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x["input_ids"]})

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./output_lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
)

# 6. Training menggunakan Trainer API
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("Starting LoRA Fine-Tuning...")
trainer.train()

# 7. Simpan Adapter LoRA saja (Ukuran file kecil)
output_adapter_dir = "./lora_adapter"
peft_model.save_pretrained(output_adapter_dir)
tokenizer.save_pretrained(output_adapter_dir)
print(f"LoRA Adapter saved successfully to {output_adapter_dir}")

# 8. Pengujian Inference dengan Adapter
print("\n--- Testing Inference with Fine-Tuned Model ---")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
merged_model = PeftModel.from_pretrained(base_model, output_adapter_dir)
merged_model.eval()

prompt_input = "Question: What is RAG? Answer:"
inputs = tokenizer(prompt_input, return_tensors="pt")

with torch.no_grad():
    outputs = merged_model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)

print("Generated Response:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))