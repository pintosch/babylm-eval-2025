import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import os

# Path to your local model directory
local_model_path = "./models/Qwen3-0.6B"

# === Step 1: Load config and tokenizer ===
config = AutoConfig.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# === Step 2: Load model (with fallback) ===
load_pretrained = True

if load_pretrained:
    print("✅ Loading pretrained weights...")
    model = AutoModelForCausalLM.from_pretrained(local_model_path, config=config, torch_dtype=torch.bfloat16)
else:
    print("⚠️ Using no weights — initializing with random weights.")
    model = AutoModelForCausalLM.from_config(config)

# === Step 3: Prepare for inference ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === Step 4: Generate text ===
prompt = "What is smaller than a plane but larger than a cat? A: dog, B: ship, C: snake"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_length=200,
        temperature=0.8,
        do_sample=True
    )

output_text = tokenizer.decode(generated[0], skip_special_tokens=True)

print("\n🧠 Model output:")
print(output_text)
