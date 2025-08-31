import torch
import time
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from model_hf import LilleConfig, LilleForCausalLM

print("Registering custom 'lille-130m' architecture...")
AutoConfig.register("lille-130m", LilleConfig)
AutoModelForCausalLM.register(LilleConfig, LilleForCausalLM)
print("Registration complete.")

MODEL_DIR = "model"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch_dtype = torch.float32

if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        print("Hardware supports bfloat16, using it for better performance.")
    else:
        torch_dtype = torch.float16
        print("Hardware does not support bfloat16, falling back to float16.")

print(f"Loading tokenizer from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print(f"Loading model from {MODEL_DIR}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch_dtype,
    device_map=DEVICE,
)
print("Model loaded successfully!")

model.eval()

print("Compiling the model with torch.compile... (This may take a moment)")
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
print("Model compiled successfully!")

print("Performing a warmup run...")
with torch.inference_mode():
    _ = model.generate(
        tokenizer("<|startoftext|>", return_tensors="pt").input_ids.to(DEVICE),
        max_new_tokens=2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
print("Warmup complete.")


chat = [
    {"role": "user", "content": "What is the capital of France?"},
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

print(f"\n--- Prompt ---\n{prompt}")

inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

start_time = time.time()
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.5,
        top_p=0.95,
        use_cache=True
    )
end_time = time.time()

response_ids = outputs[0][inputs['input_ids'].shape[1]:]
response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

num_tokens = len(response_ids)
elapsed_time = end_time - start_time
tokens_per_second = num_tokens / elapsed_time if elapsed_time > 0 else float('inf')

print(f"\n--- Response ---\n{response_text}")
print(f"\n--- Statistics ---")
print(f"Tokens generated: {num_tokens}")
print(f"Time taken: {elapsed_time:.2f} seconds")
print(f"Tokens/second: {tokens_per_second:.2f}")
