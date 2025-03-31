import textwrap
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import pandas as pd
from datetime import datetime

# Initialize Accelerator for multi-GPU support
accelerator = Accelerator()

# Model ID
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Load tokenizer and model (on CPU first)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16  # Use FP16 for efficiency
)

# Prepare model for distributed inference
model = accelerator.prepare(model)

# Define system prompt
SYSTEM_PROMPT = textwrap.dedent("""Let's think step by step and respond in this format:
<reasoning>
str
</reasoning>
<answer>
str
</answer>
""").strip()

# Get sample prompts
df = pd.read_parquet('data/glavieai/train.parquet')
prompts = df['prompt'].sample(10, random_state=42).tolist()

# Apply chat template to each prompt (on CPU)
chat_prompts = []
for prompt in prompts:
    prompt_chat = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt[0]['content']}
    ]
    prompt_chat_str = tokenizer.apply_chat_template(
        prompt_chat,
        add_generation_prompt=True,
        tokenize=False
    )
    chat_prompts.append(prompt_chat_str)

# Tokenize the chat-formatted prompts
inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True)

# Move inputs to the distributed device
inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}

# Custom generation loop (since DDP doesn't support generate)
# Gather inputs to main process and use unwrapped model for generation
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    with torch.no_grad():
        outputs = unwrapped_model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=True,
            temperature=0.9
        )
else:
    outputs = None

# Synchronize processes and broadcast outputs
outputs = accelerator.gather(outputs)
if outputs is not None:
    outputs = outputs[:len(prompts)]  # Trim to match prompt count

# Decode outputs
if accelerator.is_main_process and outputs is not None:
    results = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
else:
    results = None

# Define output file name with timestamp
output_dir = "inference_logs"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"prompt_response_{timestamp}.txt")

# Write results to file and print, only on main process
if accelerator.is_main_process and results is not None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Inference Results\n")
        f.write("=" * 50 + "\n\n")
        for original_prompt, result in zip(chat_prompts, results):
            f.write(f"Prompt:\n{original_prompt}\n\n")
            f.write(f"Response:\n{result}\n")
            f.write("-" * 50 + "\n\n")
            print(f"Prompt: {original_prompt}\nResponse: {result}\n")
    
    print(f"Results saved to {output_file}")

# Synchronize all processes
accelerator.wait_for_everyone()