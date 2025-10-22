import os

CUDA_VISIBLE_DEVICES = [1, 2, 3, 4]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, CUDA_VISIBLE_DEVICES))
tensor_parallel_size = len(CUDA_VISIBLE_DEVICES)
temperature = 0.4
question = "Pete thinks of a number. He doubles it, adds 10, multiplies by 4, and ends up with 120. What was his original number?"
model_path = "Qwen/Qwen3-8B"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

# Function to load a model with vLLM
def load_model_with_vllm(model_path, dtype="auto", tensor_parallel_size=None, gpu_memory_utilization=0.85, max_model_len=None):
    """Load a model with vLLM for optimized inference across multiple GPUs"""
    try:
        torch.cuda.empty_cache()
        
        model = LLM(
            model=model_path,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,  # Number of GPUs to use for tensor parallelism
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True
        )
        
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        torch.cuda.empty_cache()
        raise

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = load_model_with_vllm(
    model_path=model_path,
    tensor_parallel_size=tensor_parallel_size,
    gpu_memory_utilization=0.9,
    max_model_len=38912,
)

sampling_params = SamplingParams(
    temperature=temperature,
    max_tokens=38912,
    top_p=1.0,
    top_k=-1,
    skip_special_tokens=True,
)

messages = [
    {
        "role": "system",
        "content": (
            "You are a math reasoning assistant.\n"
            "\n"
            "Formatting rules:\n"
            "1. Solve math problems in a **numbered list**, one logical step per line.\n"
            "2. Each step must start with the step number and a period (e.g., '1.').\n"
            "3. Each step must be a complete sentence that describes one reasoning move.\n"
            "4. Inline LaTeX expressions should use $...$.\n"
            "5. The final step must include the final boxed answer written as \\boxed{}.\n"
            "6. Do not include any explanations, headers, or summaries outside the numbered list.\n"
            "7. Do not use 'Step x:' or extra newlines between steps.\n"
            "8. End with no extra text after the final step."
        ),
    },
    {
        "role": "user",
        "content": question,
    },
]

text: str = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

print(text)
print("-----Split line-----")

output = model.generate(text, sampling_params=sampling_params)
print(output[0].outputs[0].text)
