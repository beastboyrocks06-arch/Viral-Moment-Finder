
from llama_cpp import Llama
from pathlib import Path

MODEL_PATH = Path("models/qwen2.5-1.5b-instruct-q4_k_m.gguf")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("Loading Qwen 2.5 1.5B...")

llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=4096,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False
)

print("Model loaded successfully!")

response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are an expert video content analyst."
        },
        {
            "role": "user",
            "content": """
            Analyze this podcast statement:

            "Everyone thinks success comes from working harder.
            But the real secret is learning which things
            are not worth your time. I wasted five years
            doing things that never moved my business forward."

            Explain:
            1. Is this an interesting Shorts moment?
            2. What makes it engaging?
            3. Suggest a catchy Shorts title.
            """
        }
    ],
    temperature=0.3,
    max_tokens=300
)

print("\n========== QWEN RESPONSE ==========\n")
print(response["choices"][0]["message"]["content"])
