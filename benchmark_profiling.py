import onnxruntime as ort
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
import torchvision.transforms as T
import time
import statistics
import psutil
import os
import csv

# -------------------------
# Config
# -------------------------
IMAGE_PATH = "shaking_hands.png"
ENCODER_MODEL = "vit_encoder.onnx"
DECODER_MODEL = "gpt2_decoder.onnx"
NUM_RUNS = 5  # Benchmark over N runs
MAX_TOKENS = 15
SAVE_CSV = True

# -------------------------
# Setup
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("gpt2")
bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 50256
eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

# load the ONNX encoder and decoder into memory for inference
encoder_sess = ort.InferenceSession(ENCODER_MODEL, providers=["CPUExecutionProvider"])
decoder_sess = ort.InferenceSession(DECODER_MODEL, providers=["CPUExecutionProvider"])

# image processing function
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"{IMAGE_PATH} not found.")
pil_img = Image.open(IMAGE_PATH).convert("RGB")
pixel_values = transform(pil_img).unsqueeze(0).numpy()

# -------------------------
# Benchmark
# -------------------------
results = []

print(f"\nBenchmarking over {NUM_RUNS} runs...\n")

for i in range(NUM_RUNS):
    run_start = time.perf_counter()

    # --- Encoder ---
    t0 = time.perf_counter()
    encoder_outputs = encoder_sess.run(None, {"pixel_values": pixel_values})
    encoder_hidden_state = encoder_outputs[0]
    t1 = time.perf_counter()
    encoder_time = t1 - t0

    # --- Decoder ---
    t2 = time.perf_counter()
    generated_ids = [bos_token_id]
    for _ in range(MAX_TOKENS):
        input_ids = np.array([generated_ids], dtype=np.int64)
        decoder_output = decoder_sess.run(None, {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden_state
        })
        next_token_id = int(np.argmax(decoder_output[0][0, -1]))
        if next_token_id == eos_token_id:
            break
        generated_ids.append(next_token_id)
    t3 = time.perf_counter()
    decoder_time = t3 - t2

    # --- Caption ---
    caption = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # --- CPU + RAM ---
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent

    total_time = time.perf_counter() - run_start
    print(f"Run {i+1}: {total_time:.3f}s |  Encoder: {encoder_time:.3f}s |  Decoder: {decoder_time:.3f}s | 🧮 CPU: {cpu}% | 💾 RAM: {mem}%")
    print(f"    → Caption: {caption}")

    results.append({
        "Run": i + 1,
        "Encoder Time (s)": round(encoder_time, 3),
        "Decoder Time (s)": round(decoder_time, 3),
        "Total Time (s)": round(total_time, 3),
        "CPU (%)": cpu,
        "RAM (%)": mem,
        "Caption": caption
    })

# -------------------------
# Summary
# -------------------------
total_times = [r["Total Time (s)"] for r in results]
print("\n Summary:")
print(f"• Avg Time: {statistics.mean(total_times):.3f}s")
print(f"• Min Time: {min(total_times):.3f}s")
print(f"• Max Time: {max(total_times):.3f}s")

# -------------------------
# Optional: Save CSV
# -------------------------
if SAVE_CSV:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = "benchmark_logs"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.csv")

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n Results saved to {csv_file}")
