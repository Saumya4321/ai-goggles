import onnxruntime as ort
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
import torchvision.transforms as T
import cv2
import time
import psutil
import statistics
import os
import pyttsx3

# Check if it is running on a device which is headless or has display

HEADLESS = not os.environ.get("DISPLAY")
print(f"Running in {'headless' if HEADLESS else 'GUI'} mode\n")


# Set up TTS output
engine = pyttsx3.init()
print("TTS engine setup complete.")

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("gpt2")
bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 50256
eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

# --- Load ONNX encoder and decoder sessions ---
encoder_sess = ort.InferenceSession("vit_encoder.onnx", providers=["CPUExecutionProvider"])
decoder_sess = ort.InferenceSession("gpt2_decoder.onnx", providers=["CPUExecutionProvider"])

# --- Image preprocessing (ViT base) ---
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- Webcam setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

print("Webcam started. Press Ctrl+C to stop.\n")

# --- Metrics storage ---
run_times = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if not HEADLESS:
        # Show the frame
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Start total timer
        run_start = time.perf_counter()

        # Convert to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pixel_values = transform(pil_img).unsqueeze(0).numpy()

        # --- Encoder ---
        t0 = time.perf_counter()
        encoder_outputs = encoder_sess.run(None, {"pixel_values": pixel_values})
        encoder_hidden_state = encoder_outputs[0]
        t1 = time.perf_counter()
        encoder_time = t1 - t0

        # --- Decoder ---
        t2 = time.perf_counter()
        generated_ids = [bos_token_id]
        for _ in range(15):  # max tokens
            decoder_input_ids = np.array([generated_ids], dtype=np.int64)
            decoder_out = decoder_sess.run(None, {
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_state
            })
            next_token_id = int(np.argmax(decoder_out[0][0, -1]))
            if next_token_id == eos_token_id:
                break
            generated_ids.append(next_token_id)
        t3 = time.perf_counter()
        decoder_time = t3 - t2

        # --- Caption decoding ---
        caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
        total_time = time.perf_counter() - run_start

        # --- System stats ---
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent

        # Log metrics
        run_times.append(total_time)

        print(f"{time.strftime('%H:%M:%S')} | Encoder: {encoder_time:.3f}s | Decoder: {decoder_time:.3f}s | 🧮 CPU: {cpu}% | 💾 RAM: {mem}%")
        print(f"    → Caption: {caption}")
        tts_start = time.perf_counter()
        

        engine.say(f"{caption}")
        engine.runAndWait()
        tts_time = time.perf_counter() - tts_start
        print(f"TTS Time: {tts_time:.2f}s")

        time.sleep(3)  # avoid spamming

except KeyboardInterrupt:
    print("\nStopped.")

# --- Summary ---
cap.release()
if not HEADLESS:
    cv2.destroyAllWindows()

if run_times:
    print("\n📊 Live Session Summary:")
    print(f"• Avg Time: {statistics.mean(run_times):.3f}s")
    print(f"• Min Time: {min(run_times):.3f}s")
    print(f"• Max Time: {max(run_times):.3f}s")
