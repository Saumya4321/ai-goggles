import onnxruntime as ort # to run ONNX models
from PIL import Image  # image processibg
import numpy as np
from transformers import AutoTokenizer # to decode output tokens
import torchvision.transforms as T # image processing
import cv2 # for frame capture from webcam
import time

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("gpt2") # for bos_token_id decoding; converts token IDS to text

# --- Load ONNX encoder and decoder sessions ---
encoder_sess = ort.InferenceSession("vit_encoder.onnx", providers=["CPUExecutionProvider"])
decoder_sess = ort.InferenceSession("gpt2_decoder.onnx", providers=["CPUExecutionProvider"])

# --- Image preprocessing (ViT base) ---
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- Read frame from webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

print("Webcam started. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Show the frame
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Convert to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pixel_values = transform(pil_img).unsqueeze(0).numpy()

        # Run encoder
        encoder_outputs = encoder_sess.run(None, {"pixel_values": pixel_values})
        encoder_hidden_state = encoder_outputs[0] # tensor representing image features

        # Decoder inference loop - generate caption token by token
        generated_ids = [tokenizer.bos_token_id or tokenizer.cls_token_id or 50256]
        for _ in range(15): # max tokens reduced to 15 to make decoding faster
            decoder_input_ids = np.array([generated_ids], dtype=np.int64)
            decoder_out = decoder_sess.run(None, {
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_state
            })
            next_token_id = int(np.argmax(decoder_out[0][0, -1])) # pick token with highest probability
            if next_token_id == tokenizer.eos_token_id or next_token_id == tokenizer.pad_token_id:
                break
            generated_ids.append(next_token_id)

        # Greedy decoding loop - always picks most likely word => Faster option

        # Decode token ID to text and show
        caption = tokenizer.decode(generated_ids, skip_special_tokens=True) # skip special tokens like <BOS> and <EOS>
        print(f"[Caption @ {time.strftime('%H:%M:%S')}] {caption}")

        time.sleep(3)  # avoid spamming every frame

except KeyboardInterrupt:
    print("Stopped.")

cap.release()
cv2.destroyAllWindows()
