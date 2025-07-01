import onnxruntime as ort
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
import torchvision.transforms as T
import os

# -------------------------
# Load tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("gpt2")
bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 50256
eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

# -------------------------
# Load ONNX models
# -------------------------
encoder_sess = ort.InferenceSession("vit_encoder.onnx", providers=["CPUExecutionProvider"])
decoder_sess = ort.InferenceSession("gpt2_decoder.onnx", providers=["CPUExecutionProvider"])

# -------------------------
# Preprocessing transform
# -------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# -------------------------
# Load image
# -------------------------
image_path = "shaking_hands.png"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"{image_path} not found.")

pil_img = Image.open(image_path).convert("RGB")
pixel_values = transform(pil_img).unsqueeze(0).numpy()

# -------------------------
# Run encoder
# -------------------------
encoder_outputs = encoder_sess.run(None, {"pixel_values": pixel_values})
encoder_hidden_state = encoder_outputs[0]

# -------------------------
# Decode caption (greedy loop)
# -------------------------
generated_ids = [bos_token_id]

for _ in range(15):  # max caption length
    input_ids = np.array([generated_ids], dtype=np.int64)
    decoder_output = decoder_sess.run(None, {
        "input_ids": input_ids,
        "encoder_hidden_states": encoder_hidden_state
    })
    next_token_id = int(np.argmax(decoder_output[0][0, -1]))
    
    if next_token_id == eos_token_id:
        break

    generated_ids.append(next_token_id)

# -------------------------
# Decode tokens
# -------------------------
caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"\nCaption: {caption}")
