# inference_rpi.py

from PIL import Image
import numpy as np
import onnxruntime as ort
from transformers import ViTImageProcessor, AutoTokenizer

# Load tokenizer and processor (same as PC)
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load quantized ONNX models
encoder_session = ort.InferenceSession("vit_encoder_quant.onnx")
decoder_session = ort.InferenceSession("gpt2_decoder_quant.onnx")

# Load and preprocess image
image = Image.open("shaking_hands.png").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"].numpy()

# Run encoder
encoder_outputs = encoder_session.run(None, {"pixel_values": pixel_values})
encoder_hidden = encoder_outputs[0]

# Start decoding
generated_ids = [tokenizer.cls_token_id]
for _ in range(20):
    decoder_input = np.array([generated_ids], dtype=np.int64)
    decoder_outputs = decoder_session.run(
        None,
        {"input_ids": decoder_input, "encoder_hidden_states": encoder_hidden}
    )
    next_token_logits = decoder_outputs[0][0, -1, :]
    next_token = int(np.argmax(next_token_logits))
    generated_ids.append(next_token)
    if next_token == tokenizer.eos_token_id:
        break

caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
print("Caption:", caption)
