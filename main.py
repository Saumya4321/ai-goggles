# export_and_quantize.py

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# Load model and processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.eval()

# Dummy image
image = Image.new("RGB", (224, 224), color="white")
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]
decoder_input_ids = torch.tensor([[tokenizer.bos_token_id]])


# --- Export encoder ---
torch.onnx.export(
    model.encoder,
    pixel_values,
    "vit_encoder.onnx",
    input_names=["pixel_values"],
    output_names=["encoder_hidden_states"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "encoder_hidden_states": {0: "batch_size"}},
    opset_version=16  # ✅ increased from 13 to 16
)


# --- Export decoder ---
with torch.no_grad():
    encoder_output = model.vision_encoder(pixel_values)

decoder_inputs = {
    "input_ids": decoder_input_ids,
    "encoder_hidden_states": encoder_output.last_hidden_state,
}

torch.onnx.export(
    model.decoder,
    (decoder_inputs["input_ids"], decoder_inputs["encoder_hidden_states"]),
    "gpt2_decoder.onnx",
    input_names=["input_ids", "encoder_hidden_states"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size", 1: "sequence"},
    },
    opset_version=16  # ✅ also here
)


# --- Quantize ---
quantize_dynamic("vit_encoder.onnx", "vit_encoder_quant.onnx", weight_type=QuantType.QInt8)
quantize_dynamic("gpt2_decoder.onnx", "gpt2_decoder_quant.onnx", weight_type=QuantType.QInt8)

print("Export and quantization complete.")
