from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
from types import MethodType
import os

# ------------------------------
# Load model and processors
# ------------------------------
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.eval()

# ------------------------------
# Dummy input
# ------------------------------
image = Image.new("RGB", (224, 224), color="white")
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]
decoder_input_ids = torch.tensor([[tokenizer.bos_token_id]])  # ✅ use BOS token, not CLS

# ------------------------------
# Export encoder
# ------------------------------
print("Exporting encoder...")
torch.onnx.export(
    model.encoder,
    pixel_values,
    "vit_encoder.onnx",
    input_names=["pixel_values"],
    output_names=["encoder_hidden_states"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "encoder_hidden_states": {0: "batch_size"}},
    opset_version=16
)

# ------------------------------
# Patch decoder forward to avoid caching
# ------------------------------
print("Patching decoder forward method to disable caching...")
original_forward = model.decoder.forward

def no_cache_forward(self, input_ids, encoder_hidden_states, **kwargs):
    return original_forward(
        input_ids=input_ids,
        encoder_hidden_states=encoder_hidden_states,
        use_cache=False
    )

model.decoder.forward = MethodType(no_cache_forward, model.decoder)

# ------------------------------
# Prepare dummy encoder output
# ------------------------------
with torch.no_grad():
    encoder_output = model.encoder(pixel_values)

decoder_inputs = {
    "input_ids": decoder_input_ids,
    "encoder_hidden_states": encoder_output.last_hidden_state,
}

# ------------------------------
# Export decoder
# ------------------------------
print("Exporting decoder...")
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
    opset_version=16
)

# Optional: restore original decoder forward method
model.decoder.forward = original_forward

# ------------------------------
# Quantize ONNX models
# ------------------------------
print("Quantizing encoder...")
quantize_dynamic("vit_encoder.onnx", "vit_encoder_quant.onnx", weight_type=QuantType.QInt8)

print("Quantizing decoder...")
quantize_dynamic("gpt2_decoder.onnx", "gpt2_decoder_quant.onnx", weight_type=QuantType.QInt8)

print("✅ Export and quantization complete.")
