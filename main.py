import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'c' to capture and caption a frame. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Convert OpenCV BGR to RGB, then to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(image, return_tensors="pt")

        with torch.no_grad():
            out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        print(f"Caption: {caption}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
