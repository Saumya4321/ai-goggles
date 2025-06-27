import cv2
import time
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pyttsx3
import os

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up TTS output
engine = pyttsx3.init()



# Check if it is running on a device which is headless or has display

HEADLESS = not os.environ.get("DISPLAY")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("Webcam feed started. Capturing image every 2 minutes... Press 'Ctrl+C' to quit.")

# Timer setup
last_capture_time = time.time()
interval = 120  # seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Show webcam feed if not headless

        if not HEADLESS:
            cv2.imshow('Live Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
    

        # Check if 2 minutes have passed
        current_time = time.time()
        if current_time - last_capture_time >= interval:
            last_capture_time = current_time

            # Convert frame to RGB PIL image
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Run BLIP captioning
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)

            print(f"\n[Caption @ {time.strftime('%H:%M:%S')}] {caption}")
            engine.say(f"{caption}")
            engine.runAndWait()

    

except KeyboardInterrupt:
    print("Stopped by user.")

cap.release()
cv2.destroyAllWindows()
