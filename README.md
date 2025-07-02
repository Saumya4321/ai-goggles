# AI Goggles - Live Video Captioning for the Visually Impaired

This project aims to develop an AI system that uses live video from a webcam, generates real-time captions describing the scene, and assists visually impaired individuals.

## Present Features
- Live webcam capture
- Scene understanding via image captioning 

## Tech Stack
- Python
- OpenCV
- ViT- GPT-2
- pyttsx3 (https://pypi.org/project/pyttsx3/)

## Setup
```bash
pip install -r requirements.txt
python main.py
```
## Notes
+ On Raspberry Pi, install onnxruntime 1.17.1
+ For onnxruntime 1.17.1, numpy versions above 2 is not compatible. Hence uninstall the current numpy versions and reinstall older versions (<2)
```bash
pip uninstall numpy
pip install "numpy<2"
```
