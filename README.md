# AI Goggles — Edge Deployment on Raspberry Pi 4

Real-time scene captioning for the visually impaired, running fully on-device on a Raspberry Pi 4 using an ONNX-optimized ViT + GPT-2 pipeline with text-to-speech output.


## Overview

This branch is the deployment-optimized version of AI Goggles. The original prototype (`loop.py`, `loop_with_tts.py`) used BLIP for image captioning running on PyTorch; functional, but too heavy for CPU-only edge inference and not exportable to ONNX in a straightforward way.

This branch documents the full journey: BLIP prototype → architecture swap → ONNX export engineering → benchmarked edge deployment on RPi 4.


## Development Journey

### Stage 1 — BLIP Prototype (`loop.py`, `loop_with_tts.py`)
Initial implementation using `Salesforce/blip-image-captioning-base` via PyTorch. Captures a webcam frame every 2 minutes, generates a caption, and optionally speaks it via `pyttsx3` TTS. Works on a machine with sufficient RAM (like a laptop) but is not ONNX-exportable due to BLIP's tightly coupled encoder-decoder architecture.

### Stage 2 — Architecture Swap
BLIP was replaced with `nlpconnect/vit-gpt2-image-captioning` — a `VisionEncoderDecoderModel` combining a ViT image encoder and a GPT-2 text decoder. This architecture was chosen specifically because the two components can be exported to ONNX independently, enabling CPU-only inference without PyTorch at runtime.

### Stage 3 — ONNX Export (`main.py`)
Both models exported using `torch.onnx.export` at opset 16. The decoder required a non-trivial patch before export (see [ONNX Export Pipeline](#onnx-export-pipeline) below).

### Stage 4 — Inference Scripts
Four inference scripts of increasing complexity, developed iteratively: static image → live webcam → live + benchmarking → live + benchmarking + TTS.

### Stage 5 — Benchmarking on RPi 4
Deployed and benchmarked on Raspberry Pi 4, headless mode over SSH. Results logged per-frame and saved to CSV.


## Architecture

```
Webcam frame (OpenCV)
        ↓
Image preprocessing — resize 224×224, ViT normalization
        ↓
vit_encoder.onnx        ← ViT image encoder (ONNX, CPUExecutionProvider)
        ↓
encoder_hidden_states
        ↓
gpt2_decoder.onnx       ← GPT-2 decoder with cross-attention (ONNX, CPUExecutionProvider)
        ↓
Greedy token decoding (max 15 tokens, no KV-cache)
        ↓
Caption string → pyttsx3 TTS (spoken output)
```


## ONNX Export Pipeline

This was the core engineering challenge. `VisionEncoderDecoderModel` contains a ViT encoder and GPT-2 decoder that must be exported separately.

**The decoder problem:**
GPT-2's `forward()` uses KV-caching by default (`use_cache=True`), which returns `past_key_values` as dynamically nested tuples. `torch.onnx.export` cannot trace this structure. The fix is to monkey-patch the decoder's `forward` method to force `use_cache=False` before export, then restore the original afterward.

> **Tradeoff:** Disabling KV-cache means the decoder re-processes all previous tokens at every generation step (O(n²) attention). This is less efficient per step but is the correct tradeoff for ONNX compatibility on constrained hardware where session simplicity matters more than per-step speed.


## File Structure

```
├── main.py                      # ONNX export script — run this first to generate .onnx files
│
├── inference_img.py             # Single image inference — sanity check on a static image
├── inference_live.py            # Live webcam inference, no metrics logging
├── inference_live_benchmark.py  # Live webcam + per-frame latency/CPU/RAM logging
├── inference_live_tts.py        # Live webcam + benchmarking + pyttsx3 TTS spoken output
│
├── loop.py                      # Stage 1 prototype — BLIP via PyTorch, no TTS
├── loop_with_tts.py             # Stage 1 prototype — BLIP via PyTorch + TTS
│
├── benchmark_profiling.py       # Offline benchmark — N passes on a static image,
│                                #   logs per-run stats and saves results to .txt
├── benchmark_logs/              # .txt outputs from benchmark_profiling.py
│
├── TTS_testing/                 # Early TTS integration experiments
├── rpi_encoder-decoder-test.txt # Raw output log from initial RPi deployment test
├── shaking_hands.png            # Test image used for static inference benchmarking
└── requirements.txt
```


## Benchmark Results (RPi 4, 4GB RAM, headless)

Measured on live webcam feed via `inference_live_benchmark.py`:

| Stage | First run (warmup) | Steady state |
|---|---|---|
| ViT Encoder | ~22.9s | ~3.1s |
| GPT-2 Decoder | ~30.1s | ~7–11s |
| Total per caption | ~53s | ~10–14s |

> First inference is significantly slower due to ONNX Runtime session initialization and memory allocation. Subsequent frames stabilize.

**System utilization (steady state):**
- CPU: 74–82%
- RAM: ~95% (4GB fully utilized by ONNX sessions + OS)

**Sample live session output:**
Can be found in ```benchmark_logs.txt```.

## Setup

**Hardware:** Raspberry Pi 4 (4GB), USB webcam

**Install dependencies:**
```bash
pip install onnxruntime pillow numpy transformers torchvision opencv-python psutil pyttsx3
```

**Generate ONNX models** (run once, on any machine with PyTorch):
```bash
python main.py
# Outputs: vit_encoder.onnx, gpt2_decoder.onnx
```

**Copy `.onnx` files to RPi, then run:**
```bash
# Headless (SSH) — auto-detected, no DISPLAY needed
python inference_live_tts.py        # full pipeline with TTS
python inference_live_benchmark.py  # full pipeline with metrics, no TTS
python inference_live.py            # minimal, no metrics or TTS

# Offline benchmark on static image
python benchmark_profiling.py       # runs 5 passes, saves CSV to benchmark_logs/
```


## Key Implementation Details

+ **Headless auto-detection** — all scripts check `os.environ.get("DISPLAY")` and disable OpenCV GUI automatically when running over SSH.

+ **Greedy decoding** — argmax at each step rather than beam search, reducing decoder latency on constrained hardware.

+ **Frame sampling** — 3-second sleep between captures avoids redundant inference on near-identical frames and manages CPU thermal load.

+ **TTS latency tracking** — `inference_live_tts.py` separately times the `pyttsx3` speech output so TTS overhead is visible in logs and doesn't inflate inference metrics.

+ **CSV logging** — `benchmark_profiling.py` saves timestamped CSVs to `benchmark_logs/` for offline analysis.

---

## Limitations & Future Work

- **Latency** — ~10–14s per caption is too slow for real-time assistive use. INT8 quantization of the ONNX models would likely reduce decoder latency by 30–50%.
- **RAM pressure** — 95%+ RAM utilization leaves little headroom. An 8GB RPi 4 or RPi 5 would give more stability.
- **No KV-cache** — disabling caching for ONNX compatibility means O(n²) attention per decoding step. A future approach using ONNX-compatible static KV-cache would recover this efficiency.
- **TTS is sequential** — `pyttsx3` blocks inference during speech. Threading would reduce perceived lag significantly.
- **Caption repetition** — greedy decoding with short max-token limit produces repetitive captions on static scenes.
