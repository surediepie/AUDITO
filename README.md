@"
# Audito — Voice Authenticity Screening

Audito is a Python-based prototype designed to screen audio recordings for potential AI-generated or spoofed speech.

## Features

- Upload audio recordings for analysis
- AI audio classification using a pretrained Hugging Face model
- Classification labels and confidence scores
- Audio validation and error handling
- Web interface built with Gradio

## Tech Stack

- Python 3.11
- PyTorch
- Hugging Face Transformers
- Gradio
- NumPy
- SoundFile

## Model

Audito uses the following pretrained model:

[MelodyMachine/Deepfake-audio-detection-V2](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2)

The model's output is a screening signal, not definitive proof that a recording is genuine or AI-generated. Displayed scores are not calibrated probabilities.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/audito.git
cd audito
