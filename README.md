# Audito MVP

A small Gradio app for voice-spoof screening using a Hugging Face audio-classification model.

## Important limitations

- Treat output as a screening signal, not proof of authenticity.
- Confirm the model's label mapping and test it on a labeled evaluation set before making performance claims.
- Do not upload private or sensitive recordings.
- The first inference may take time because model weights are downloaded at runtime.

## Run locally (Windows PowerShell)

Use Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open http://localhost:7860.

## Docker

```powershell
docker build -t audito:latest .
docker run --rm -p 7860:7860 audito:latest
```

## Configuration

Set `AUDITO_MODEL_ID` to another compatible Hugging Face audio-classification model if needed. Verify its task, labels, license, and intended use before changing it.

## AWS

The Dockerfile can be used with Amazon ECR and a container hosting service such as Amazon ECS Express Mode. AWS resources are billable; set a budget alert and delete resources when finished.
