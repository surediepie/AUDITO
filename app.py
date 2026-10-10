
import os
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
from transformers import pipeline

MODEL_ID = os.getenv(
    "AUDITO_MODEL_ID",
    "MelodyMachine/Deepfake-audio-detection-V2",
)
MAX_BYTES = int(os.getenv("AUDITO_MAX_BYTES", str(15 * 1024 * 1024)))
MAX_SECONDS = int(os.getenv("AUDITO_MAX_SECONDS", "30"))

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "audio-classification",
            model=MODEL_ID,
            device=-1,
        )
    return _classifier


def explain_label(label):
    normalized = label.lower().replace("_", " ").replace("-", " ")

    if any(word in normalized for word in (
        "fake", "spoof", "synthetic", "generated", "deepfake"
    )):
        return (
            "Potential spoof indicators detected",
            "The model selected a label associated with spoofed or synthetic "
            "audio. This is a screening signal, not proof."
        )

    if any(word in normalized for word in (
        "real", "bonafide", "bona fide", "genuine", "human"
    )):
        return (
            "Model classified this clip as likely bona fide",
            "The model selected a label associated with genuine speech. "
            "This does not prove the recording is authentic."
        )

    return (
        f"Model result: {label}",
        "Check the model's label definitions before interpreting this result."
    )


def analyze(audio_file):
    if not audio_file:
        raise gr.Error("Upload an audio file first.")

    path = Path(audio_file)

    if not path.exists():
        raise gr.Error("File not found. Please upload the audio again.")

    if path.stat().st_size > MAX_BYTES:
        raise gr.Error("File too large. Please use an audio file under 15 MB.")

    try:
        audio, sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=False,
        )
    except Exception as exc:
        print(f"Audio read error: {type(exc).__name__}: {exc}")
        raise gr.Error("Could not read the audio. Try a WAV file.") from exc

    if audio.size == 0:
        raise gr.Error("The audio file is empty.")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = len(audio) / float(sr)

    if duration > MAX_SECONDS:
        raise gr.Error(f"Use an audio clip of {MAX_SECONDS} seconds or less.")

    if duration < 0.25:
        raise gr.Error("Audio is too short. Use at least 0.25 seconds.")

    if not np.isfinite(audio).all():
        raise gr.Error("The audio contains invalid values.")

    try:
        classifier = get_classifier()
        results = classifier(
            {"raw": audio, "sampling_rate": int(sr)},
            top_k=None,
        )
    except Exception as exc:
        print(f"Inference error: {type(exc).__name__}: {exc}")
        raise gr.Error(
            "Analysis failed. Check the PowerShell terminal for details."
        ) from exc

    if not results:
        raise gr.Error("The model returned no results.")

    results = sorted(
        results,
        key=lambda item: float(item.get("score", 0)),
        reverse=True,
    )

    top = results[0]
    label = str(top.get("label", "unknown"))
    percentage = float(top.get("score", 0)) * 100
    verdict, guidance = explain_label(label)

    summary = (
        f"### {verdict}\n\n"
        f"**Top model label:** `{label}`\n\n"
        f"**Model score:** {percentage:.1f}%\n\n"
        f"{guidance}\n\n"
        "**Recommended action:** Verify the speaker through a separate, "
        "trusted channel before acting on a sensitive request."
    )

    table = [
        [
            str(item.get("label", "unknown")),
            f"{float(item.get('score', 0)) * 100:.1f}%",
        ]
        for item in results
    ]

    return summary, table



CSS = """
/* Page background and default text */
.gradio-container {
    background: #fff1f5 !important;
    color: #3b2631 !important;
}

/* Main content */
.gradio-container .main {
    max-width: 1000px !important;
    margin: auto !important;
}

/* Force readable text throughout the interface */
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container p,
.gradio-container label,
.gradio-container span,
.gradio-container .prose,
.gradio-container .markdown {
    color: #3b2631 !important;
}

/* Headings use a darker rose */
.gradio-container h1 {
    color: #873e60 !important;
}

.gradio-container h2,
.gradio-container h3 {
    color: #9b4f70 !important;
}

/* Cards and input areas */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .wrap {
    background-color: #fffafd !important;
    color: #3b2631 !important;
}

/* Audio upload and player */
.gradio-container [data-testid="file-upload"],
.gradio-container [data-testid="audio"],
.gradio-container .audio-container {
    background-color: #fffafd !important;
    color: #3b2631 !important;
}

/* Inputs */
.gradio-container input,
.gradio-container textarea {
    background-color: #ffffff !important;
    color: #3b2631 !important;
    border-color: #e4bdce !important;
}

/* Primary button */
.gradio-container button.primary {
    background-color: #c8668e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
}

.gradio-container button.primary:hover {
    background-color: #ad4e77 !important;
}

/* Results table */
.gradio-container table,
.gradio-container td {
    background-color: #ffffff !important;
    color: #3b2631 !important;
}

.gradio-container th {
    background-color: #f5dce6 !important;
    color: #6e304c !important;
}

/* Footer and links */
.gradio-container footer {
    color: #59414c !important;
}

.gradio-container a {
    color: #963e66 !important;
}

"""


with gr.Blocks(
    title="Audito | Voice Authenticity Screening",
    theme=gr.themes.Soft(
        primary_hue="pink",
        secondary_hue="pink",
        neutral_hue="slate",
    ),
    css=CSS,
) as demo:

    gr.Markdown(
        """
        # Audito
        ### Voice authenticity screening

        Upload a short audio clip to inspect the model's classification.
        Audito is a screening aid, not a forensic determination.
        """
    )

    audio_input = gr.Audio(
        label="Upload audio (WAV recommended)",
        type="filepath",
    )

    analyze_button = gr.Button(
        "Analyze audio",
        variant="primary",
    )

    result = gr.Markdown()

    scores = gr.Dataframe(
        headers=["Model label", "Score"],
        datatype=["str", "str"],
        label="Model outputs",
        interactive=False,
    )

    analyze_button.click(
        fn=analyze,
        inputs=audio_input,
        outputs=[result, scores],
    )

    gr.Markdown(
        """
        ---

        **Privacy:** Uploaded recordings are not intentionally saved by
        this application. Hosting platforms may create operational logs.
        Avoid uploading private recordings.

        **Limitations:** Scores are model outputs, not calibrated
        probabilities of authenticity. Results depend on the model and
        recording quality.
        """
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=False,
    )
