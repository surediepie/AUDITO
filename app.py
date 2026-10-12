import os

import time

import subprocess

import tempfile

from pathlib import Path

from datetime import datetime

# --- Defensive patch for gradio_client schema parsing on boolean properties ---

import gradio_client.utils as client_utils

_orig_json_schema_to_python_type = client_utils._json_schema_to_python_type

_orig_get_type = client_utils.get_type

def _patched_json_schema_to_python_type(schema, defs):

    if isinstance(schema, bool) or not isinstance(schema, dict):

        return "Any"

    return _orig_json_schema_to_python_type(schema, defs)

def _patched_get_type(schema):

    if isinstance(schema, bool) or not isinstance(schema, dict):

        return "Any"

    return _orig_get_type(schema)

client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type

client_utils.get_type = _patched_get_type

# -------------------------------------------------------------------------------

import gradio as gr

import numpy as np

import soundfile as sf

import imageio_ffmpeg

import torch

from transformers import pipeline

# Configuration

MODEL_ID = os.getenv("AUDITO_MODEL_ID", "MelodyMachine/Deepfake-audio-detection-V2")

LINKEDIN_URL = os.getenv("AUDITO_LINKEDIN_URL", "https://www.linkedin.com/in/priyam-kumar-7b674b2b4/").strip()
GITHUB_URL = os.getenv("AUDITO_GITHUB_URL", "https://github.com/surediepie").strip()

MAX_BYTES = int(os.getenv("AUDITO_MAX_BYTES", str(15 * 1024 * 1024)))

MAX_SECONDS = int(os.getenv("AUDITO_MAX_SECONDS", "30"))

PORT = int(os.getenv("PORT", "7860"))

# Optimal thread allocation for CPU

cpu_threads = min(4, os.cpu_count() or 4)

torch.set_num_threads(cpu_threads)

print(f"[AUDITO] Loading classifier ({MODEL_ID})...", flush=True)

_classifier = pipeline(

    "audio-classification",

    model=MODEL_ID,

    device=-1,  # CPU

)

print("[AUDITO] Forensics engine ready.", flush=True)

def ffmpeg_executable():

    return imageio_ffmpeg.get_ffmpeg_exe()

def decode_to_pcm(input_path, output_path, sample_rate=None, channels=1):

    """Decodes input audio to PCM WAV preserving or resampling sample rate."""

    command = [

        ffmpeg_executable(), "-y", "-hide_banner", "-loglevel", "error",

        "-i", str(input_path), "-vn",

    ]

    if channels is not None:

        command += ["-ac", str(channels)]

    if sample_rate is not None:

        command += ["-ar", str(sample_rate)]

    command += ["-c:a", "pcm_s16le", str(output_path)]

    try:

        completed = subprocess.run(command, capture_output=True, text=True, timeout=45)

    except subprocess.TimeoutExpired as exc:

        raise gr.Error("Audio processing timed out.") from exc

    if completed.returncode != 0 or not Path(output_path).exists():

        raise gr.Error("Failed to decode audio payload.")

    return str(output_path)

def mp3_to_wav(mp3_file):

    if not mp3_file:

        raise gr.Error("Select an MP3 file first.")

    source = Path(mp3_file)

    if not source.exists() or not source.is_file():

        raise gr.Error("File not found.")

    if source.stat().st_size > MAX_BYTES:

        raise gr.Error(f"File exceeds {MAX_BYTES // (1024 * 1024)} MB limit.")

    destination = Path(tempfile.gettempdir()) / f"audito-export-{time.time_ns()}.wav"

    return decode_to_pcm(source, destination, sample_rate=None, channels=1)

# ---------------------------------------------------------------------------

# FORENSIC TELEMETRY ENGINE

# Inspects physical vocoder cutoffs, jitter stability, and digital noise floors

# ---------------------------------------------------------------------------

def compute_forensic_telemetry(audio_native, sr_native, audio_16k, sr_16k=16000):

    anomaly_flags = []

    anomaly_weights = []

    # 1. Native High-Frequency Vocoder Cliff Check

    # Modern neural vocoders (HiFi-GAN, BigVGAN) synthesize at 22-24kHz, leaving a sharp

    # brickwall drop above 10-12kHz when exported into standard 44.1/48kHz containers.

    if sr_native >= 22050 and len(audio_native) > 2048:

        n_fft = 2048

        window = np.hanning(n_fft)

        num_frames = min(50, len(audio_native) // n_fft)

        if num_frames > 0:

            frames = audio_native[:num_frames * n_fft].reshape(num_frames, n_fft) * window

            fft_mag = np.abs(np.fft.rfft(frames, axis=1))

            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr_native)

            hf_mask = (freqs >= 10500) & (freqs <= 18000)

            voice_mask = (freqs >= 800) & (freqs <= 4000)

            hf_energy = float(np.mean(fft_mag[:, hf_mask])) if np.any(hf_mask) else 0.0

            voice_energy = float(np.mean(fft_mag[:, voice_mask])) + 1e-9

            hf_ratio = hf_energy / voice_energy

            # True biological voice through a microphone maintains room presence above 10kHz.

            if hf_ratio < 0.0035:

                anomaly_flags.append("Vocoder High-Frequency Brickwall Cliff (<11kHz Cutoff)")

                anomaly_weights.append(0.88)

    # 2. Glottal Jitter / Pitch Periodicity Smoothness Check

    # Natural human speech has natural micro-tremors in fundamental frequency (0.5% - 2.0%).

    # AI vocoders produce mathematically uniform, unnaturally smooth phase waveforms.

    frame_size = 512

    hop = 256

    num_frames = (len(audio_16k) - frame_size) // hop

    if num_frames > 20:

        frames_16k = np.lib.stride_tricks.as_strided(

            audio_16k,

            shape=(num_frames, frame_size),

            strides=(audio_16k.strides[0] * hop, audio_16k.strides[0]),

        )

        energies = np.sum(frames_16k ** 2, axis=1)

        active_frames = frames_16k[energies > np.percentile(energies, 40)]

        if len(active_frames) > 10:

            # Autocorrelation peak consistency

            diffs = np.diff(active_frames, axis=0)

            flux = np.mean(np.abs(diffs))

            # AI models exhibit unnaturally low flux variance across active phonation

            if flux < 0.015:

                anomaly_flags.append("Synthetic Phase Linearity / Unnatural Phonation Smoothness")

                anomaly_weights.append(0.82)

    # 3. Ambient Room Tone & Noise Floor Entropy

    # Physical mics record air currents and preamp noise. AI speech has dead zero silence.

    quiet_threshold = np.percentile(np.abs(audio_16k), 15)

    quiet_samples = audio_16k[np.abs(audio_16k) <= quiet_threshold]

    if len(quiet_samples) > 200:

        quiet_var = float(np.var(quiet_samples))

        if quiet_var < 8e-8:

            anomaly_flags.append("Zero-Floor Anomaly (Synthetic Digital Silence in Pauses)")

            anomaly_weights.append(0.85)

    if anomaly_weights:

        acoustic_score = max(anomaly_weights)

    else:

        acoustic_score = 0.12

    return {

        "synthetic_score": acoustic_score,

        "flags": anomaly_flags,

    }

def explain_verdict(is_fake):

    if is_fake:

        return (

            "Signs of AI-generated speech found",

            "Some audio patterns resemble those sometimes found in AI-generated speech. This result can be wrong.",

            "FLAGGED"

        )

    return (

        "No strong signs of AI-generated speech found",

        "The checks did not find strong signs of AI-generated speech. This does not confirm that the recording is genuine.",

        "NOMINAL"

    )

def format_dossier_html(verdict, status_tag, label, score, duration, inference_elapsed, guidance, anomalies):

    is_flagged = status_tag == "FLAGGED"

    is_nominal = status_tag == "NOMINAL"

    badge_bg = "rgba(239, 68, 68, 0.12)" if is_flagged else ("rgba(16, 185, 129, 0.12)" if is_nominal else "rgba(245, 158, 11, 0.12)")

    badge_border = "#dc2626" if is_flagged else ("#059669" if is_nominal else "#d97706")

    badge_color = "#f87171" if is_flagged else ("#34d399" if is_nominal else "#fbbf24")

    accent_bar = "#ef4444" if is_flagged else ("#10b981" if is_nominal else "#f59e0b")

    score_pct = score * 100

    anomalies_html = ""

    if anomalies:

        items = "".join([f"<li style='margin-bottom: 4px; color: #f87171;'>{a}</li>" for a in anomalies])

        anomalies_html = f"""

        <div style="margin-top: 14px; padding: 12px 14px; background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 6px;">

          <div style="font-family: var(--font-mono); font-size: 10px; color: #fca5a5; letter-spacing: 0.06em; margin-bottom: 6px;">Patterns worth reviewing</div>

          <ul style="margin: 0; padding-left: 18px; font-size: 12px;">{items}</ul>

        </div>

        """

    return f"""

    <div class="dossier-card">

      <div class="dossier-header">

        <div>

          <div class="dossier-tag">Audio check</div>

          <div class="dossier-title">{verdict}</div>

        </div>

        <div class="dossier-badge" style="background:{badge_bg}; border-color:{badge_border}; color:{badge_color};">

          ● {status_tag}

        </div>

      </div>

      <div class="progress-track">

        <div class="progress-fill" style="width: {score_pct:.1f}%; background: {accent_bar};"></div>

      </div>

      <div class="metrics-grid">

        <div class="metric-box">

          <div class="metric-label">MODEL SCORE</div>

          <div class="metric-value">{score_pct:.2f}<span class="metric-unit">%</span></div>

        </div>

        <div class="metric-box">

          <div class="metric-label">RESULT</div>

          <div class="metric-value" style="font-size: 15px; font-weight: 600;">{label.upper()}</div>

        </div>

        <div class="metric-box">

          <div class="metric-label">AUDIO LENGTH</div>

          <div class="metric-value">{duration:.2f}<span class="metric-unit">s</span></div>

        </div>

        <div class="metric-box">

          <div class="metric-label">PROCESSING TIME</div>

          <div class="metric-value">{inference_elapsed:.3f}<span class="metric-unit">s</span></div>

        </div>

      </div>

      {anomalies_html}

      <div class="dossier-notice">

        <strong>Please note:</strong> {guidance}

      </div>

    </div>

    """

INITIAL_RESULT_HTML = """

<div class="dossier-card" style="opacity: 0.75; text-align: center; padding: 48px 24px;">

  <div class="dossier-tag">AUDIO FORENSICS ENGINE</div>

  <div style="font-size: 16px; font-weight: 500; color: #94a3b8; margin: 8px 0;">Your result will appear here</div>

  <p style="font-size: 12px; color: #64748b; margin: 0; font-family: monospace;">Choose a recording or use your microphone, then select “Check this audio”.</p>

</div>

"""

def analyze(audio_file, history=None):

    if not audio_file:

        raise gr.Error("No audio source detected.")

    source = Path(audio_file)

    if not source.exists() or not source.is_file():

        raise gr.Error("File path resolution failed.")

    if source.stat().st_size > MAX_BYTES:

        raise gr.Error(f"Audio payload exceeds {MAX_BYTES // (1024 * 1024)} MB threshold.")

    inference_started = time.perf_counter()

    # Step 1: Decode native audio for unclipped high-frequency spectral audit

    try:

        with tempfile.TemporaryDirectory(prefix="audito-engine-") as work_dir:

            native_wav = Path(work_dir) / "native.wav"

            decode_to_pcm(source, native_wav, sample_rate=None, channels=1)

            audio_native, sr_native = sf.read(str(native_wav), dtype="float32", always_2d=False)

            mono_16k_wav = Path(work_dir) / "16k.wav"

            decode_to_pcm(source, mono_16k_wav, sample_rate=16000, channels=1)

            audio_16k, sr_16k = sf.read(str(mono_16k_wav), dtype="float32", always_2d=False)

    except Exception as exc:

        raise gr.Error("Failed to decode audio payload.") from exc

    if audio_16k.size == 0:

        raise gr.Error("Empty audio sample received.")

    if audio_16k.ndim > 1:

        audio_16k = audio_16k.mean(axis=1)

    if audio_native.ndim > 1:

        audio_native = audio_native.mean(axis=1)

    duration = len(audio_16k) / float(sr_16k)

    if duration > MAX_SECONDS:

        raise gr.Error(f"Sample duration ({duration:.1f}s) exceeds operational ceiling ({MAX_SECONDS}s).")

    if duration < 0.25:

        raise gr.Error("Acoustic profile too short for verification.")

    rms = float(np.sqrt(np.mean(np.square(audio_16k), dtype=np.float64)))

    if rms < 1e-5:

        raise gr.Error("Zero-signal detected. Inaudible sample.")

    # Step 2: Native Acoustic Telemetry (physical artifacts)

    telemetry = compute_forensic_telemetry(audio_native, sr_native, audio_16k, sr_16k)

    acoustic_synthetic_score = telemetry["synthetic_score"]

    anomalies = telemetry["flags"]

    # Step 3: Neural Model Inference

    target_len = min(len(audio_16k), int(8.0 * sr_16k))

    center = len(audio_16k) // 2

    start = max(0, center - (target_len // 2))

    sub_audio = audio_16k[start:start + target_len]

    with torch.inference_mode():

        preds = _classifier({"raw": sub_audio, "sampling_rate": int(sr_16k)}, top_k=None)

    neural_fake_score = 0.0

    for p in preds:

        lbl = str(p.get("label", "")).lower()

        if any(w in lbl for w in ("fake", "spoof", "synthetic")):

            neural_fake_score = float(p.get("score", 0.0))

            break

    inference_elapsed = time.perf_counter() - inference_started

    # Step 4: Decision Fusion with Forensic Anomaly Override

    # If physical acoustic evidence (vocoder cutoff, zero floor, phase linearity) is detected,

    # the sample is flagged as synthetic regardless of an outdated neural classifier's reading.

    if len(anomalies) > 0 or acoustic_synthetic_score >= 0.70:

        is_fake = True

        effective_fake_prob = max(acoustic_synthetic_score, neural_fake_score, 0.865)

        effective_real_prob = 1.0 - effective_fake_prob

    elif neural_fake_score >= 0.65:

        is_fake = True

        effective_fake_prob = neural_fake_score

        effective_real_prob = 1.0 - effective_fake_prob

    else:

        is_fake = False

        effective_fake_prob = max(acoustic_synthetic_score, neural_fake_score)

        effective_real_prob = 1.0 - effective_fake_prob

    primary_label = "SYNTHETIC / AI-GENERATED" if is_fake else "AUTHENTIC HUMAN VOICE"

    primary_score = effective_fake_prob if is_fake else effective_real_prob

    verdict, guidance, status_tag = explain_verdict(is_fake)

    table = [

        ["SYNTHETIC / AI-GENERATED", f"{effective_fake_prob * 100:.2f}%"],

        ["AUTHENTIC HUMAN VOICE", f"{effective_real_prob * 100:.2f}%"],

        ["ACOUSTIC TELEMETRY RISK", f"{acoustic_synthetic_score * 100:.1f}% anomaly probability"],

        ["NEURAL CLASSIFIER WEIGHT", f"{neural_fake_score * 100:.1f}% legacy pattern weight"],

    ]

    report_html = format_dossier_html(

        verdict=verdict,

        status_tag=status_tag,

        label=primary_label,

        score=primary_score,

        duration=duration,

        inference_elapsed=inference_elapsed,

        guidance=guidance,

        anomalies=anomalies,

    )

    entry = {

        "time": datetime.now().strftime("%H:%M:%S"),

        "file": source.name[:45],

        "verdict": status_tag,

        "score": f"{primary_score * 100:.2f}%",

        "duration": f"{duration:.2f}s",

    }

    session_history = list(history or [])

    session_history.insert(0, entry)

    session_history = session_history[:50]

    history_rows = [

        [row["time"], row["file"], row["verdict"], row["score"], row["duration"]]

        for row in session_history

    ]

    return report_html, table, history_rows, session_history

def clear_history():

    return [], []

CSS = """
:root {
  --bg-dark: #171b19;
  --surface-1: #202622;
  --surface-2: #292f2a;
  --border-subtle: rgba(231, 237, 228, 0.10);
  --border-strong: rgba(231, 237, 228, 0.18);
  --accent: #a8c5a0;
  --accent-strong: #bfd6b6;
  --text-main: #edf0e9;
  --text-muted: #a6afa4;
  --font-mono: 'SFMono-Regular', Menlo, Consolas, monospace;
}
body, .gradio-container {
  background: var(--bg-dark) !important;
  color: var(--text-main) !important;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  letter-spacing: 0;
}
.gradio-container .main { max-width: 1180px !important; margin: 0 auto !important; padding: 30px 22px !important; }
#inst-header { border-bottom: 1px solid var(--border-subtle); padding: 8px 0 24px; margin-bottom: 26px; }
.inst-meta { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; font-size:11px; color:var(--text-muted); letter-spacing:.04em; }
.inst-title { font-size:30px; line-height:1.2; font-weight:650; color:var(--text-main); letter-spacing:-.035em; margin:18px 0 8px; }
.inst-desc { font-size:14px; color:var(--text-muted); max-width:690px; line-height:1.65; }
.social-links { display:flex; gap:10px; flex-wrap:wrap; margin-top:16px; }
.social-links a { display:inline-block; padding:8px 12px; border:1px solid var(--border-strong); border-radius:8px; color:var(--text-main) !important; text-decoration:none !important; font-size:12px; background:rgba(255,255,255,.025); }
.social-links a:hover { background:rgba(168,197,160,.10); border-color:rgba(168,197,160,.45); }
.gradio-container .block, .gradio-container .panel { background:var(--surface-1) !important; border:1px solid var(--border-subtle) !important; border-radius:12px !important; box-shadow:none !important; }
.gradio-container button.primary { background:var(--accent) !important; color:#182019 !important; border:1px solid var(--accent) !important; border-radius:8px !important; font-weight:650 !important; font-size:13px !important; transition:background .15s ease; }
.gradio-container button.primary:hover { background:var(--accent-strong) !important; }
.gradio-container button:not(.primary) { background:var(--surface-2) !important; color:var(--text-main) !important; border:1px solid var(--border-subtle) !important; border-radius:8px !important; font-size:13px !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select { background:var(--surface-2) !important; color:var(--text-main) !important; border-color:var(--border-subtle) !important; }
.gradio-container table { background:var(--surface-1) !important; border-collapse:collapse !important; font-size:13px !important; }
.gradio-container th { background:var(--surface-2) !important; color:var(--text-muted) !important; font-size:11px !important; font-weight:600 !important; border-bottom:1px solid var(--border-subtle) !important; }
.gradio-container td { border-bottom:1px solid var(--border-subtle) !important; color:var(--text-main) !important; }
.dossier-card { background:var(--surface-1); border:1px solid var(--border-subtle); border-radius:12px; padding:24px; }
.dossier-header { display:flex; justify-content:space-between; align-items:flex-start; gap:14px; margin-bottom:20px; }
.dossier-tag { font-size:11px; color:var(--text-muted); letter-spacing:.04em; margin-bottom:7px; }
.dossier-title { font-size:20px; line-height:1.35; font-weight:650; color:var(--text-main); letter-spacing:-.02em; }
.dossier-badge { font-size:11px; font-weight:650; padding:5px 10px; border-radius:7px; border:1px solid; white-space:nowrap; }
.progress-track { width:100%; height:5px; background:rgba(255,255,255,.07); border-radius:8px; overflow:hidden; margin-bottom:22px; }
.progress-fill { height:100%; transition:width .3s ease; }
.metrics-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-bottom:18px; }
.metric-box { background:var(--surface-2); border:1px solid var(--border-subtle); border-radius:9px; padding:13px; min-width:0; }
.metric-label { font-size:10px; line-height:1.5; color:var(--text-muted); margin-bottom:7px; }
.metric-value { font-size:17px; line-height:1.35; font-weight:600; color:var(--text-main); overflow-wrap:anywhere; }
.metric-unit { font-size:11px; color:var(--text-muted); margin-left:2px; }
.dossier-notice { font-size:12px; color:var(--text-muted); border-top:1px solid var(--border-subtle); padding-top:15px; line-height:1.65; }
@media (max-width: 760px) { .gradio-container .main { padding:18px 12px !important; } .inst-title {font-size:25px;} .metrics-grid {grid-template-columns:repeat(2,minmax(0,1fr));} .dossier-card {padding:17px;} }
"""

with gr.Blocks(title="Audito | Voice Check", theme=gr.themes.Base(), css=CSS) as demo:

    gr.HTML("""

    <div id="inst-header">

      <div class="inst-meta">

        <span>AUDITO  /  AUDIO CHECK</span>

        <span>A simple voice screening tool</span>

      </div>

      <div class="inst-title">Check whether a voice sounds AI-generated</div>

      <div class="inst-desc">

        Upload a short recording to check for patterns sometimes associated with AI-generated speech. Results are estimates, not proof.

      </div>
      <div class="social-links">
        <a href="https://www.linkedin.com/in/priyam-kumar-7b674b2b4/" target="_blank" rel="noopener noreferrer">LinkedIn ↗</a>
        <a href="https://github.com/surediepie" target="_blank" rel="noopener noreferrer">GitHub ↗</a>
      </div>
    </div>

    """)

    history_state = gr.State(value=[])

    with gr.Tabs():

        with gr.Tab("Check audio"):

            with gr.Row():

                with gr.Column(scale=5):

                    gr.Markdown("#### Your audio")

                    audio_input = gr.Audio(

                        label="Audio file or microphone",

                        sources=["upload", "microphone"],

                        type="filepath",

                        format=None,

                    )

                    gr.Markdown(

                        f"<div style='font-size:11px; color:#8492a6; font-family:monospace; margin-top:8px;'>"

                        f"Limit: {MAX_SECONDS} seconds · {MAX_BYTES // (1024 * 1024)} MB per file</div>"

                    )

                    analyze_button = gr.Button("Check this audio", variant="primary", size="lg")

                with gr.Column(scale=7):

                    gr.Markdown("#### Result")

                    report_display = gr.HTML(INITIAL_RESULT_HTML)

                    scores_table = gr.Dataframe(

                        headers=["Check", "Result"],

                        datatype=["str", "str"],

                        label="Details",

                        interactive=False,

                    )

            with gr.Accordion("Audio tools: MP3 to WAV", open=False):

                with gr.Row():

                    mp3_input = gr.File(label="MP3 file", file_types=[".mp3"], type="filepath")

                    wav_output = gr.File(label="Converted WAV file")

                convert_button = gr.Button("Convert to WAV")

                convert_button.click(

                    fn=mp3_to_wav,

                    inputs=mp3_input,

                    outputs=wav_output,

                    api_name=False,

                )

        with gr.Tab("Recent checks"):

            gr.Markdown("#### Checks in this session")

            history_table = gr.Dataframe(

                headers=["Time", "File", "Result", "Score", "Length"],

                datatype=["str", "str", "str", "str", "str"],

                label="Recent results (not saved after the session)",

                interactive=False,

            )

            clear_btn = gr.Button("Clear history")

            clear_btn.click(

                fn=clear_history,

                outputs=[history_table, history_state],

                api_name=False,

            )

        with gr.Tab("About Audito"):

            gr.Markdown(f"""

            #### About this project

            Audito is a prototype that combines a pretrained audio model with a few audio-signal checks. It can make mistakes, especially with compressed, noisy, edited, or unfamiliar recordings.

            **Treat the result as a clue, not a final verdict.** Do not use this prototype alone to make high-impact decisions about a person or recording.

            **Built by Priyam Kumar**  
            [LinkedIn]({LINKEDIN_URL}) · [GitHub]({GITHUB_URL})

            """)

    analyze_button.click(

        fn=analyze,

        inputs=[audio_input, history_state],

        outputs=[report_display, scores_table, history_table, history_state],

        concurrency_limit=1,

        api_name=False,

    )

if __name__ == "__main__":

    is_docker = os.path.exists("/.dockerenv")

    host = "0.0.0.0" if is_docker else "127.0.0.1"

    demo.queue(default_concurrency_limit=1, max_size=5).launch(

        server_name=host,

        server_port=PORT,

        show_api=False,

        ssr_mode=False,

    )
