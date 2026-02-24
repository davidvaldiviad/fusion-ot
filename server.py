from io import BytesIO
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")  # backend non-interactif pour Flask
import matplotlib.pyplot as plt

from flask import Flask, request, send_file, jsonify, send_from_directory

# Adapte ces imports selon ton arborescence réelle
# Si tes fichiers sont dans src/, garde ces imports:
from src.spectrogram import Spectrogram
from src.display import display_spectrogram

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"   # si dumb.html/app.js/style.css sont dans docs/
# sinon mets BASE_DIR directement si les fichiers sont à la racine

# Si tu testes avec les fichiers uploadés actuels à plat, remplace par:
# from spectrogram import Spectrogram
# from display import display_spectrogram




app = Flask(__name__)

# Dossier des sons de la librairie
EXAMPLE_SOUNDS_DIR = Path("example_sounds")

# Liste blanche simple (ids/nom de fichier autorisés)
# Tu peux la remplir manuellement, ou la générer dynamiquement.
def allowed_audio_names():
    if not EXAMPLE_SOUNDS_DIR.exists():
        return set()
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    return {p.name for p in EXAMPLE_SOUNDS_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts}


@app.post("/api/spectrogram")
def api_spectrogram():
    try:
        data = request.get_json(force=True)

        audio_name = data["audio_name"]
        sr = int(data["sr"])
        win_sec = float(data["win_sec"])
        hop_sec = float(data["hop_sec"])
        nfft = int(data["nfft"])

        # validations minimales
        if sr <= 0 or win_sec <= 0 or hop_sec <= 0 or nfft <= 0:
            return jsonify({"error": "Invalid parameters"}), 400

        if hop_sec > win_sec:
            return jsonify({"error": "Hop size must be <= window size"}), 400

        # nfft en samples vs window en sec -> comparer avec window_samples
        window_samples = int(win_sec * sr)
        if window_samples < 1:
            return jsonify({"error": "Window too small for this sample rate"}), 400
        if nfft < window_samples:
            return jsonify({"error": "NFFT must be >= window size (in samples)"}), 400

        allowed = allowed_audio_names()
        if audio_name not in allowed:
            return jsonify({"error": "Audio not allowed"}), 400

        audio_path = EXAMPLE_SOUNDS_DIR / audio_name

        signal, _ = librosa.load(audio_path, sr=sr)

        X = Spectrogram(
            signal,
            sr,
            win_sec,
            hop_size_s=hop_sec,
            nfft=nfft,
        )

        P = X.power_spectrogram()

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        display_spectrogram(
            P,
            ax=ax,
            scale="log",
            f_bins=X.f_bins,
            t_bins=X.t_bins,
        )

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        return send_file(buf, mimetype="image/png")

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.get("/")
def index():
    return send_from_directory(DOCS_DIR, "dumb.html")

@app.get("/app.js")
def js_file():
    return send_from_directory(DOCS_DIR, "app.js")

@app.get("/style.css")
def css_file():
    return send_from_directory(DOCS_DIR, "style.css")

@app.get("/example_sounds/<path:filename>")
def example_sound(filename):
    return send_from_directory(EXAMPLE_SOUNDS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)