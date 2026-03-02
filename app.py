from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import os
from pathlib import Path
import requests

app = Flask(__name__)
CORS(app)

# =========================
# MODEL LOADING (Railway-safe)
# =========================
MODEL_PATH = Path(os.getenv("MODEL_PATH", "glucose_model_v3.pkl"))
MODEL_URL = os.getenv("MODEL_URL")  # WAJIB diset di Railway (link direct download model)


def is_lfs_pointer(path: Path) -> bool:
    """
    Detect apakah file itu pointer Git LFS (bukan file biner model).
    Pointer biasanya berisi teks: 'version https://git-lfs.github.com/spec/v1'
    """
    try:
        if not path.exists():
            return False
        with open(path, "rb") as f:
            head = f.read(120)
        return b"git-lfs.github.com/spec" in head
    except Exception:
        return False


def ensure_model_file():
    """
    Pastikan file model beneran tersedia:
    - Kalau file tidak ada: download dari MODEL_URL
    - Kalau file masih pointer LFS: download dari MODEL_URL
    - Kalau file ada dan ukurannya masuk akal dan bukan pointer: skip
    """
    if MODEL_PATH.exists():
        size = MODEL_PATH.stat().st_size
        if size > 10_000_000 and not is_lfs_pointer(MODEL_PATH):
            # Sudah file biner model asli (misal >10MB), tidak perlu download
            print(f"✅ Model file exists: {MODEL_PATH} ({size} bytes)")
            return
        else:
            print(
                f"⚠️ Model file seems invalid/pointer: {MODEL_PATH} "
                f"({size} bytes, lfs_pointer={is_lfs_pointer(MODEL_PATH)})"
            )

    if not MODEL_URL:
        raise RuntimeError(
            "MODEL_URL belum diset. Set Railway Variables: MODEL_URL = direct download link model .pkl"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇️ Downloading model from MODEL_URL -> {MODEL_PATH}")

    r = requests.get(MODEL_URL, stream=True, timeout=300)
    r.raise_for_status()

    tmp_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    tmp_size = tmp_path.stat().st_size
    tmp_path.replace(MODEL_PATH)
    print(f"✅ Model downloaded: {MODEL_PATH} ({tmp_size} bytes)")


# Load model saat startup
try:
    ensure_model_file()
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# =========================
# HISTORY SETTINGS
# =========================
user_histories = defaultdict(lambda: deque(maxlen=10))
JUMLAH_SAMPEL = 10


# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "online",
            "message": "Glucose Prediction API",
            "version": "1.0",
            "model_path": str(MODEL_PATH),
            "model_loaded": model is not None,
            "endpoints": {
                "predict": "/predict [POST]",
                "health": "/health [GET]",
                "clear": "/clear/<user_id> [DELETE]",
                "status": "/status/<user_id> [GET]",
            },
        }
    ), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
        }
    ), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json(silent=True) or {}

        # Validasi input
        required_fields = ["user_id", "ir", "red", "bpm"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"{field} is required"}), 400

        user_id = str(data["user_id"])
        ir = float(data["ir"])
        red = float(data["red"])
        bpm = float(data["bpm"])

        # Validasi nilai
        if ir <= 0 or red <= 0 or bpm <= 0:
            return jsonify({"error": "Invalid sensor values"}), 400

        # Prediksi (samakan nama kolom dengan model saat training)
        input_df = pd.DataFrame(
            [
                {
                    "IR": ir,
                    "RED": red,
                    "BPM": bpm,
                }
            ]
        )

        prediction = model.predict(input_df)[0]

        # Tambahkan ke histori user
        user_histories[user_id].append(float(prediction))
        current_count = len(user_histories[user_id])

        response = {
            "user_id": user_id,
            "prediction": float(prediction),
            "sample_count": current_count,
            "max_samples": JUMLAH_SAMPEL,
            "ready": False,
        }

        # Jika sudah 10 sampel, hitung rata-rata
        if current_count == JUMLAH_SAMPEL:
            average = int(round(np.mean(list(user_histories[user_id]))))
            response["final_prediction"] = average
            response["ready"] = True

            # Tentukan status
            if average < 70:
                response["status"] = "Low"
                response["status_id"] = "Rendah"
            elif average <= 140:
                response["status"] = "Normal"
                response["status_id"] = "Normal"
            else:
                response["status"] = "High"
                response["status_id"] = "Cukup Tinggi"

            # Clear histori setelah prediksi final
            user_histories[user_id].clear()

        return jsonify(response), 200

    except ValueError as e:
        return jsonify({"error": f"Invalid data type: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


@app.route("/clear/<user_id>", methods=["DELETE"])
def clear_history(user_id):
    """Manual clear histori untuk user tertentu"""
    user_id = str(user_id)
    if user_id in user_histories:
        user_histories[user_id].clear()
        return jsonify({"message": f"History cleared for {user_id}"}), 200
    return jsonify({"message": "No history found"}), 404


@app.route("/status/<user_id>", methods=["GET"])
def get_status(user_id):
    """Cek status histori user"""
    user_id = str(user_id)
    count = len(user_histories.get(user_id, []))
    return jsonify(
        {
            "user_id": user_id,
            "sample_count": count,
            "max_samples": JUMLAH_SAMPEL,
            "samples_needed": JUMLAH_SAMPEL - count,
        }
    ), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)