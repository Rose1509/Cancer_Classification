from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

from inference import (
    CLASS_NAMES,
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    get_device,
    load_models,
    run_inference,
)

BASE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

models, device = load_models(BASE_DIR)
if not models:
    raise FileNotFoundError(
        "No model weights found. Place .pth files in the project root "
        "(best_baseline_model.pth, best_improved_model.pth, resnet18_melanoma_best.pth)."
    )

print(f"Using device: {device}")
print(f"Loaded models: {', '.join(models.keys())}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template(
        "home.html",
        models=list(MODEL_REGISTRY.keys()),
        loaded_models=list(models.keys()),
        default_model=DEFAULT_MODEL if DEFAULT_MODEL in models else next(iter(models)),
        class_names=CLASS_NAMES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    model_key = request.form.get("model", DEFAULT_MODEL)
    if model_key not in MODEL_REGISTRY:
        return jsonify({"error": f"Unknown model '{model_key}'."}), 400
    if model_key not in models:
        return jsonify({"error": f"Model '{model_key}' is not available (weights file missing)."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, JPEG, GIF, BMP, or WEBP."}), 400

    try:
        image = Image.open(file.stream)
        image.load()
    except Exception:
        return jsonify({"error": "Could not read image. Upload a valid image file."}), 400

    try:
        result = run_inference(model_key, models[model_key], image, device)
    except Exception as exc:
        return jsonify({"error": f"Inference failed: {exc}"}), 500

    return jsonify({
        "model": model_key,
        "label": result["label"],
        "confidence": result["confidence"],
        "malignant_probability": result["malignant_probability"],
        "filename": secure_filename(file.filename),
        "class_names": CLASS_NAMES,
    })


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": "File too large. Maximum size is 50MB."}), 413


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
