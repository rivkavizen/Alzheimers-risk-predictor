from pathlib import Path

from flask import Flask, jsonify, send_from_directory

from config import FRONTEND_DIR
from routes.predict import predict_bp
from services.ml_service import load_model

app = Flask(__name__, static_folder=None)
app.register_blueprint(predict_bp)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def serve_frontend(path):
    if not FRONTEND_DIR.exists():
        return jsonify({"message": "Frontend not yet built. API is available at /api/health"}), 200
    file_path = Path(path)
    if file_path.suffix and (FRONTEND_DIR / path).is_file():
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


load_model()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
