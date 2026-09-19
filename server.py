import argparse
import os
import socket
import tempfile

from flask import Flask, jsonify, request, send_from_directory

from analyze import IMAGE_EXT, VIDEO_EXT, analyze_file, compare_faces
from liveness import OnnxFaceClassifier

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB
CLASSIFIERS = []


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.post("/api/analyze")
def api_analyze():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in IMAGE_EXT | VIDEO_EXT:
        return jsonify({"error": f"Unsupported file type '{ext}'"}), 400

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        file.save(tmp.name)
        path = tmp.name
    try:
        result = analyze_file(path, CLASSIFIERS)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        os.remove(path)
    return jsonify(result)


@app.post("/api/compare")
def api_compare():
    file_a = request.files.get("file_a")
    file_b = request.files.get("file_b")
    if not file_a or not file_a.filename or not file_b or not file_b.filename:
        return jsonify({"error": "Two files are required (file_a and file_b)"}), 400

    paths = []
    try:
        for f in (file_a, file_b):
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in IMAGE_EXT | VIDEO_EXT:
                return jsonify({"error": f"Unsupported file type '{ext}'"}), 400
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                f.save(tmp.name)
                paths.append(tmp.name)
        result = compare_faces(*paths)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        for p in paths:
            os.remove(p)
    return jsonify(result)


def find_free_port(host: str, preferred: int, tries: int = 20) -> int:
    """Return `preferred` if it's free, otherwise the next open port after it."""
    for port in range(preferred, preferred + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port found in {preferred}-{preferred + tries - 1}")


def main():
    parser = argparse.ArgumentParser(description="Deepfake/liveness analysis web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--spoof-model", help="ONNX live/spoof model from classifier.py")
    parser.add_argument("--deepfake-model", help="ONNX real/fake model from classifier.py")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.spoof_model:
        CLASSIFIERS.append(OnnxFaceClassifier(args.spoof_model, "anti-spoof"))
    if args.deepfake_model:
        CLASSIFIERS.append(OnnxFaceClassifier(args.deepfake_model, "deepfake"))

    port = find_free_port(args.host, args.port)
    if port != args.port:
        print(f"Port {args.port} is busy -- using {port} instead.")
    print(f" * Open http://{args.host}:{port}")
    app.run(host=args.host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
