# Deepfake & Liveness Checker

A set of tools for checking whether a face in a photo, video, or webcam feed
is real — not a certified detector, just a heuristic-plus-optional-model
signal.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Web app

Drag and drop a photo or video (or record a short clip from your webcam) and
get a real/fake verdict with an explanation. Also includes an ID/selfie
face-match tool and a downloadable report.

```bash
python server.py
```

Open the URL it prints (defaults to `http://127.0.0.1:5000`; it auto-picks
the next free port if that one's busy). Optional flags:

```bash
python server.py --spoof-model models/spoof.onnx --deepfake-model models/deepfake.onnx
```

## CLI liveness check

Runs a random sequence of active challenges (blink, turn, smile...) over
your webcam — a pre-recorded video can't know the order in advance.

```bash
python liveness.py --num-challenges 3
```

Optional flags: `--camera`, `--timeout`, `--spoof-model`, `--deepfake-model`,
`--threshold`, `--debug`. Run `python liveness.py --help` for the full list.

## Analyzing a file from the command line

```bash
python analyze.py --input path/to/photo.jpg
```

## Training your own model

The web app and CLI tools work heuristic-only out of the box. For a real
trained classifier:

1. **Collect face crops** from a folder of labeled real/fake photos and
   videos:
   ```bash
   python extractface.py --input raw_data/ --output crops/
   ```
   (expects `raw_data/<class>/...`, where one class is named `live` or `real`)

2. **Train and export to ONNX**:
   ```bash
   python classifier.py --data crops/ --out models/deepfake.onnx
   ```

3. Point `server.py` or `liveness.py` at the resulting `.onnx` file with
   `--deepfake-model` (or `--spoof-model` for an anti-spoof model).

