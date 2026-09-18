import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")

# ImageNet normalisation - used in training AND inference, keep them identical.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CROP_SIZE = 224

# MediaPipe face-mesh landmark indices
NOSE_TIP = 1
CHEEK_A = 234
CHEEK_B = 454


def ensure_landmarker_model(path: str = MODEL_PATH) -> str:
    """Download the Face Landmarker model the first time it's needed."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading face landmarker model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def create_landmarker(video: bool = True, max_faces: int = 2):
    """Create a Face Landmarker. max_faces=2 lets us detect 'extra face' cheating."""
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=ensure_landmarker_model(),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_faces=max_faces,
        output_face_blendshapes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def to_mp_image(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def face_bbox(landmarks, width: int, height: int, margin: float = 0.25):
    """Square bounding box around the landmarks, expanded by `margin`, clipped to the frame."""
    xs = [p.x * width for p in landmarks]
    ys = [p.y * height for p in landmarks]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + margin)
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(width, cx + side / 2)), int(min(height, cy + side / 2))
    return x1, y1, x2, y2


def crop_face(bgr, landmarks, size: int = CROP_SIZE):
    """Return a size x size BGR face crop, or None if the box is degenerate."""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = face_bbox(landmarks, w, h)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return cv2.resize(bgr[y1:y2, x1:x2], (size, size))


def blendshapes(result, face_index: int = 0) -> dict:
    """{'eyeBlinkLeft': 0.03, 'mouthSmileLeft': 0.8, ...} for one detected face."""
    return {c.category_name: c.score for c in result.face_blendshapes[face_index]}


def yaw_ratio(landmarks) -> float:
    """Where the nose sits between the two cheeks horizontally.

    ~0.5 = facing the camera. On a MIRRORED (selfie-view) frame, turning your head
    to your left pushes this towards 0 and turning right pushes it towards 1.
    """
    nose = landmarks[NOSE_TIP].x
    a, b = landmarks[CHEEK_A].x, landmarks[CHEEK_B].x
    lo, hi = min(a, b), max(a, b)
    if hi - lo < 1e-6:
        return 0.5
    return (nose - lo) / (hi - lo)