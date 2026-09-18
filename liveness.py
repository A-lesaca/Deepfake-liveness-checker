"""Webcam liveness check.

1. Active: the user completes a random sequence of challenges (blink, turn, smile...).
   A pre-recorded video can't know the order in advance.
2. Passive (optional): ONNX classifiers score face crops during the session -
   an anti-spoof model (live vs spoof) and/or a deepfake model (real vs fake).
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np

from face_utils import (CROP_SIZE, MEAN, STD, blendshapes, create_landmarker,
                        crop_face, face_bbox, to_mp_image, yaw_ratio)

GENUINE_CLASS_NAMES = {"live", "real"}

class BlinkChallenge:
    """Counts closed -> open transitions. Hysteresis (0.5 / 0.25) stops flicker double-counting."""

    def __init__(self, target: int = 2):
        self.prompt = f"Blink {target} times"
        self.target, self.count, self.closed = target, 0, False

    def update(self, bs: dict, yaw: float) -> bool:
        score = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
        if not self.closed and score > 0.5:
            self.closed = True
        elif self.closed and score < 0.25:
            self.closed = False
            self.count += 1
        return self.count >= self.target


class HoldChallenge:
    """Passes once a condition holds for `frames` consecutive frames (filters out noise)."""

    def __init__(self, prompt: str, condition, frames: int = 6):
        self.prompt, self.condition, self.frames, self.streak = prompt, condition, frames, 0

    def update(self, bs: dict, yaw: float) -> bool:
        self.streak = self.streak + 1 if self.condition(bs, yaw) else 0
        return self.streak >= self.frames


CHALLENGES = {
    "blink": lambda: BlinkChallenge(target=2),
    "turn_left": lambda: HoldChallenge("Turn your head LEFT", lambda bs, yaw: yaw < 0.30),
    "turn_right": lambda: HoldChallenge("Turn your head RIGHT", lambda bs, yaw: yaw > 0.70),
    "smile": lambda: HoldChallenge(
        "Smile", lambda bs, yaw: (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2 > 0.6),
    "open_mouth": lambda: HoldChallenge("Open your mouth", lambda bs, yaw: bs.get("jawOpen", 0) > 0.5),
}

class OnnxFaceClassifier:
    """Wraps an ONNX model exported by train_classifier.py. Returns P(genuine) for a face crop."""

    def __init__(self, path: str, name: str):
        import onnxruntime as ort

        self.name = name
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        classes_path = os.path.splitext(path)[0] + ".classes.json"
        with open(classes_path) as f:
            classes = json.load(f)
        matches = [i for i, c in enumerate(classes) if c.lower() in GENUINE_CLASS_NAMES]
        if not matches:
            raise ValueError(f"{classes_path} needs a 'live' or 'real' class, got {classes}")
        self.genuine_index = matches[0]
        self.scores = []

    def score(self, face_bgr) -> float:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        logits = self.session.run(None, {self.input_name: x})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        p = float(probs[self.genuine_index])
        self.scores.append(p)
        return p

    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0

def put(frame, text, y, color=(255, 255, 255), scale=0.7):
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Webcam liveness check")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--num-challenges", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=6.0, help="seconds allowed per challenge")
    parser.add_argument("--spoof-model", help="ONNX live/spoof model from train_classifier.py")
    parser.add_argument("--deepfake-model", help="ONNX real/fake model from train_classifier.py")
    parser.add_argument("--threshold", type=float, default=0.5, help="min mean P(genuine) per model")
    parser.add_argument("--score-every", type=int, default=3, help="run models every N frames")
    parser.add_argument("--debug", action="store_true", help="show raw yaw/blendshape values")
    args = parser.parse_args()

    classifiers = []
    if args.spoof_model:
        classifiers.append(OnnxFaceClassifier(args.spoof_model, "anti-spoof"))
    if args.deepfake_model:
        classifiers.append(OnnxFaceClassifier(args.deepfake_model, "deepfake"))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    landmarker = create_landmarker(video=True)

    names = random.sample(list(CHALLENGES), k=min(args.num_challenges, len(CHALLENGES)))
    challenges = [CHALLENGES[n]() for n in names]
    current = 0
    challenge_start = time.monotonic()
    last_face_seen = time.monotonic()
    face_ever_seen = False
    t0, last_ts, frame_no = time.monotonic(), -1, 0
    failure = None

    while current < len(challenges) and failure is None:
        ok, frame = cap.read()
        if not ok:
            failure = "camera stream ended"
            break
        frame = cv2.flip(frame, 1)  # selfie view, so "left" means the user's left
        clean = frame.copy()  # crop the models from this, BEFORE any overlay is drawn
        frame_no += 1

        # MediaPipe VIDEO mode needs strictly increasing timestamps
        ts = max(int((time.monotonic() - t0) * 1000), last_ts + 1)
        last_ts = ts
        result = landmarker.detect_for_video(to_mp_image(frame), ts)
        now = time.monotonic()
        challenge = challenges[current]
        faces = result.face_landmarks

        if len(faces) == 1:
            last_face_seen = now
            face_ever_seen = True
            lm = faces[0]
            bs = blendshapes(result)
            yaw = yaw_ratio(lm)

            if classifiers and frame_no % args.score_every == 0:
                crop = crop_face(clean, lm, CROP_SIZE)
                if crop is not None:
                    for clf in classifiers:
                        clf.score(crop)

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = face_bbox(lm, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

            if challenge.update(bs, yaw):
                current += 1
                challenge_start = now
                continue

            if args.debug:
                put(frame, f"yaw={yaw:.2f} blink={bs.get('eyeBlinkLeft', 0):.2f} "
                           f"smile={bs.get('mouthSmileLeft', 0):.2f} jaw={bs.get('jawOpen', 0):.2f}",
                    h - 20, (200, 200, 0), 0.55)
        elif len(faces) > 1:
            put(frame, "Only one face in view please", 110, (0, 0, 255))
        elif not face_ever_seen:
            put(frame, "Position your face in the frame", 110, (0, 200, 255))
        elif now - last_face_seen > 1.5:
            failure = "face left the frame during the check"

        # Don't start the per-challenge clock until the user has actually shown up
        if not face_ever_seen:
            challenge_start = now

        remaining = args.timeout - (now - challenge_start)
        if remaining <= 0:
            failure = f"timed out on '{challenge.prompt}'"

        put(frame, f"Step {current + 1}/{len(challenges)}: {challenge.prompt}", 35, (0, 255, 255), 0.8)
        put(frame, f"{max(0.0, remaining):.1f}s left", 70)
        for i, clf in enumerate(classifiers):
            put(frame, f"{clf.name}: {clf.mean_score():.2f}", 150 + 30 * i, (255, 200, 0), 0.6)

        cv2.imshow("Liveness check", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            failure = "cancelled by user"

    model_scores = {clf.name: round(clf.mean_score(), 3) for clf in classifiers}
    if failure is None:
        for clf in classifiers:
            if not clf.scores:
                failure = f"{clf.name} model never got a usable face crop"
            elif clf.mean_score() < args.threshold:
                failure = f"{clf.name} score {clf.mean_score():.2f} below threshold {args.threshold}"
    passed = failure is None

    report = {
        "passed": passed,
        "challenges": names,
        "challenges_completed": current,
        "model_scores": model_scores,
        "reason": failure or "all checks passed",
    }
    print(json.dumps(report, indent=2))

    # Show the verdict for a couple of seconds
    ok, frame = cap.read()
    if ok:
        frame = cv2.flip(frame, 1)
        put(frame, "LIVE - PASSED" if passed else "FAILED", 50,
            (0, 220, 0) if passed else (0, 0, 255), 1.2)
        if failure:
            put(frame, failure, 90, (255, 255, 255), 0.6)
        cv2.imshow("Liveness check", frame)
        cv2.waitKey(2500)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()