import io
import json
import os
import sys
import types
from contextlib import redirect_stdout
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import liveness_check as lc

WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_testwork")
failures = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {extra}")
    if not cond:
        failures.append(name)


def landmarks_centre():
    """455 landmarks: face fills the middle, nose centred between the cheeks."""
    lm = [types.SimpleNamespace(x=0.5, y=0.5) for _ in range(455)]
    lm[0] = types.SimpleNamespace(x=0.40, y=0.40)   # spread the bbox out
    lm[10] = types.SimpleNamespace(x=0.60, y=0.60)
    lm[1] = types.SimpleNamespace(x=0.50, y=0.50)   # nose tip
    lm[234] = types.SimpleNamespace(x=0.40, y=0.50)  # cheek A
    lm[454] = types.SimpleNamespace(x=0.60, y=0.50)  # cheek B
    return lm


class FakeCap:
    def __init__(self, frames=400):
        self.left = frames

    def isOpened(self):
        return True

    def read(self):
        self.left -= 1
        if self.left < 0:
            return False, None
        return True, np.full((480, 640, 3), 120, dtype=np.uint8)

    def release(self):
        pass


class FakeLandmarker:
    """Returns one face every frame, with blendshapes taken from `script`."""

    def __init__(self, script):
        self.script = script
        self.i = -1

    def detect_for_video(self, image, ts):
        self.i += 1
        bs = self.script(self.i)
        cats = [types.SimpleNamespace(category_name=k, score=v) for k, v in bs.items()]
        return types.SimpleNamespace(face_landmarks=[landmarks_centre()], face_blendshapes=[cats])

    def close(self):
        pass


def run(argv, script, challenge="blink"):
    """Run main() with everything external faked; returns the printed JSON report."""
    buf = io.StringIO()
    with mock.patch.object(lc.cv2, "VideoCapture", lambda *a, **k: FakeCap()), \
         mock.patch.object(lc.cv2, "imshow", lambda *a, **k: None), \
         mock.patch.object(lc.cv2, "waitKey", lambda *a, **k: -1), \
         mock.patch.object(lc.cv2, "destroyAllWindows", lambda: None), \
         mock.patch.object(lc, "create_landmarker", lambda **k: FakeLandmarker(script)), \
         mock.patch.object(lc.random, "sample", lambda pop, k: [challenge]), \
         mock.patch.object(sys, "argv", ["liveness_check.py"] + argv):
        with redirect_stdout(buf):
            lc.main()
    out = buf.getvalue()
    return json.loads(out[out.index("{"):out.rindex("}") + 1])


def blinking(i):
    # closed for 3 frames, open for 3, repeated -> counts blinks
    phase = (i // 3) % 2
    s = 0.9 if phase == 0 else 0.05
    return {"eyeBlinkLeft": s, "eyeBlinkRight": s}


r = run(["--num-challenges", "1", "--timeout", "30"], blinking)
check("blink challenge passes", r["passed"] is True, r["reason"])
check("reports 1 challenge completed", r["challenges_completed"] == 1, str(r))
check("no model scores when no models", r["model_scores"] == {}, str(r["model_scores"]))

r = run(["--num-challenges", "1", "--timeout", "0.05"],
        lambda i: {"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0})
check("no blink -> fails", r["passed"] is False)
check("failure reason is a timeout", "timed out" in r["reason"], r["reason"])
r = run(["--num-challenges", "1", "--timeout", "30"],
        lambda i: {"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9}, challenge="smile")
check("smile challenge passes", r["passed"] is True, r["reason"])

onnx_path = os.path.join(WORK, "spoof.onnx")
if os.path.exists(onnx_path):
    r = run(["--num-challenges", "1", "--timeout", "30", "--spoof-model", onnx_path,
             "--threshold", "0.99", "--score-every", "1"], blinking)
    check("high threshold rejects", r["passed"] is False, r["reason"])
    check("reason names the model", "anti-spoof" in r["reason"], r["reason"])
    check("model score recorded", "anti-spoof" in r["model_scores"], str(r["model_scores"]))

    # Same run, threshold of 0 -> the gate lets it through
    r = run(["--num-challenges", "1", "--timeout", "30", "--spoof-model", onnx_path,
             "--threshold", "0.0", "--score-every", "1"], blinking)
    check("zero threshold accepts", r["passed"] is True, r["reason"])
else:
    check("spoof.onnx available from previous test", False)

print()
print(f"{'ALL LOOP TESTS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)