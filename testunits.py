import json
import os
import subprocess
import sys
import types

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_testwork")
os.makedirs(WORK, exist_ok=True)

failures = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {extra}")
    if not cond:
        failures.append(name)


from liveness_check import CHALLENGES, BlinkChallenge, HoldChallenge

b = BlinkChallenge(target=2)
# two clean blinks: closed (0.9) then open (0.1)
seq = [0.9, 0.9, 0.1, 0.1, 0.9, 0.1]
done = [b.update({"eyeBlinkLeft": s, "eyeBlinkRight": s}, 0.5) for s in seq]
check("blink counts 2 blinks", done[-1] and b.count == 2, f"count={b.count}")

# values stuck in the hysteresis dead zone must not count
b2 = BlinkChallenge(target=1)
for _ in range(20):
    b2.update({"eyeBlinkLeft": 0.35, "eyeBlinkRight": 0.35}, 0.5)
check("blink ignores dead-zone jitter", b2.count == 0, f"count={b2.count}")

h = HoldChallenge("turn", lambda bs, yaw: yaw < 0.3, frames=6)
res = [h.update({}, 0.2) for _ in range(6)]
check("hold passes after 6 frames", res[-1] and not res[4])

h2 = HoldChallenge("turn", lambda bs, yaw: yaw < 0.3, frames=6)
for yaw in [0.2, 0.2, 0.9, 0.2, 0.2, 0.2]:
    got = h2.update({}, yaw)
check("hold resets on interruption", not got, f"streak={h2.streak}")

check("5 challenges registered", len(CHALLENGES) == 5, str(list(CHALLENGES)))
for name, factory in CHALLENGES.items():
    c = factory()
    check(f"challenge '{name}' has prompt+update", bool(c.prompt) and callable(c.update))


from face_utils import CROP_SIZE, crop_face, face_bbox, yaw_ratio

P = lambda x, y: types.SimpleNamespace(x=x, y=y)

# a face occupying the middle of a 640x480 frame
lms = [P(0.4, 0.4), P(0.6, 0.4), P(0.5, 0.6), P(0.45, 0.5), P(0.55, 0.5)]
x1, y1, x2, y2 = face_bbox(lms, 640, 480)
check("bbox inside frame", 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480, f"{(x1, y1, x2, y2)}")
check("bbox is square-ish", abs((x2 - x1) - (y2 - y1)) <= 2, f"w={x2-x1} h={y2-y1}")

# landmarks partly off-frame must still clip cleanly
edge = [P(-0.3, -0.3), P(0.2, 0.2)]
ex1, ey1, ex2, ey2 = face_bbox(edge, 640, 480)
check("bbox clips off-frame landmarks", ex1 >= 0 and ey1 >= 0 and ex2 <= 640 and ey2 <= 480,
      f"{(ex1, ey1, ex2, ey2)}")

frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
crop = crop_face(frame, lms)
check("crop is 224x224x3", crop is not None and crop.shape == (CROP_SIZE, CROP_SIZE, 3),
      str(None if crop is None else crop.shape))

tiny = [P(0.5, 0.5), P(0.501, 0.501)]
check("crop rejects degenerate box", crop_face(frame, tiny) is None)

# yaw: nose at cheek_a -> 0, midway -> 0.5, at cheek_b -> 1
def yaw_with(nose_x):
    lm = [P(0, 0)] * 455
    lm[1] = P(nose_x, 0.5)
    lm[234] = P(0.3, 0.5)
    lm[454] = P(0.7, 0.5)
    return yaw_ratio(lm)

check("yaw centre ~0.5", abs(yaw_with(0.5) - 0.5) < 1e-6, f"{yaw_with(0.5):.3f}")
check("yaw left ~0.0", abs(yaw_with(0.3) - 0.0) < 1e-6, f"{yaw_with(0.3):.3f}")
check("yaw right ~1.0", abs(yaw_with(0.7) - 1.0) < 1e-6, f"{yaw_with(0.7):.3f}")

degenerate = [P(0, 0)] * 455
degenerate[1] = P(0.5, 0.5)
degenerate[234] = P(0.4, 0.5)
degenerate[454] = P(0.4, 0.5)
check("yaw handles zero-width face", yaw_ratio(degenerate) == 0.5)

data = os.path.join(WORK, "faces")
rng = np.random.default_rng(0)
for split, n in [("train", 12), ("val", 6)]:
    for cls, base in [("live", 40), ("spoof", 200)]:
        d = os.path.join(data, split, cls)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            img = np.clip(rng.normal(base, 20, (CROP_SIZE, CROP_SIZE, 3)), 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(d, f"{i}.jpg"), img)

onnx_path = os.path.join(WORK, "spoof.onnx")
cmd = [sys.executable, "train_classifier.py", "--data", data, "--out", onnx_path,
       "--epochs", "2", "--batch-size", "4", "--workers", "0",
       "--model", "mobilenetv3_small_050", "--no-pretrained"]
proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, timeout=900)
print("---- train stdout ----")
print(proc.stdout[-1800:])
if proc.returncode != 0:
    print("---- train stderr ----")
    print(proc.stderr[-2500:])

check("training script exits 0", proc.returncode == 0)
check("reports APCER/BPCER/ACER", all(k in proc.stdout for k in ("APCER", "BPCER", "ACER", "AUC")))
check("identifies genuine class", "genuine = live" in proc.stdout)
check("onnx file written", os.path.exists(onnx_path))
classes_json = os.path.splitext(onnx_path)[0] + ".classes.json"
check("classes.json written", os.path.exists(classes_json))
if os.path.exists(classes_json):
    with open(classes_json) as f:
        check("classes.json content", json.load(f) == ["live", "spoof"])


if os.path.exists(onnx_path) and os.path.exists(classes_json):
    from liveness_check import OnnxFaceClassifier

    clf = OnnxFaceClassifier(onnx_path, "anti-spoof")
    check("genuine index resolved to 'live'", clf.genuine_index == 0)
    check("mean_score empty is 0.0", clf.mean_score() == 0.0)

    face = np.full((CROP_SIZE, CROP_SIZE, 3), 40, dtype=np.uint8)
    p = clf.score(face)
    check("score in [0,1]", 0.0 <= p <= 1.0, f"p={p:.4f}")
    clf.score(np.full((CROP_SIZE, CROP_SIZE, 3), 200, dtype=np.uint8))
    check("mean over 2 scores", len(clf.scores) == 2 and 0.0 <= clf.mean_score() <= 1.0,
          f"mean={clf.mean_score():.4f}")

    # a model whose classes.json has no live/real class must be rejected loudly
    bad = os.path.join(WORK, "bad.onnx")
    import shutil
    shutil.copy(onnx_path, bad)
    with open(os.path.splitext(bad)[0] + ".classes.json", "w") as f:
        json.dump(["catA", "catB"], f)
    try:
        OnnxFaceClassifier(bad, "bad")
        check("rejects classes.json without live/real", False)
    except ValueError:
        check("rejects classes.json without live/real", True)


r = subprocess.run([sys.executable, "train_classifier.py", "--data", WORK, "--out",
                    os.path.join(WORK, "x.onnx"), "--workers", "0"],
                   cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, timeout=300)
check("train rejects a bad data dir", r.returncode != 0)

print()
print(f"{'ALL TESTS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)