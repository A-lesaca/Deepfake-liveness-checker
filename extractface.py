import argparse
import os
import random

import cv2

from face_utils import CROP_SIZE, create_landmarker, crop_face, to_mp_image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def frames_from(path, every):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
        img = cv2.imread(path)
        if img is not None:
            yield 0, img
    elif ext in VIDEO_EXT:
        cap = cv2.VideoCapture(path)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every == 0:
                yield i, frame
            i += 1
        cap.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--every", type=int, default=10, help="keep every Nth video frame")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--max-per-file", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    landmarker = create_landmarker(video=False, max_faces=1)

    for cls in sorted(os.listdir(args.input)):
        cls_dir = os.path.join(args.input, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [os.path.join(r, f) for r, _, fs in os.walk(cls_dir) for f in fs
                 if os.path.splitext(f)[1].lower() in IMAGE_EXT | VIDEO_EXT]
        random.shuffle(files)
        n_val = int(len(files) * args.val_split)
        saved = 0
        for idx, path in enumerate(files):
            split = "val" if idx < n_val else "train"
            out_dir = os.path.join(args.output, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.splitext(os.path.relpath(path, cls_dir))[0].replace(os.sep, "_")
            kept = 0
            for frame_idx, frame in frames_from(path, args.every):
                result = landmarker.detect(to_mp_image(frame))
                if not result.face_landmarks:
                    continue
                crop = crop_face(frame, result.face_landmarks[0], CROP_SIZE)
                if crop is None:
                    continue
                cv2.imwrite(os.path.join(out_dir, f"{stem}_{frame_idx:06d}.jpg"), crop)
                kept += 1
                saved += 1
                if kept >= args.max_per_file:
                    break
        print(f"{cls}: {len(files)} source files -> {saved} face crops")

    landmarker.close()


if __name__ == "__main__":
    main()