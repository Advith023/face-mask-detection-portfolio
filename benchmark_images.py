"""Compare three inference configurations on images labelled by folder."""
import argparse
import csv
from collections import Counter
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

LABELS = ("with_mask", "without_mask")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}
CONFIGS = (
    ("original_full_050_pad0", 0.50, 0.0),
    ("full_030_pad30", 0.30, 0.30),
    ("tiled_030_pad30", 0.30, 0.30),
)


def image_paths(folder):
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def best_box(image, detector, left=0, top=0):
    height, width = image.shape[:2]
    detector.setInput(cv2.dnn.blobFromImage(
        image, 1.0, (300, 300), (104.0, 177.0, 123.0)
    ))
    scale = np.array([width, height, width, height])
    rows = sorted(detector.forward()[0, 0], key=lambda row: -row[2])
    for row in rows:
        score = float(row[2])
        if score <= 0.30:
            break
        x1, y1, x2, y2 = map(int, row[3:7] * scale)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x1 < x2 and y1 < y2:
            return score, (x1 + left, y1 + top, x2 + left, y2 + top)
    return None


def classify(image, hit, padding, model):
    if hit is None:
        return "no_prediction"
    height, width = image.shape[:2]
    x1, y1, x2, y2 = hit[1]
    px = round((x2 - x1) * padding)
    py = round((y2 - y1) * padding)
    crop = image[max(0, y1 - py):min(height, y2 + py),
                 max(0, x1 - px):min(width, x2 + px)]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    batch = np.asarray([cv2.resize(rgb, (224, 224))], dtype=np.float32)
    scores = np.asarray(model(preprocess_input(batch), training=False))[0]
    return LABELS[int(scores.argmax())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--exclude-images", required=True, type=Path)
    parser.add_argument("--face-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    excluded_paths = image_paths(args.exclude_images)
    if not excluded_paths:
        parser.error("--exclude-images must contain the recovered dataset")
    excluded = {fingerprint(path) for path in excluded_paths}
    samples = {}
    raw_count = 0
    for label in LABELS:
        paths = image_paths(args.images / label)
        if not paths:
            parser.error(f"No images found in {args.images / label}")
        for path in paths:
            raw_count += 1
            digest = fingerprint(path)
            if digest in samples and samples[digest][0] != label:
                parser.error(f"Conflicting labels for identical image: {path}")
            samples.setdefault(digest, (label, path))

    overlap = sum(digest in excluded for digest in samples)
    selected = [
        (digest, label, path) for digest, (label, path) in samples.items()
        if digest not in excluded
    ]
    if not selected:
        parser.error("No images remain after excluding recovered images")

    print(f"Raw images: {raw_count}")
    print(f"Exact duplicate copies removed: {raw_count - len(samples)}")
    print(f"Exact recovered-dataset matches excluded: {overlap}")
    print(f"Selected images: {len(selected)}", flush=True)

    detector = cv2.dnn.readNetFromCaffe(
        str(args.face_dir / "deploy.prototxt"),
        str(args.face_dir / "res10_300x300_ssd_iter_140000.caffemodel"),
    )
    model = load_model(str(args.model), compile=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"image_predictions_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    counts = Counter()
    totals = Counter(label for _, label, _ in selected)

    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("sha256", "relative_path", "folder_label",
                         "configuration", "prediction", "unreadable"))
        for number, (digest, actual, path) in enumerate(selected, 1):
            image = cv2.imread(str(path))
            hits = (None, None, None)
            if image is not None:
                height, width = image.shape[:2]
                full = best_box(image, detector)
                candidates = [full] if full is not None else []
                tile_w = min(width, max(2, round(width * 0.60)))
                tile_h = min(height, max(2, round(height * 0.60)))
                for top in sorted({0, height - tile_h}):
                    for left in sorted({0, width - tile_w}):
                        hit = best_box(
                            image[top:top + tile_h, left:left + tile_w],
                            detector, left, top,
                        )
                        if hit is not None:
                            candidates.append(hit)
                tiled = max(candidates, key=lambda hit: hit[0], default=None)
                hits = (full, full, tiled)

            for (name, threshold, padding), hit in zip(CONFIGS, hits):
                if hit is not None and hit[0] <= threshold:
                    hit = None
                predicted = classify(image, hit, padding, model)
                counts[name, actual, predicted] += 1
                writer.writerow((digest, path.relative_to(args.images), actual,
                                 name, predicted, image is None))
            if number % 250 == 0 or number == len(selected):
                handle.flush()
                print(f"Processed {number}/{len(selected)}", flush=True)

    for name, _, _ in CONFIGS:
        print(f"\n{name}")
        correct = 0
        for actual in LABELS:
            mask = counts[name, actual, LABELS[0]]
            no_mask = counts[name, actual, LABELS[1]]
            missing = counts[name, actual, "no_prediction"]
            print(f"  {actual}: Mask {mask}, No Mask {no_mask}, no prediction {missing}")
            correct += counts[name, actual, actual]
        print(f"  Overall correct: {correct}/{sum(totals.values())}")
    print(f"\nPer-image results saved: {output.resolve()}")


if __name__ == "__main__":
    main()
