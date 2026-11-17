import argparse
from pathlib import Path
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser(description="Predict mask status for one image.")
parser.add_argument("--image", required=True, type=Path)
parser.add_argument("--face-dir", required=True, type=Path)
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--threshold", type=float, default=0.30)
parser.add_argument("--padding", type=float, default=0.30)
args = parser.parse_args()

0 <= args.threshold <= 1 and 0 <= args.padding <= 1 or parser.error("Threshold and padding must be between 0 and 1.")
image = cv2.imread(str(args.image))
image is not None or parser.error(f"Could not read image: {args.image}")
height, width = image.shape[:2]
detector = cv2.dnn.readNetFromCaffe(
str(args.face_dir / "deploy.prototxt"),
str(args.face_dir / "res10_300x300_ssd_iter_140000.caffemodel"),
)
detector.setInput(cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0)))
rows = detector.forward()[0, 0]
candidates = rows[rows[:, 2].astype(float) > args.threshold]
scale = np.array([width, height, width, height])
boxes = (candidates[:, 3:7] * scale).astype(int)
boxes[:, :2] = np.maximum(boxes[:, :2], 0)
boxes[:, 2:] = np.minimum(boxes[:, 2:], [width - 1, height - 1])
valid = np.flatnonzero(np.all(boxes[:, :2] < boxes[:, 2:], axis=1))
valid.size or print("No prediction: no valid face box passed the threshold.")
valid.size or exit(0)
best = valid[np.argmax(candidates[valid, 2])]
x1, y1, x2, y2 = boxes[best]
px = round((x2 - x1) * args.padding)
py = round((y2 - y1) * args.padding)
crop = image[max(0, y1 - py):min(height, y2 + py), max(0, x1 - px):min(width, x2 + px)]
rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
batch = np.asarray([cv2.resize(rgb, (224, 224))], dtype=np.float32)
model = load_model(str(args.model), compile=False)
scores = np.asarray(model(preprocess_input(batch), training=False))[0]
index = int(scores.argmax())
print(f"{('Mask', 'No Mask')[index]} (model score {scores[index]:.1%})")
