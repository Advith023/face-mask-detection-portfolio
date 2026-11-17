# Face Mask Detection

A face-mask screening project built with an OpenCV face detector
and a Keras mask classifier. The follow-up benchmarks reused the original
checkpoint and varied the detection threshold, crop padding and image tiling.

| Configuration | Reproduced image-level correct |
|---|---:|
| Original full image, 0.50 threshold, no padding | 3,174/3,748 (84.69%) |
| Full image, 0.30 threshold, 30% padding | 3,416/3,748 (91.14%) |
| Full image plus four tiles, 0.30 threshold, 30% padding | 3,365/3,748 (89.78%) |

The Kaggle benchmark scored one prediction per image against its folder label
and counted no prediction as a miss. It excluded 85 exact duplicate copies.
Earlier per-face experiments informed configuration selection. Their results
are retained separately, with an unresolved evaluation-subset discrepancy
documented in [validation status](VALIDATION.md).
The original checkpoint's training data is unverified.

- [Kaggle dataset](https://www.kaggle.com/datasets/belsonraja/face-mask-dataset-with-and-without-mask)
- [Kaggle results](results/kaggle-image-level.csv)
- [Earlier per-face results — not yet reproduced](results/recovered-validation.csv)

## Run locally

Use Python with OpenCV, NumPy and TensorFlow/Keras compatible with the original
checkpoint. Supply the local Caffe detector directory and classifier `.h5` file.

```sh
python predict_image.py --image /path/to/image.jpg --face-dir /path/to/face_detector --model /path/to/mask_detector.h5
```

## Original project source

The project was built by Advith Ashok. See the [original source archive](original/README.md).

## Installation

See [local setup instructions](SETUP.md) and [dependency versions](requirements.txt).

## Reproduce the image benchmark

On 17 November 2026, a local run reproduced every aggregate count in
results/kaggle-image-level.csv across all three configurations.

The run processed 3,833 images, removed 85 exact duplicate copies and
evaluated 3,748 images. No exact file matches were found against the
supplied recovered dataset. The original training data remains unverified.

- [Per-image predictions and SHA-256 hashes](results/kaggle-per-image.csv)
- [Recorded macOS Python 3.9 environment](requirements-macos-py39.lock.txt)
- [Benchmark implementation](benchmark_images.py)

After following SETUP.md, run:

```sh
.venv/bin/python benchmark_images.py \
  --images /path/to/images \
  --exclude-images /path/to/MaskedFace \
  --face-dir /path/to/face_detector \
  --model models/mask_detector.h5 \
  --output-dir outputs
```

## Development and contribution

**Project developer and maintainer: Advith Ashok (@Advith023).**

Developed and integrated the face-mask screening application, including image and video inference, a desktop interface and serial hardware communication.

The November 2026 work addressed inconsistencies in the recovered implementation and verified three inference configurations. The best configuration achieved 91.14% correct image-level predictions across 3,748 deduplicated images, compared with an 84.69% baseline, without retraining.

The repository includes recovered source code, benchmark scripts, per-image predictions, dependency versions and model fingerprints.

See [references and acknowledgements](ATTRIBUTION.md).

## Validation status

The image-level benchmark was reproduced locally. Check the published
prediction records against the aggregate CSV without ML dependencies:

    python3 check_results.py

See [validation and implementation status](VALIDATION.md) for benchmark
interpretation, model availability and the status of archived modules.
