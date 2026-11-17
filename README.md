# Face Mask Detection

An undergraduate computer-vision prototype built with an OpenCV face detector
and a Keras mask classifier. The follow-up benchmarks reused the original
checkpoint and varied the detection threshold, crop padding and image tiling.

| Configuration | Kaggle image-level correct | Recovered per-face correct |
|---|---:|---:|
| Original full image, 0.50 threshold, no padding | 3,174/3,748 (84.7%) | 111/476 (23.3%) |
| Full image, 0.30 threshold, 30% padding | 3,416/3,748 (91.1%) | 152/476 (31.9%) |
| Full image plus four tiles, 0.30 threshold, 30% padding | 3,365/3,748 (89.8%) | 236/476 (49.6%) |

The Kaggle benchmark scored one prediction per image against its folder label
and counted no prediction as a miss. It excluded 85 exact duplicate copies.
The recovered face-box validation split was used to shortlist configurations.
The original checkpoint's training data is unverified.

- [Kaggle dataset](https://www.kaggle.com/datasets/belsonraja/face-mask-dataset-with-and-without-mask)
- [Kaggle results](results/kaggle-image-level.csv)
- [Recovered validation results](results/recovered-validation.csv)

## Run locally

Use Python with OpenCV, NumPy and TensorFlow/Keras compatible with the original
checkpoint. Supply the local Caffe detector directory and classifier `.h5` file.

```sh
python demo.py --image /path/to/image.jpg --face-dir /path/to/face_detector --model /path/to/mask_detector.h5
```
