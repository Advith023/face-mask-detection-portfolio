# Local setup

The prediction script was smoke-tested using Python 3.9.6 on an Apple Silicon Mac with the versions in requirements.txt. Other platforms have not been verified.

From the repository folder, create an environment and install dependencies.

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

Place the recovered mask classifier in models/mask_detector.h5. If the original file is named mask_detector.model, copy it with the .h5 extension. Its format must be HDF5.

Place deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel in models/face_detector/.

Run one image prediction.

    .venv/bin/python predict_image.py --image /path/to/photo.jpg --face-dir models/face_detector --model models/mask_detector.h5

The default detector threshold is 0.30 and crop padding is 30% per side. The output describes the highest-scoring valid face box.

The model files are currently supplied locally. Their hashes, provenance and acquisition instructions remain to be documented.
