# Sign In with ID Verification

A single-page Streamlit proof-of-concept that signs a user in by comparing a live
webcam capture against the photo on their uploaded ID card, then auto-fills their
details by reading the card with OCR.

> ⚠️ **Proof of concept, not production-ready.** This project processes government
> ID images and biometric face data entirely in memory for demo purposes — it does
> not encrypt, store, or transmit that data anywhere. Do not deploy this as-is for
> real identity verification without a proper security and privacy review.

## Features

- **Face verification** — detects the face on the ID card and the live camera
  frame using OpenCV's DNN face detector (no GPU required), then compares them
  with [DeepFace](https://github.com/serengil/deepface) (`ArcFace` model, cosine
  distance).
- **Auto-fill from OCR** — reads the card with [EasyOCR](https://github.com/JaidedAI/EasyOCR)
  and automatically fills in Name, Place, and Center once verification succeeds.
- **Sign In flow** — a single page: upload card → open camera → Compare →
  auto-filled fields → Sign In button.
- **Debug panel** — an expandable "Raw OCR text" section shows exactly what the
  OCR engine read, to help diagnose extraction issues.

## Project structure

```
id_verify_app/
├── app.py                  # Main Streamlit page (Sign In flow)
├── core/
│   ├── card_detector.py    # Card boundary detection (Canny + contour)
│   ├── face_detector.py    # Face detection (OpenCV DNN, padded crop)
│   ├── verifier.py         # DeepFace.verify wrapper (ArcFace)
│   └── ocr_extractor.py    # EasyOCR wrapper — reads Name / Place / Center
├── models/                 # Pretrained DNN weights (included)
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── requirements.txt
└── problem.txt              # Notes from tuning the detection heuristics
```

## Setup

```bash
conda create -n ocr python=3.10 -y
conda activate ocr
pip install --upgrade pip

# Install the lightweight CPU build of torch first — otherwise easyocr pulls
# the default CUDA build (several GB) even if you don't have a GPU.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The first run downloads the EasyOCR language models (~64MB) automatically, so
it needs an internet connection once.

## How it works

1. Upload an ID card image and take a live photo with the camera.
2. `CardDetector` finds the card boundary in the uploaded image.
3. `FaceDetector` (OpenCV DNN, SSD ResNet10) locates the face on the card and
   in the live frame, with padding around the box for better alignment.
4. `OcrExtractor` upscales the card crop and reads its text with EasyOCR,
   extracting **Name**, **Place**, and **Center** from the address lines.
5. `FaceVerifier` compares the two face crops with DeepFace (`ArcFace`,
   cosine distance) and returns `verified` + `distance`.
6. If verified, the extracted fields populate the form and a **Sign In**
   button becomes available.

## Known limitations

- **National ID number is not reliably extracted.** Egyptian ID cards print
  the national ID in an embossed/dot-matrix font that generic OCR engines
  (EasyOCR, Tesseract) struggle to read without specialized preprocessing
  targeted at that specific region. For that reason this build extracts
  **Place** and **Center** from the address lines instead, which OCR reads
  much more reliably.
- **Verification accuracy depends on photo quality/age.** A large gap in
  lighting, angle, or time between the ID photo and the live capture can push
  the distance above the model's threshold even for a genuine match. If you
  see frequent false rejections, you can pass a manual `threshold` to
  `FaceVerifier(...)` in `app.py`.
- This is a CPU-only pipeline by design (no GPU dependency), which trades
  some speed for easy setup on any machine.

## Troubleshooting

**`AttributeError: module 'cv2.dnn' has no attribute 'readNetFromCaffe'`**

This happens when both `opencv-python` and `opencv-python-headless` end up
installed in the same environment (a common side effect of `easyocr`/`deepface`
pulling in `opencv-python-headless` as a dependency). Fix:

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
pip install opencv-python-headless==4.10.0.84
```

Then verify:

```bash
python -c "import cv2; print(cv2.dnn.readNetFromCaffe)"
```

## License

For demonstration purposes only.
