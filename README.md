# 🚗 License Plate Recognition using SIFT Feature Matching

> A Computer Vision project for automatic vehicle license plate recognition using SIFT keypoint detection, feature descriptor matching, Brute-Force Matcher, and Lowe's Ratio Test with OpenCV.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green)
![SIFT](https://img.shields.io/badge/SIFT-Feature_Detection-red)
![BFMatcher](https://img.shields.io/badge/Brute_Force-Matching-orange)
![Master](https://img.shields.io/badge/Master-MIV-lightgrey)

---

# 📖 Overview

This project presents a complete **Computer Vision pipeline** for **vehicle license plate recognition** based on local feature extraction.

Instead of using Optical Character Recognition (OCR), the system identifies vehicles by comparing the visual characteristics of license plates using **SIFT (Scale-Invariant Feature Transform)** descriptors.

Each query license plate is compared against a dataset of authorized vehicles. The system determines the most similar plate by matching local image descriptors and decides whether the vehicle is authorized to enter the parking area.

The project was developed as part of the **Computer Vision** course in the **Master's Program in Image Processing & Artificial Intelligence (MIV)** at **USTHB**.

---

# ✨ Features

- 🚗 Automatic license plate recognition
- 🔍 SIFT keypoint detection
- 🧠 SIFT descriptor extraction
- 🔗 Brute-Force descriptor matching
- 📊 Lowe's Ratio Test filtering
- 🎯 Best-match identification
- ✅ Authorized / Unauthorized vehicle decision
- 🖼️ Visualization of matched keypoints
- ⚡ Fast feature-based image comparison
- 📈 Multiple query image processing

---

# 📑 Project Report

## Project Objective

Automatic vehicle identification has become an essential component of modern intelligent transportation systems.

Parking lots, residential compounds, universities, and secured facilities often require automatic verification of authorized vehicles before granting access.

The objective of this project is to develop an image-based identification system capable of recognizing a vehicle by matching its license plate against a database of registered plates.

Unlike OCR systems that attempt to read individual characters, this project considers the license plate as a complete visual object and identifies it through local feature matching.

The problem can therefore be formulated as an **image retrieval and matching task**, where:

- Query Image → Incoming vehicle plate
- Train Images → Authorized vehicle database

The system searches for the most similar plate and decides whether the vehicle should be granted access.

---

# 🧠 Why SIFT?

SIFT (Scale-Invariant Feature Transform) is one of the most influential feature extraction algorithms in computer vision.

It detects distinctive keypoints and computes highly descriptive local feature vectors that remain stable under various image transformations.

SIFT was selected because it provides:

- Scale invariance
- Rotation invariance
- Robustness to illumination changes
- Robustness to moderate viewpoint changes
- Excellent matching performance
- High repeatability

Unlike simple pixel comparison methods, SIFT focuses only on the most informative regions of the image, making it much more reliable for license plate recognition.

---

# 🔬 Why Feature Matching?

The objective is not to classify plates using deep learning but to determine whether a query image corresponds to an existing plate in the dataset.

Feature matching provides several advantages:

- No model training required
- Works well with small datasets
- Fast implementation
- Easy to interpret
- Robust against small image transformations

The similarity between two plates is estimated from the number of valid feature correspondences.

---

# ⚙️ Matching Strategy

The project follows several processing stages.

## Stage 1 — Image Loading

The system loads:

- Authorized license plate dataset
- Query license plate images

All images are converted to grayscale before processing.

---

## Stage 2 — SIFT Feature Extraction

For each image, SIFT computes:

- Keypoints
- Local descriptors

Each descriptor summarizes the visual appearance around a detected keypoint.

---

## Stage 3 — Descriptor Matching

Feature descriptors are compared using the OpenCV **Brute-Force Matcher (BFMatcher)**.

Each descriptor from the query image searches for its nearest neighbors in every training image.

---

## Stage 4 — Lowe's Ratio Test

Not every nearest neighbor represents a reliable correspondence.

To eliminate ambiguous matches, Lowe's Ratio Test is applied:

- Compute the two nearest neighbors
- Accept the match only if

```
distance(first) < 0.75 × distance(second)
```

This greatly reduces false matches.

---

## Stage 5 — Best Match Selection

For every authorized plate:

- Count valid matches
- Keep the highest score

The image with the largest number of valid correspondences is selected as the best candidate.

---

## Stage 6 — Decision

The final decision is based on a threshold.

If:

```
Number of Good Matches > Threshold
```

The vehicle is considered:

✅ Authorized

Otherwise:

❌ Unauthorized

---

# 📊 Evaluation Strategy

Performance is evaluated using:

- Number of detected keypoints
- Number of descriptor matches
- Number of good matches
- Best matching score
- Visualization of correspondences

Rather than classification accuracy, the matching score indicates how similar two license plates are.

---

# 🏗️ Pipeline

## 1️⃣ Image Acquisition

- Load query images
- Load authorized license plate database

---

## 2️⃣ Image Preprocessing

- Convert images to grayscale
- Remove invalid images
- Prepare descriptors

---

## 3️⃣ Feature Extraction

Using SIFT:

- Detect keypoints
- Compute descriptors

---

## 4️⃣ Descriptor Matching

Using BFMatcher:

- Find nearest neighbors
- Compare descriptors

---

## 5️⃣ Match Filtering

Apply Lowe's Ratio Test to remove incorrect matches.

---

## 6️⃣ Vehicle Identification

- Count valid correspondences
- Select best matching plate
- Compare score against decision threshold

---

## 7️⃣ Result Visualization

Display:

- Query plate
- Authorized plate
- Matched keypoints
- Number of good matches
- Authorization decision

---

# 📂 Project Structure

```text
SIFT-License-Plate-Recognition-Using-OpenCV/

│
├── Sift_Detect_Draw.py
├── BruteForceMatching.py
├── BruteForceMatchingSorting.py
├── exo1.py
├── exo2.py
├── exo22.py
├── exo222.py
├── exo2222.py
│
├── Matricules/
│   ├── plate1.jpg
│   ├── plate2.jpg
│   └── ...
│
├── License Plates/
│   ├── vehicle1.jpg
│   ├── vehicle2.jpg
│   └── ...
│
├── TP1 CV 2026.pdf
│
└── README.md
```

---

# ▶️ Running the Project

## Install dependencies

```bash
pip install opencv-python matplotlib numpy
```

Run one of the implementations:

```bash
python exo2222.py
```

or

```bash
python abdou22.py
```

---

# ✅ Strengths

- Robust SIFT feature extraction
- Scale and rotation invariant
- No deep learning required
- Simple and interpretable pipeline
- Efficient feature matching
- Good performance on small datasets
- Clear visualization of correspondences
- Suitable for parking access control

---

# ⚠️ Limitations

- Sensitive to severe occlusions
- Performance depends on image quality
- Computationally slower than ORB
- Requires good lighting conditions
- Fixed threshold may require tuning for different datasets

---

# 🚀 Future Improvements

- OCR-based character recognition
- Automatic license plate detection
- ORB implementation for real-time performance
- FLANN-based matching
- RANSAC homography verification
- Deep learning-based plate recognition
- Real-time webcam integration
- Parking management system deployment

---

# 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- SIFT
- BFMatcher
- Lowe's Ratio Test

---

# 📚 References

- David G. Lowe. Distinctive Image Features from Scale-Invariant Keypoints. IJCV, 2004.
- OpenCV Documentation
- OpenCV Feature2D Module
- OpenCV BFMatcher Documentation
- SIFT Algorithm Documentation
- Computer Vision: Algorithms and Applications — Richard Szeliski
