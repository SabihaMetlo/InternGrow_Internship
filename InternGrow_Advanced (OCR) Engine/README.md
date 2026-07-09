# Advanced OCR Engine

An end-to-end Optical Character Recognition (OCR) system that classifies handwritten characters using a Convolutional Neural Network, and extends into a full pipeline that segments handwritten words from an image and converts them into digital text.

---

## Overview

This project has two parts:

1. **Base Task** — Train a CNN to classify individual handwritten characters (digits + letters) using the EMNIST dataset.
2. **Upgrade Feature** — Build a segmentation pipeline that takes a photo of a full handwritten word, automatically detects each character, and feeds it through the trained CNN to reconstruct the word as editable text.

---

## Dataset

**EMNIST — Balanced Split**

- 131,600 images total (112,800 train / 18,800 test)
- 47 classes — digits (0–9) and letters, with visually-similar uppercase/lowercase pairs merged
- Chosen over plain MNIST (digits only — can't spell words) and over `byclass`/`bymerge` (unbalanced class distribution, which biases the model toward overrepresented characters)

**Known issue handled:** torchvision's EMNIST loader returns images rotated 90° and mirrored by default. This was corrected with a custom transform before training.

---

## Tech Stack

- **Framework:** PyTorch
- **Segmentation:** OpenCV (thresholding + contour detection)
- **Environment:** Google Colab (T4 GPU)

---

## Model Architecture

A CNN with:
- 2 convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool), 32 → 64 filters
- Dropout (0.25) for regularization
- 2 fully connected layers, output layer with 47 classes (Softmax handled internally by `CrossEntropyLoss`)

**Training setup:** Adam optimizer, learning rate 0.001, batch size 64, 5 epochs.

---

## Results

| Metric | Score |
|---|---|
| Training Accuracy (final epoch) | 85.56% |
| **Test Accuracy (unseen data)** | **87.04%** |

Test accuracy slightly exceeding training accuracy indicates the model generalized well, with no signs of overfitting.

---

## Upgrade Feature: Word Segmentation Pipeline

**Pipeline steps:**
1. Upload a photo of handwritten text
2. Apply Otsu's thresholding to convert the image to a clean black-and-white binary format
3. Detect character regions using OpenCV contour detection
4. Sort detected regions left-to-right (reading order)
5. Crop, pad, and resize each character to 28×28 to match EMNIST's format
6. Classify each character with the trained CNN
7. Reconstruct the predicted word from individual character predictions

### Demo Result

Tested on the cursive handwritten word **"products"**:

| Stage | Result |
|---|---|
| Segmentation | 9 regions detected (letters mostly separated; "t" and "s" merged into one region due to connected cursive strokes) |
| Classification (before padding fix) | `PPP0dUFCH` — letters distorted from tight-crop stretching |
| Classification (after padding fix) | `JPPOdUICH` — improved letter framing, but case/style mismatch remained |

---

## Known Limitations & Key Learning

This project intentionally documents *where* the pipeline breaks, not just where it succeeds — real OCR systems have to deal with these tradeoffs.

1. **Cursive letter-joining:** Contour detection identifies connected ink regions, not individual letters. When cursive strokes physically connect two letters (e.g. "t" and "s"), they are detected as a single blob instead of two.
2. **Domain shift:** EMNIST was trained on clean, printed-style, well-centered characters. Real handwriting — especially cursive — looks visually different, which reduces classification accuracy even when segmentation works correctly. This is a well-documented challenge in OCR research, not a bug in this implementation.
3. **Tight-crop distortion:** Cropping directly around ink without padding stretches letters when resized to 28×28. This was partially mitigated by adding a padding step before resizing.

**Potential improvements:**
- Train/fine-tune on a cursive-specific handwriting dataset
- Use a dedicated text-detection model (e.g. CRAFT) instead of simple contour detection for connected/cursive handwriting
- Add line-segmentation as a pre-step to support full multi-line documents, not just single words

---

## Repository Structure

```
Task_3_OCR_Engine/
├── TASK_3_OCR_Engine.ipynb
└── README.md
```

---

## Author

**Sabiha Metlo**
BS Information Technology, Quaid-e-Awam University, Nawabshah
[GitHub](https://github.com/SabihaMetlo) | [LinkedIn](www.linkedin.com/in/sabiha-metlo-7945bb328)
