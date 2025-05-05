# Pneumonia Detection from Chest X-Ray

Automated pneumonia detection using deep learning techniques applied to chest X-ray images. This project demonstrates how convolutional neural networks (CNNs) can assist in medical diagnosis by accurately classifying X-ray images as either **Normal** or **Pneumonia**.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Early and accurate detection of pneumonia is crucial for effective treatment. This repository presents a step-by-step implementation for building a classifier to detect pneumonia using Keras and TensorFlow on a labeled X-ray images dataset.

---

## Features

- Deep learning-based image classification
- Data augmentation and preprocessing
- Model training and validation
- Performance evaluation using accuracy and confusion matrix
- Inference with new chest X-ray images

---

## Dataset

The training and evaluation are conducted on the [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset, consisting of 5,800+ X-ray images labeled as **NORMAL** or **PNEUMONIA**.

**Note:**  
Please download the dataset manually from Kaggle and place it in the designated `dataset` directory.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anubhab0101/Pneumonia-Detection-from-Chest-X-Ray.git
   cd Pneumonia-Detection-from-Chest-X-Ray
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
   - Extract it to the `dataset` folder as:
     ```
     dataset/chest_xray/train/
     dataset/chest_xray/val/
     dataset/chest_xray/test/
     ```

---

## Usage

1. **Train the Model**
   ```bash
   python train_model.py
   ```

2. **Predict on New Images**
   Modify the `pp.py` or relevant inference script, providing the path to your X-ray image.

---

## Results

- High accuracy in pneumonia/normal classification.
- Sample output and confusion matrix visualization in Jupyter notebook or result files.

---

## Project Structure

```
Pneumonia-Detection-from-Chest-X-Ray/
│
├── dataset/                   # Place dataset here
├── train_model.py             # Main training script
├── Model.ipynb                # Jupyter notebook with complete workflow
├── pp.py                      # Script for prediction on new images
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss a change.

---

## License

[MIT License](LICENSE)

---

**Note:**  
This project is for academic/research purposes only and not intended for clinical use.
if any one want to try this here(https://anubhab0101-pneumonia-detection-from-chest-x-ray-pp-qdevtn.streamlit.app/ #here) link to try
