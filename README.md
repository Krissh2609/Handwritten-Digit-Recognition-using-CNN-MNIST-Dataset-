# Handwritten-Digit-Recognition-using-CNN-MNIST-Dataset-
This project focuses on building a Convolutional Neural Network (CNN) model to recognize handwritten digits (0–9) from the MNIST dataset. The goal is to accurately classify images of digits and understand how CNNs learn spatial features in image data.

---

## 📂 Dataset Description
- **Dataset:** MNIST Handwritten Digits (available via Keras Datasets)  
- **Images:** 70,000 total (60,000 training + 10,000 testing)  
- **Image Size:** 28×28 pixels (grayscale)  
- **Classes:** 10 (digits 0–9)  

Each image is a grayscale digit centered in a 28×28 pixel frame. The dataset is clean, balanced, and ideal for benchmarking image classification models.

---

## ⚙️ Model Architecture
The CNN model was built using **TensorFlow and Keras** and includes the following layers:

| Layer | Type | Purpose |
|-------|------|----------|
| Conv2D (32 filters) | Convolution | Extracts basic features like edges |
| MaxPooling2D | Pooling | Reduces spatial size and computation |
| Conv2D (64 filters) | Convolution | Learns more complex shapes |
| MaxPooling2D | Pooling | Further reduces data size |
| Conv2D (128 filters) | Convolution | Captures deeper abstract features |
| Flatten | Transformation | Converts feature maps to 1D vector |
| Dropout (0.3) | Regularization | Prevents overfitting |
| Dense (128 units, ReLU) | Fully Connected | Learns feature combinations |
| Dense (10 units, Softmax) | Output | Predicts probability for each digit |

**Optimizer:** Adam (learning rate = 0.001)  
**Loss Function:** Sparse Categorical Crossentropy  
**Metrics:** Accuracy  

---

## 🧠 Training Details
- **Epochs:** 8  
- **Batch Size:** 128  
- **Validation Split:** 10%  
- **Callbacks Used:**  
  - `EarlyStopping` → Stops training when validation accuracy stops improving  
  - `ReduceLROnPlateau` → Reduces learning rate when performance plateaus  
  - `ModelCheckpoint` → Saves the best-performing model automatically  

---

## 📊 Results & Performance
- **Training Accuracy:** ~99.8%  
- **Validation Accuracy:** ~99.0%  
- **Test Accuracy:** **99.22%**  
- **Validation Loss:** ~0.04  

### Classification Report Summary:
- Precision, Recall, and F1-score ≈ **0.99** for all digits  
- Model performs consistently across all classes  
- **Accuracy:** 99.22% | **Macro F1:** 0.9921 | **Weighted F1:** 0.9922  

### Confusion Matrix Summary:
- Matrix is almost perfectly diagonal  
- Very few misclassifications (mostly between 5↔3 and 9↔4)  
- Confirms the model’s high accuracy and generalization  

---

## 📈 Visualization
- **Accuracy & Loss Curves:** Show smooth convergence and no overfitting.  
- **Confusion Matrix:** Demonstrates strong diagonal dominance (correct predictions).  
- **Sample Predictions:** The model predicts nearly all digits correctly.

---

## 🧩 Tech Stack
`Python` • `TensorFlow` • `Keras` • `NumPy` • `Matplotlib` • `Scikit-learn`

---
