# Breast Cancer Classification using CNN

This project focuses on automated breast cancer detection using **Convolutional Neural Networks (CNNs)** applied to histopathology image patches. The goal is to classify breast tissue patches as **Benign (Non-IDC)** or **Malignant (IDC-positive)** using deep learning techniques.

---

## Project Overview

Breast cancer diagnosis through histopathology is a manual, time-consuming process that requires expert analysis of microscopic tissue slides. With the growing volume of biopsy samples, there is a strong need for automated systems that can assist pathologists in early and accurate detection.

This project implements a **custom CNN model** trained on the **IDC (Invasive Ductal Carcinoma) breast histopathology dataset**, enabling patch-level classification of cancerous and non-cancerous tissue.

---

## Dataset

- **Dataset Name:** Breast Histopathology Images (IDC Dataset)
- **Source:** Kaggle / Case Western Reserve University
- **Dataset Link:**  
  https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
- **Folder Used:** `IDC_regular_ps50_idx5`
- **Image Size:** 50×50 RGB patches (resized to 64×64×3)
- **Classes:**
  - `0` – Benign (Non-IDC)
  - `1` – Malignant (IDC-positive)

### Dataset Subset Used
Due to memory and runtime constraints, a controlled subset of **20,000 images** was used:
- Benign: 15,408
- Malignant: 4,592

---

### File Description
- **Breast_Cancer_classification_with_DL.ipynb**  
  Jupyter Notebook containing the complete implementation:
  - Data loading & preprocessing  
  - CNN model definition  
  - Training & evaluation  
  - Confusion matrix  
  - Grad-CAM visualizations  

- **BreastCancerClassification_Report.pdf**  
  Full dissertation report including methodology, results, visualizations, and analysis.

- **BreastCancerClassification.pptx**  
  Presentation summarizing the project workflow, model architecture, and results.

---

## Methodology

1. **Data Preprocessing**
   - Image resizing to 64×64
   - Pixel normalization (0–1 range)
   - Data augmentation using `ImageDataGenerator`

2. **CNN Architecture**
   - Convolution + ReLU layers
   - MaxPooling layers
   - Flatten + Dense layers
   - Sigmoid output for binary classification

3. **Model Compilation**
   - Optimizer: Adam
   - Loss Function: Binary Crossentropy
   - Metric: Accuracy

4. **Training Strategy**
   - Train/validation split
   - Callbacks:
     - EarlyStopping
     - ModelCheckpoint

5. **Evaluation**
   - Accuracy & loss curves
   - Confusion matrix
   - Grad-CAM visualizations for interpretability

---

## Results

- Stable training and validation accuracy
- Consistent loss reduction across epochs
- Effective learning of localized cancer patterns
- Grad-CAM confirms focus on medically relevant regions
- Model shows strong potential for patch-level breast cancer screening

---

## Limitations

- Patch-level classification lacks global tissue context
- Dataset class imbalance (Benign > Malignant)
- No external dataset validation
- Not intended for direct clinical deployment without expert validation

---

## Future Improvements

- Use deeper architectures (ResNet, EfficientNet, DenseNet)
- Train on larger and more diverse datasets
- Extend to multi-class classification (IDC, DCIS, benign subtypes)
- Deploy as a web-based diagnostic support system
- Perform whole-slide image (WSI) analysis

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Google Colab




