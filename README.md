# Malaria Detection Using CNN

This repository contains an interactive web application built with Streamlit to detect malaria from microscopic images using a Convolutional Neural Network (CNN). The project leverages machine learning to classify images as either "Parasitized" or "Uninfected," aiding in malaria diagnosis.

---

## 🚀 Features

- **Interactive Web App**: Allows users to upload images and receive real-time predictions.
- **Deep Learning Model**: Powered by TensorFlow and a CNN architecture for high accuracy.
- **Visualization**: Displays input images and prediction probabilities.
- **Streamlined Interface**: User-friendly interface using Streamlit.

---

## 🛠️ Technologies Used

- **Programming Language**: Python
- **Frameworks and Libraries**:
  - [TensorFlow](https://www.tensorflow.org/): For building and training the CNN model.
  - [NumPy](https://numpy.org/): For numerical computations.
  - [Matplotlib](https://matplotlib.org/): For data visualization.
  - [Seaborn](https://seaborn.pydata.org/): For enhanced data visualizations.
  - [Streamlit](https://streamlit.io/): For building the interactive web app.

---

## 📂 Project Structure

```
Malaria-Detection-Using-CNN/
├── app.py                 # Streamlit web app script
├── model/                 # Pretrained CNN model
├── data/                  # Microscopic image dataset (if applicable)
├── utils/                 # Helper scripts for preprocessing and visualization
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── Malaria Detection Project.ipynb  # Jupyter notebook for model development
```

---

## 📊 Dataset

The model is trained on a publicly available dataset of microscopic images of red blood cells. The dataset contains two classes:

- **Parasitized**: Cells infected with malaria parasites.
- **Uninfected**: Healthy cells without infection.

Dataset preprocessing and augmentation steps are included in the notebook.

---

## 🧠 Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Optimizer**: Adam Optimizer with learning rate adjustments
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

---

## 📈 Model Results and Performance

The CNN model achieved the following results on the validation dataset:

- **Training Accuracy**: 94.5%
- **Validation Accuracy**: 67.9%
- **Loss**: 0.195 (validation loss)

These results demonstrate that the model can effectively differentiate between parasitized and uninfected red blood cells.

---

## 🎨 Screenshots

1. **Upload Section**: Users can upload their image files.
2. **Prediction Results**: Displays the predicted class with confidence scores.

---
---

