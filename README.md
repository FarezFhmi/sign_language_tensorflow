# 🤟 SignSpeak: Sign Language Recognition Prototype (Hackathon 2025)

**SignSpeak** is a machine learning prototype developed for **Hackathon 2025**. It uses TensorFlow to recognize hand gestures for sign language and serves as the core model for the upcoming SignSpeak app — a tool designed to bridge communication gaps for the Deaf and Hard-of-Hearing community through real-time gesture recognition.

---

## 🎯 Project Goal

To create a working prototype that:

- Recognizes sign language letters using custom-collected hand gesture images.
- Demonstrates real-time predictions using a webcam.
- Lays the groundwork for integration with a future mobile/web application.

---

## ✨ Features

- 🖐️ Real-time hand tracking and keypoint extraction using **MediaPipe Hands**
- 📈 Saves extracted keypoints to `.csv` for model training
- 🧠 Simple and efficient **fully connected network** for classification
- 🧪 Real-time testing with webcam via `real_time_testing.py`
- 🗂️ Model saved as `.h5`, labels stored using `pickle`

---

## 🧰 Tech Stack

- **Language:** - Python 3
- **ML Framework:** - TensorFlow / Keras
- **MediaPipe** – for extracting hand landmarks (21 keypoints × 3 dimensions)
- **Computer Vision:** - OpenCV
- **Visualization:** - Matplotlib, TensorBoard
- **Dataset:** - Self-collected images using webcam
- **Pickle** – for label encoding and loading

---

## 📦 Dataset Collection

The dataset used for training is **self-collected** using the `collect_imgs.py` script. It captures hand gestures via webcam and stores them in labeled folders (one folder per letter).

## 🔄 Workflow

1. **Data Collection & Feature Extraction**
   - Use `collect_imgs.py` to capture webcam input.
   - Use `create_dataset.ipynb` extract hand landmarks via MediaPipe.
   - Save landmarks to a `.csv` file.

2. **Preprocessing & Model Training**
   - Use `preprocess_data.ipynb` to load and clean the CSV data.
   - Encode labels and normalize features.
   - Train a simple fully connected model with TensorFlow.
   - Save model to `.h5` and labels using `pickle`.

3. **Real-Time Inference**
   - Run `real_time_testing.py` to use your webcam and predict gestures in real time.