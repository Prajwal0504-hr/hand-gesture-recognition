# Hand Gesture Recognition using MobileNetV2

## Overview

This project implements a deep learning-based hand gesture recognition system using MobileNetV2 with transfer learning. The system classifies hand gestures such as FIST, ONE, PALM, and SUPER. A Flask-based web application is developed to allow users to upload images and receive predictions in real time.

---

## Features

* User authentication system (registration and login)
* Image upload for gesture prediction
* Deep learning model using MobileNetV2
* Performance evaluation using confusion matrix and accuracy metrics
* Web-based interface using Flask

---

## Technology Stack

* Python
* TensorFlow / Keras
* OpenCV
* Flask
* SQLite

---

## Project Structure

```
hand-gesture-recognition/
│
├── app/            # Flask application (routes, templates, uploads)
├── data/           # Dataset (not included in repository)
├── models/         # Trained model file
├── results/        # Output images (accuracy, confusion matrix)
├── src/            # Model training scripts
│
├── README.md
├── requirements.txt
├── .gitignore
```

---

## Model Details

* Model: MobileNetV2 (Transfer Learning)
* Input size: 128x128 grayscale images
* Number of classes: 4 (FIST, ONE, PALM, SUPER)
* Accuracy: Approximately 88%

---

## Results

The model performance is evaluated using:

* Classification report
* Confusion matrix
* Accuracy and loss graphs

(Include images from the `results/` folder here)

---

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition
```

2. Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Application

```
cd app
python app.py
```

Open a browser and navigate to:

```
http://127.0.0.1:5000
```

---

## Use Cases

* Sign language recognition
* Human-computer interaction systems
* Accessibility tools

---

## Future Improvements

* Real-time webcam-based gesture detection
* Increase dataset size for better accuracy
* Deployment on cloud platforms

---

## Author

Prajwal H R
