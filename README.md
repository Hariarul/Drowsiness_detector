# Sleeping Person Detection in Car 🚗💤

This project detects sleeping individuals in a car and predicts their age. It uses YOLOv8 for person detection and a custom model for age prediction. The system works with both images and videos and can handle multiple people in the car.

## Features ✨

Detects sleeping people in a car 🛌.

Predicts the age of detected individuals 🧑‍🦳.

Handles multiple sleeping people in a car 👥.

Displays results with bounding boxes 📦 and triggers pop-up notifications with detected people’s names and emotions 💬.

## Technologies Used 🔧

YOLOv8: Pre-trained and custom-trained models for detecting sleeping persons in a car.

PyTorch: Deep learning framework for model training and inference.

OpenCV: For image and video processing.

TensorFlow: Used for predicting age from detected faces.

Streamlit: For creating an interactive web interface.

## Installation ⚙️
Clone the repo:

git clone https://github.com/Hariarul/Drowsiness_detector

### Install dependencies:

pip install -r requirements.txt

Download pre-trained models or train custom model (YOLOv8 and age prediction models).

### Run the app:

streamlit run Driver_drowsiness.py

## How It Works 🎬

Upload an image or video containing a car with sleeping people.

The model detects sleeping individuals and predicts their ages.

Pop-up notifications will display the results, including:

## Number of sleeping individuals 🛏️.

Ages of each person 👵👶.

Emotions (optional) 😃😴.

## Example 🖼️

Input Video/Image 📹

A car with sleeping people.

### Output 🧑‍🦳👥

"Sleeping People Detected: 2"

Person 1 - Age: 30 🧑

Person 2 - Age: 40 👵

Pop-up notification: "Person 1 - Sleeping 😴, Age: 30"
