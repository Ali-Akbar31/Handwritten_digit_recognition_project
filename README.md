# 🧠 End-to-End Handwritten Digit Recognition System

## 📌 Project Overview
This project is a complete, full-stack Deep Learning application built to accurately recognize handwritten digits (0-9) in real-time. It bridges the gap between complex Machine Learning models and user-friendly web interfaces. 

Trained on the standard MNIST dataset using a custom Convolutional Neural Network (CNN), the application doesn't just predict digits; it features robust image preprocessing and a built-in SQLite database to track, log, and display prediction history seamlessly.

## 🚀 Key Features
* **Dual Input Methods:** Users can either draw digits on a responsive interactive canvas or upload pre-existing image files.
* **Intelligent Preprocessing:** Built-in dynamic color inversion. The system calculates image mean pixels and automatically inverts white backgrounds to black, ensuring the model always receives optimal data formatting.
* **Prediction History & Logging:** A fully integrated SQLite backend (`classification_history.db`) that automatically logs the timestamp, predicted digit, model confidence score, and input method, saving the actual image as a BLOB for future review.
* **History Dashboard:** A dedicated web route (`/prediction_history`) to view past interactions and monitor model performance in real-world scenarios.

## 🛠️ Technology Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Backend Web Framework:** Flask (Python)
* **Database:** SQLite3
* **Data Processing:** NumPy, Pillow (PIL), Matplotlib
* **Frontend:** HTML5, CSS3, JavaScript (Canvas API)

## 🏗️ Model Architecture (CNN)
The prediction engine relies on a multi-layer Convolutional Neural Network designed for low-latency inference:
1. **Conv2D Layer 1:** 32 filters (3x3), ReLU activation + MaxPooling (2x2)
2. **Conv2D Layer 2:** 64 filters (3x3), ReLU activation + MaxPooling (2x2)
3. **Conv2D Layer 3:** 64 filters (3x3), ReLU activation
4. **Flatten Layer:** Converting 2D matrices to 1D vectors
5. **Dense Layer:** 64 neurons, ReLU activation
6. **Output Layer:** 10 neurons (0-9 classes), Softmax activation for probability distribution

## 📂 Repository Structure
```text
├── app.py                      # Core Flask application and API routes
├── model_training.py           # CNN model architecture and training script
├── digit_model.keras           # Pre-trained deep learning model
├── classification_history.db   # SQLite database for logging predictions
├── templates/                  # Frontend HTML templates
│   ├── index.html              # Main drawing/upload interface
│   └── history.html            # Dashboard to view logged predictions
└── README.md                   # Project documentation

🔧 Installation & Setup
1. Clone the repository:
git clone [https://github.com/YourUsername/Handwritten_digit_recognition_project.git](https://github.com/YourUsername/Handwritten_digit_recognition_project.git)
cd Handwritten_digit_recognition_project

2. Install required dependencies:
pip install tensorflow numpy flask pillow matplotlib

3. Run the Flask Server:
python app.py

4. Access the Application:
Open your web browser and navigate to: http://127.0.0.1:5000/

Developed by Ali Akbar | Freelance Data Scientist & ML Engineer
