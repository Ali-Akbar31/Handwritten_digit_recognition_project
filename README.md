# Handwritten Digit Recognition System using Deep Learning (CNN)

## 📌 Project Overview
This project is an end-to-end Deep Learning application designed to recognize handwritten digits (0-9) with high accuracy. It features a custom-trained Convolutional Neural Network (CNN) model and an interactive web interface for real-time predictions.

## 🚀 Features
- **Real-time Recognition:** Users can draw digits on a web-based canvas and get instant results.
- **High Accuracy:** Trained on the classic MNIST dataset with optimized hyperparameters.
- **Web Integration:** A Flask-based backend to serve the model and handle user input.
- **Data Visualization:** Includes scripts for model performance analysis (Loss/Accuracy plots).

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow / Keras
- **Web Framework:** Flask
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML, CSS, JavaScript (Canvas API)

## 🧠 Model Architecture
The system uses a Convolutional Neural Network (CNN) which includes:
- **Convolutional Layers:** To extract spatial features from images.
- **Max-Pooling Layers:** To reduce dimensionality.
- **Dropout Layers:** To prevent overfitting.
- **Dense Layers:** Fully connected layers for classification.
- **Softmax Activation:** To provide probability scores for digits 0-9.

## 📂 Project Structure
- `app.py`: Flask application for deployment.
- `model.h5`: The pre-trained CNN model.
- `notebook.ipynb`: Data exploration and model training script.
- `static/`: Contains CSS and JS for the web interface.
- `templates/`: HTML files for the web app.

## 🔧 How to Run
1. Clone the repository: `git clone [Your Repo Link]`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Open `http://127.0.0.1:5000/` in your browser.

---
**Author:** Ali Akbar
