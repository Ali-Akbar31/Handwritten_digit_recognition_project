from flask import Flask, render_template, request, jsonify, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3
import datetime
import io
import base64
import re

app = Flask(__name__)

# Load Model (Simple approach)
try:
    model = tf.keras.models.load_model('digit_model.keras')
    print("Model loaded successfully")
except:
    print("Error: 'digit_model.keras' file not found!")

# --- Database Functions ---
def execute_query(query, params=(), fetch=False):
    with sqlite3.connect('classification_history.db') as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        if fetch:
            return cursor.fetchall()
        conn.commit()

def setup_database():
    query = '''CREATE TABLE IF NOT EXISTS prediction_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                predicted_value INTEGER,
                confidence_score REAL,
                input_method TEXT,
                image_blob BLOB)'''
    execute_query(query)

def store_prediction_record(digit, confidence, method, image_data):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query = '''INSERT INTO prediction_records 
               (timestamp, predicted_value, confidence_score, input_method, image_blob)
               VALUES (?, ?, ?, ?, ?)'''
    execute_query(query, (current_time, int(digit), float(confidence), method, image_data))

# --- Prediction Logic (DO NOT TOUCH - Maintains Accuracy) ---
def get_digit_prediction(image):
    # 1. Grayscale & Resize
    image = image.convert('L').resize((28, 28))
    image_array = np.array(image)
    
    # 2. Invert colors if needed (White background -> Black background)
    if np.mean(image_array) > 127:
        image_array = 255 - image_array
        
    # 3. Normalize (0-1 range) & Reshape
    image_array = image_array.astype('float32') / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    
    # 4. Predict
    prediction = model.predict(image_array, verbose=0)
    return np.argmax(prediction), float(np.max(prediction))

# --- Routes ---
@app.route('/')
def home():
    setup_database() # Ensure DB exists on start
    return render_template('index.html')

@app.route('/upload_prediction', methods=['POST'])
def upload_prediction():
    file = request.files.get('image_file')
    if not file: return jsonify({'error': 'No file'})
    
    image = Image.open(file.stream)
    digit, confidence = get_digit_prediction(image)
    
    # Save for history
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    store_prediction_record(digit, confidence, 'file_upload', img_byte_arr.getvalue())
    
    return jsonify({'success': True, 'digit': int(digit), 'confidence': round(confidence * 100, 2)})

@app.route('/drawing_prediction', methods=['POST'])
def drawing_prediction():
    # Clean base64 data
    data = re.sub('^data:image/.+;base64,', '', request.json['canvas_data'])
    image = Image.open(io.BytesIO(base64.b64decode(data)))
    
    digit, confidence = get_digit_prediction(image)
    
    # Save for history
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    store_prediction_record(digit, confidence, 'canvas_drawing', img_byte_arr.getvalue())
    
    return jsonify({'success': True, 'digit': int(digit), 'confidence': round(confidence * 100, 2)})

@app.route('/prediction_history')
def prediction_history():
    records = execute_query('SELECT id, timestamp, predicted_value, confidence_score, input_method FROM prediction_records ORDER BY timestamp DESC', fetch=True)
    return render_template('history.html', records=records)

@app.route('/record_image/<int:record_id>')
def record_image(record_id):
    conn = sqlite3.connect('classification_history.db')
    cursor = conn.cursor()
    data = cursor.execute('SELECT image_blob FROM prediction_records WHERE id = ?', (record_id,)).fetchone()
    conn.close()
    return send_file(io.BytesIO(data[0]), mimetype='image/png') if data else "Error"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)