from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import logging
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet import preprocess_input
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Disable template caching

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
MODEL_PATH = "model/bestmodel.keras"  # Path to your fine-tuned MobileNet model
model = load_model(MODEL_PATH)

# Define the list of class names
CLASS_NAMES = ["Benign", "Malignant"]

# Define image preprocessing for MobileNet
def preprocess_image(image_path):
    try:
        # Load the image using PIL
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)  # Preprocess for MobileNet
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    except Exception as e:
        logging.error(f"Error in preprocessing image: {str(e)}")
        return None

# Define a function for prediction
def predict_uploaded_image(image_path):
    try:
        # Preprocess the image
        img_array = preprocess_image(image_path)
        if img_array is None:
            return None

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class_index = int(predictions[0][0] > 0.5)  # Binary classification threshold
        predicted_class_probability = float(predictions[0][0])

        # Ensure the predicted class index is valid
        if predicted_class_index >= len(CLASS_NAMES):
            logging.error(f"Predicted class index {predicted_class_index} is out of range.")
            return None

        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = predicted_class_probability * 100 if predicted_class_index == 1 else (1 - predicted_class_probability) * 100

        logging.info(f'Predicted class: {predicted_class} with confidence: {confidence:.2f}%')
        return predicted_class, confidence

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return None

# Utility function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Add validation logic here
        if username == "admin" and password == "password":  # Replace with database validation
            session['user'] = username
            return redirect(url_for('home'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Save to database or perform other logic
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logging.info(f"File saved to: {file_path}")  # Log the file path

            # Predict
            result = predict_uploaded_image(file_path)
            if result is None:
                flash('Failed to process the image.')
                return redirect(request.url)

            predicted_class, confidence = result
            logging.info(f"Prediction result: {predicted_class}, {confidence}")  # Log the prediction result

            return render_template('predictions.html', result=predicted_class, confidence=f"{confidence:.2f}%", filepath=filename)

    return render_template('predictions.html', result=None, confidence=None)

if __name__ == '__main__':
    app.run(debug=True)