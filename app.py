import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# --- Load Model and Scaler ---
MODEL_PATH = 'placement_model.pkl'
SCALER_PATH = 'scaler.pkl'

# Check if model and scaler files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print(f"Error: Model file ({MODEL_PATH}) or Scaler file ({SCALER_PATH}) not found.")
    print("Please ensure 'placement_model.pkl' and 'scaler.pkl' are in the same directory as 'app.py'.")
    #exit as app can't run
    exit()

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit()


# --- Routes ---

@app.route('/', methods=['GET', 'POST']) # Allow both GET (initial load) and POST (form submission)
def home():
    prediction_result = None
    error_message = None
    # Initialize input values for form persistence
    cgpa = ''
    iq = ''
    attendance = ''

    if request.method == 'POST':
        try:
            # Get data from standard HTML form submission (request.form)
            cgpa = float(request.form['cgpa'])
            iq = float(request.form['iq'])
            attendance = float(request.form['attendance'])

            # Create a NumPy array for prediction (as model had array with 1 row and 3 columns)
            features = np.array([[cgpa, iq, attendance]])

            # Scale the input features using the loaded scaler
            scaled_features = scaler.transform(features)

            # Make prediction
            prediction = model.predict(scaled_features)[0] # [0] to get the single prediction value

            # Determine prediction label
            prediction_label = "PLUGGED" if prediction == 1 else "NOT PLUGGED"
            prediction_result = f"Prediction: {prediction_label}"

        except ValueError:
            error_message = "Invalid input. Please enter numbers for all fields."
        except KeyError as e:
            error_message = f"Missing form field: {e}. Please ensure all fields are filled."
        except Exception as e:
            # Catch any other unexpected errors during prediction
            error_message = f"An error occurred during prediction: {str(e)}"

    # Render the template, passing any results or errors
    return render_template('index.html',
                           prediction_result=prediction_result,
                           error_message=error_message,
                           cgpa=cgpa, # Pass back inputs to keep them in the form
                           iq=iq,
                           attendance=attendance)

# Run the Flask app
if _name_ == '_main_':
    pass 
   
