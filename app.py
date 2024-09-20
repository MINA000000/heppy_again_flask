# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from the request body
    features = np.array(data['features'])  # Extract features from JSON
    prediction = model.predict([features])  # Use the model to predict
    return jsonify({'prediction': int(prediction[0])})  # Return the prediction as JSON

if __name__ == '__main__':
    app.run(debug=True)
