from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and label encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = joblib.load("species_encoder.pkl")

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form inputs
        data = [float(x) for x in request.form.values()]
        
        # Ensure input matches expected shape (6 features)
        if len(data) != 6:
            return {"error": f"Expected 6 features, but got {len(data)}"}

        # Reshape for prediction
        features = np.array(data).reshape(1, -1)

        # Apply the label encoder transformation to the species (6th feature)
        weight = features[0, 5]  # Extract the weight (since it's the 6th feature)
        
        # Predict species (after encoding)
        prediction = model.predict(features)
        
        # Reverse the label encoding to get the actual species name
        predicted_species = encoder.inverse_transform([prediction[0]])[0]

        return {"prediction": predicted_species}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    app.run(debug=True)
