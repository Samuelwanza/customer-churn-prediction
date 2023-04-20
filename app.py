from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    # Get the data from the request
    data = request.get_json(force=True)

    # Load the XGBClassifier model
    model = joblib.load('/customer_churn.pkl')

    # Extract the features from the data
    features = np.array(data['features']).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(features)[0]

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
