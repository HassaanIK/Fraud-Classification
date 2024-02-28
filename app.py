from flask import Flask, request, render_template, jsonify
from predict import predict_fraud
import joblib


model = joblib.load('models\\random_forest_modelf.joblib')
scaler = joblib.load('models\\standard_scaler_f.joblib')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    type = float(request.form['type'])
    amount = float(request.form['amount'])
    old_balance = float(request.form['old_balance'])
    new_balance = float(request.form['new_balance'])

    input_data = {
        'type': type,
        'amount': amount,
        'old_balance': old_balance,
        'new_balance': new_balance
    }

    # Call the prediction function
    predicted_label, prediction, probabilities = predict_fraud(input_data, model, scaler)

    return render_template('result.html', prediction=prediction, probabilities=probabilities)

if __name__ == '__main__':
    app.run(debug=True)

