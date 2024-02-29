# Fraud Classification Web App

### Overview
This project is a web application that predicts fraud using a Random Forest model. It takes four input features (type, amount, old balance, new balance) and predicts whether the transaction is fraudulent or not, along with the probability of fraud.

### Steps
- Data Preprocessing: The input data is normalized using a `StandardScaler`.
- Model Training: The Random Forest model is trained on the normalized data. The model is saved as 'random_forest_modelf.joblib'.
- Web App Development: Flask is used to create the web application.
  - The app has two routes: `/` for the input form and `/predict` for the prediction result.
- Prediction Function: The `predict_fraud` function takes the input data, loads the model and scaler, and makes a prediction. It returns the predicted label ('fraud' or 'not fraud') and the probabilities of each class.
- Frontend: The input form (`index.html`) takes user inputs for the four features. The result page (`result.html`) displays the prediction result and probabilities.

### Usage
- Install the required packages: `pip install -r requirements.txt`.
- Run the Flask app: `python app.py`.
- Access the web app in your browser at `http://127.0.0.1:5000/`.

### Web App
![Screenshot (24)](https://github.com/HassaanIK/Fraud-Classification/assets/139614780/b63ab92e-d9dd-4b6a-b200-b9a57c348d5e)
![Screenshot (25)](https://github.com/HassaanIK/Fraud-Classification/assets/139614780/a0410b95-3234-4304-96f2-3b416670dc93)
