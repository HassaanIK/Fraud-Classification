# Fraud Classification
### OVERVIEW
This flask web app is a fraud transaction classifier that takes input features and predicts if it's a fraud or not. It uses a machine learning model trained on features like transaction type, amount, old balance and new balance to make predictions. The user inputs these features into forms on the web app, and the app returns if the transaction is fraud or not.
### SPECIFICATIONS
- The data used for training is taken from Kaggle. It has 10 different features out of which 4 are used.
- The preprocessing done on this data is removal of outliers.
- The features are normalized using StandardScaler from scikit learn library.
- The machine learning algorithm used is Random Forest Classifier as it was giving the best accuracy out of all.
- The metrics used for evaluation is accuracy, 99.97% of which is achieved.
- No deep learning is applied as 99.97% accuracy is more than enough.
- The project uses Flask, a lightweight web framework for Python, to create the web application.
- The input features are normalized before being fed into the model for prediction.
### USAGE
```python
def predict_fraud(input_data, model, scaler):


    # Assuming input_data is a dictionary containing normalized values for each feature
    features = [[input_data['type'], input_data['amount'], input_data['old_balance'], input_data['new_balance']]]
    input_features_norm = scaler.transform(features)  # Use transform instead of fit_transform for test data
   
    # Make predictions
    predicted_label = model.predict(input_features_norm)[0]
    if predicted_label == 1:
        prediction = 'fraud'
    else:
        prediction = 'not fraud'
    probabilities = model.predict_proba(input_features_norm)[0]
   
    return predicted_label, prediction, probabilities
input_data = {
        'type': type,
        'amount': amount,
        'old_balance': old_balance,
        'new_balance': new_balance
    }


    # Call the prediction function
    predicted_label, prediction, probabilities = predict_fraud(input_data, model, scaler)


