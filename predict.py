import joblib


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
