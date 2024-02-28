from data_preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load

#data
file_path = 'data\\PS_20174392719_1491204439457_log.csv'
X_train_norm, X_val_norm, y_train, y_val = preprocess_data(file_path)

# Instantiate the model
model = RandomForestClassifier(n_estimators=200)

# Train the model
model.fit(X_train_norm, y_train)
print("Training Done!")

# Make predictions on the validation set
y_pred = model.predict(X_val_norm)
print('Testing')

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)

print("Accuracy:", accuracy)


# Save the model to a file
save_path = 'models\\random_forest_model1.joblib'
dump(model, save_path)
print('Saved at:', save_path)