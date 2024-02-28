import pandas as pd
from scipy import stats
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(file_path):
        data = pd.read_csv(file_path)
        data.drop(columns=['step','nameOrig','nameDest','oldbalanceDest','newbalanceDest','isFlaggedFraud'], inplace=True)
        category_labels = {
            'CASH_OUT': 1,
            'PAYMENT': 2,
            'CASH_IN': 3,
            'TRANSFER': 4,
            'DEBIT': 5
        }
        data['type'] = data['type'].replace(category_labels)
        data['type'] = data['type'].astype(float)


        z_scores = stats.zscore(data[['type','amount', 'oldbalanceOrg', 'newbalanceOrig']])
        # Absolute z-score threshold
        threshold = 3
        filtered_entries = (abs(z_scores) < threshold).all(axis=1)
        # Filtered dataframe without outliers
        data_filtered = data[filtered_entries]

        # Split the data into features and targets
        features_n = data_filtered.drop('isFraud', axis=1).values
        targets_n = data_filtered['isFraud'].values

        features = torch.tensor(features_n, dtype=torch.float32)
        targets = torch.tensor(targets_n, dtype=torch.float32)


        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)

        # Normalize the data
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_val_norm = scaler.transform(X_val)
        joblib.dump(scaler, 'models\\standard_scaler_1.joblib')
        print('Preprocessing Done!')
        
        return X_train_norm, X_val_norm, y_train, y_val
        