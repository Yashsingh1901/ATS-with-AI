import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib

def load_processed_data(data_path):
    """Load preprocessed training and test data."""
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test):
    """Train Random Forest model."""
    # Initialize and train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model training completed!")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return model

def save_model(model, model_path):
    """Save the trained model."""
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(model, os.path.join(model_path, 'resume_ranker.joblib'))
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Load data
    data_path = os.path.join("Resume-Ranking-System", "data", "kaggle", "processed")
    print(f"Loading data from {data_path}")
    X_train, X_test, y_train, y_test = load_processed_data(data_path)
    print(f"Loaded data: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Save model
    model_path = os.path.join("Resume-Ranking-System", "models")
    save_model(model, model_path) 