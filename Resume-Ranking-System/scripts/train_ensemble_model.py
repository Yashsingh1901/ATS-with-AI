import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
import xgboost as xgb

def load_processed_data(data_path):
    """Load preprocessed training and test data."""
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    return X_train, X_test, y_train, y_test

def train_ensemble_model(X_train, y_train, X_test, y_test):
    """Train an ensemble model using RandomForest and XGBoost."""
    print("Training ensemble model...")
    
    # Initialize individual models
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    
    # Train individual models
    print("Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    print(f"Random Forest - MSE: {rf_mse:.4f}, R2 Score: {rf_r2:.4f}")
    
    print("Training XGBoost model...")
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)
    print(f"XGBoost - MSE: {xgb_mse:.4f}, R2 Score: {xgb_r2:.4f}")
    
    # Create and train ensemble model (VotingRegressor)
    ensemble_model = VotingRegressor([
        ('rf', rf_model),
        ('xgb', xgb_model)
    ])
    
    print("Training ensemble model...")
    ensemble_model.fit(X_train, y_train)
    ensemble_pred = ensemble_model.predict(X_test)
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    print(f"Ensemble - MSE: {ensemble_mse:.4f}, R2 Score: {ensemble_r2:.4f}")
    
    # Create a comprehensive model dictionary containing all models
    model_dict = {
        'random_forest': rf_model,
        'xgboost': xgb_model,
        'ensemble': ensemble_model
    }
    
    return model_dict

def save_models(model_dict, model_path):
    """Save the trained models."""
    os.makedirs(model_path, exist_ok=True)
    
    # Save individual models
    joblib.dump(model_dict['random_forest'], os.path.join(model_path, 'random_forest_model.joblib'))
    joblib.dump(model_dict['xgboost'], os.path.join(model_path, 'xgboost_model.joblib'))
    
    # Save the ensemble model as the main model
    joblib.dump(model_dict['ensemble'], os.path.join(model_path, 'resume_ranker.joblib'))
    
    # Save the complete model dictionary for advanced usage
    joblib.dump(model_dict, os.path.join(model_path, 'resume_ranker_all_models.joblib'))
    
    print(f"Models saved to {model_path}")

if __name__ == "__main__":
    # Load data
    data_path = os.path.join("Resume-Ranking-System", "data", "kaggle", "processed")
    print(f"Loading data from {data_path}")
    X_train, X_test, y_train, y_test = load_processed_data(data_path)
    print(f"Loaded data: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    # Train ensemble model
    model_dict = train_ensemble_model(X_train, y_train, X_test, y_test)
    
    # Save models
    model_path = os.path.join("Resume-Ranking-System", "models")
    save_models(model_dict, model_path)
    
    print("Training complete - ensemble model is now the default model for predictions.") 