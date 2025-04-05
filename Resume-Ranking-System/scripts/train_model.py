import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import os
from datetime import datetime

def load_processed_data(data_path):
    """Load preprocessed training and test data."""
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model with MLflow tracking."""
    # Initialize MLflow
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("resume-ranking")
        
        with mlflow.start_run(run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Initialize and train model
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            
            print(f"Model training completed!")
            print(f"MSE: {mse:.4f}")
            print(f"R2 Score: {r2:.4f}")
            
            return model
    except Exception as e:
        print(f"Error with MLflow: {e}")
        # Still train the model even if MLflow fails
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
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
    model.save_model(os.path.join(model_path, 'resume_ranker.json'))
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Load data
    data_path = os.path.join("Resume-Ranking-System", "data", "kaggle", "processed")
    X_train, X_test, y_train, y_test = load_processed_data(data_path)
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Save model
    model_path = os.path.join("Resume-Ranking-System", "models")
    save_model(model, model_path) 