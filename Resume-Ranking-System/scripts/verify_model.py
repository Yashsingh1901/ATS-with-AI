import os
import joblib
import numpy as np

def verify_model():
    """Verify that the model file exists and can be loaded."""
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    model_path = os.path.join(project_root, "models", "resume_ranker.joblib")
    
    print(f"Checking for model at: {model_path}")
    
    # Check if the file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
    print(f"Model file size: {file_size:.2f} MB")
    
    # Try to load the model
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully: {type(model).__name__}")
        
        # Create dummy data to test prediction
        dummy_data = np.random.rand(1, 768)
        prediction = model.predict(dummy_data)
        print(f"Test prediction successful: {prediction}")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

if __name__ == "__main__":
    result = verify_model()
    print(f"\nVerification result: {'SUCCESS' if result else 'FAILED'}") 