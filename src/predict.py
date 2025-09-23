import joblib
from pathlib import Path

# Point to the new synthetic model file
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "synthetic_model.pkl"

# Load the model once when the module is imported
model = joblib.load(MODEL_PATH)

def predict_data(X):
    """
    Predict the class labels for the input data using the pre-loaded model.
    """
    y_pred = model.predict(X)
    return y_pred