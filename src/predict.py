import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "iris_model.pkl"


def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)
    return y_pred
