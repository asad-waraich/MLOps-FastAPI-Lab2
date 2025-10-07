from sklearn.svm import SVC
import joblib
from data import load_data, split_data
from pathlib import Path

def fit_model(X_train, y_train):
    """
    Train a Support Vector Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    # Use SVC instead
    svm_classifier = SVC(kernel='rbf', random_state=42)
    svm_classifier.fit(X_train, y_train)

    # Define the path to the model directory
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "model"
    
    # Create the model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the full path for the model file and save it
    model_path = model_dir / "synthetic_model.pkl"
    joblib.dump(svm_classifier, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)