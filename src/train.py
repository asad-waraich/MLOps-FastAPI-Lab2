from sklearn.tree import DecisionTreeClassifier
import joblib
from data import load_data, split_data
from pathlib import Path

def fit_model(X_train, y_train):
    """
    Train a Decision Tree Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
    dt_classifier.fit(X_train, y_train)

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "iris_model.pkl"
    joblib.dump(dt_classifier, model_path)

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)
