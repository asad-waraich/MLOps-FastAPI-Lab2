import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def load_data():
    """
    Generate a synthetic classification dataset.
    The dataset will have 4 features and 3 classes, similar to Iris.
    Returns:
        X (numpy.ndarray): The features of the synthetic dataset.
        y (numpy.ndarray): The target values (classes 0, 1, or 2).
    """
    # Generate a dataset with 200 samples, 4 features, and 3 classes
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3, # <-- This is the only change
        n_redundant=0,
        n_classes=3,
        random_state=42
    )
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    return X_train, X_test, y_train, y_test