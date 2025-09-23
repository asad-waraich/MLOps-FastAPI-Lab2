import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data import load_data
from pathlib import Path

def create_visualization():
    """
    Loads the synthetic data, creates a pair plot to visualize
    the feature distributions and relationships, and saves the plot to a file.
    """
    print("Loading data...")
    X, y = load_data()

    # Create a pandas DataFrame for easier plotting with seaborn
    df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])
    df['Class'] = y

    print("Creating visualization...")
    # Create a pair plot colored by class
    sns.set_theme(style="ticks")
    pair_plot = sns.pairplot(df, hue='Class', palette='bright')
    pair_plot.fig.suptitle("Feature Distribution by Class", y=1.02) # Add a title

    # Define the output path in a new 'assets' folder
    output_dir = Path(__file__).resolve().parents[1] / "assets"
    output_dir.mkdir(parents=True, exist_ok=True) # Create assets dir if it doesn't exist
    output_path = output_dir / "data_distribution.png"

    # Save the plot
    pair_plot.savefig(output_path)
    print(f"Plot saved successfully to {output_path}")

if __name__ == "__main__":
    create_visualization()