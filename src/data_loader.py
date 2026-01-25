import numpy as np
from sklearn.datasets import fetch_openml
import os

# Dynamically find the path to the 'data' folder
# Get the directory where this script (data_loader.py) is
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root: e.g., .../Generative-Models-From-Scratch
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Construct the absolute path to the data file
MNIST_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'mnist_data.npz')


def download_and_save_mnist(filename=MNIST_DATA_PATH):
    # Create the directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(filename):
        print(f"{filename} already exists.")
        return

    print("Downloading MNIST from OpenML...")
    
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    except Exception as e:
        print(f"Download failed: {e}")
        return

    X = mnist.data
    y = mnist.target

    # Convert targets to integers (they come as strings)
    y = y.astype(int)

    # Normalise and Convert to float32
    print("Normalising and casting to float32...")
    X = X.astype(np.float32) / 255.0

    # Shape Check - we WANT (70000, 784)
    if X.ndim == 3:
        # Just in case a future version returns (N, 28, 28), flatten it
        X = X.reshape(X.shape[0], -1)

    print(f"\nFinal Data Shape: {X.shape} (Type: {X.dtype})")
    print(f"Data Range: [{np.min(X):.3f}, {np.max(X):.3f}]")

    # Split into train/test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print(f"\nSaving to {filename}...")
    np.savez_compressed(filename, 
                        X_train=X_train, y_train=y_train, 
                        X_test=X_test, y_test=y_test)
    print("Data saved successfully.")


def load_mnist(filename=MNIST_DATA_PATH):
    # Ensure the file exists before loading
    if not os.path.exists(filename):
        download_and_save_mnist(filename)
    
    data = np.load(filename)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']


if __name__ == "__main__":
    download_and_save_mnist()