import numpy as np
from sklearn.datasets import fetch_openml
import os

# Dynamically find the path to the 'data' folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# File paths
MNIST_PATH = os.path.join(DATA_DIR, 'mnist_data.npz')
FASHION_MNIST_PATH = os.path.join(DATA_DIR, 'fashion_mnist_data.npz')


def _ensure_directory_exists(filename):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)


def download_and_process(openml_name, save_path):
    """ Generic helper to download, normalise and save OpenML datasets. """
    _ensure_directory_exists(save_path)

    if os.path.exists(save_path):
        print(f"{save_path} already exists. Skipping download.")
        return

    print(f"Downloading {openml_name} from OpenML...")
    try:
        data = fetch_openml(openml_name, version=1, as_frame=False, parser='auto')
    except Exception as e:
        print(f"Download failed: {e}")
        return

    X = data.data
    y = data.target

    # Convert targets to integers
    y = y.astype(int)

    # Normalise to [0, 1] and float32
    print("Normalising and casting to float32...")
    X = X.astype(np.float32) / 255.0

    # Shape Check
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)

    print(f"\nFinal Data Shape: {X.shape} (Type: {X.dtype})")

    # Standard Split (60k train, 10k test for both datasets)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print(f"Saving to {save_path}...")
    np.savez_compressed(save_path, 
                        X_train=X_train, y_train=y_train, 
                        X_test=X_test, y_test=y_test)
    print("Data saved successfully.")


def download_and_save_mnist():
    download_and_process('mnist_784', MNIST_PATH)


def load_mnist():
    if not os.path.exists(MNIST_PATH):
        download_and_save_mnist()
    data = np.load(MNIST_PATH)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']


def download_and_save_fashion_mnist():
    download_and_process('Fashion-MNIST', FASHION_MNIST_PATH)


def load_fashion_mnist():
    if not os.path.exists(FASHION_MNIST_PATH):
        download_and_save_fashion_mnist()
    data = np.load(FASHION_MNIST_PATH)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']


if __name__ == "__main__":
    # download_and_save_mnist()
    download_and_save_fashion_mnist()