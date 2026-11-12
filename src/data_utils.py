"""Data loading and splitting helpers used by the thesis workflows."""

from pathlib import Path
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split

def inspect_metadata(path):
    """Print HDF5 metadata and return input and output names.

    Args:
        path: Path to the HDF5 file.

    Returns:
        A tuple containing input parameter names and output function names.
    """
    param_names = []
    function_names = []

    with h5py.File(path, "r") as f:
        print("Keys in train_file:", list(f.keys()))
        print("\nAttributes in LUTheader (inputs):")
        for key, value in f["LUTheader"].attrs.items():
            print(f"  {key}: {value}")
        print("\nAttributes in train_file (outputs):")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        print("\nLUTheader shape:", f["LUTheader"].shape)
        print("LUTdata shape:", f["LUTdata"].shape)
        print("wvl shape:", f["wvl"].shape)

        for param in f["LUTheader"].attrs["varnames"].split(","):
            param_names.append(param.strip())

        for func in f.attrs["outnames"].split(","):
            function_names.append(func.strip())

    return param_names, function_names


def load_train_h5(path):
    """Load inputs, outputs, and wavelengths from a training HDF5 file.

    Args:
        path: Path to the training HDF5 file.

    Returns:
        A tuple of input array ``X`` with shape ``(n_samples, n_features)``,
        output array ``Y`` with shape ``(n_samples, n_features)``, and the
        wavelength array.
    """
    with h5py.File(path, "r") as f:
        Y = f["LUTdata"][:]      # outputs
        X = f["LUTheader"][:]    # inputs
        wvl = f["wvl"][:]        # wavelengths

    return X, Y, wvl


def load_test_csv(path):
    """Load a test CSV file and transpose it into sample-major form.

    Args:
        path: Path to the CSV file.

    Returns:
        Input array with shape ``(n_samples, n_features)``.
    """
    df = pd.read_csv(path, header=None)
    X = df.to_numpy()
    return X.T


def load_csv_last_id(path):
    """Return the largest logged ID from a CSV file.

    Args:
        path: Path to the results CSV file.

    Returns:
        The maximum value in the ``id`` column, or 0 if the file is missing
        or the column contains no IDs.
    """

    if Path(path).exists():
        results_df = pd.read_csv(path)
        last_id = results_df["id"].max()
        if pd.isna(last_id):
            last_id = 0
    else:
        last_id = 0

    return last_id


def train_val_test_split(X, Y, wavelengths, verbose=True):
    """Split inputs and outputs into training, validation, and test sets.

    Args:
        X: Inputs with shape ``(n_samples, n_inputs)``.
        Y: Outputs with shape ``(n_samples, n_functions, n_wavelengths)``.
        wavelengths: Wavelength array used for shape reporting.
        verbose: Whether to print input and split shapes.

    Returns:
        Training, validation, and test inputs, followed by their outputs.
    """
    # first split: train (80%) and temp (20%)
    X_tr, X_temp, Y_tr, Y_temp = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)

    # second split: validation (10%) and test (10%)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, shuffle=True, random_state=42)

    if verbose:
        print("X shape:", X.shape)
        print("Y shape:", Y.shape)
        print("wavelengths shape:", wavelengths.shape)
        print()
        print("X_tr shape:", X_tr.shape)
        print("X_val shape:", X_val.shape)
        print("X_test shape:", X_test.shape)
        print()
        print("Y_tr shape:", Y_tr.shape)
        print("Y_val shape:", Y_val.shape)
        print("Y_test shape:", Y_test.shape)
    
    return X_tr, X_val, X_test, Y_tr, Y_val, Y_test


