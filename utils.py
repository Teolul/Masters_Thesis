from pathlib import Path
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import globals

# ----------------------------
# Utilities for loading data, preprocessing, and evaluation
# ----------------------------


def make_scaler(scale_type="standard"):
    """
    Create a scaler using either standard scaling or min-max scaling
    - inputs: scaling type
    - outputs: scaler object
    """
    scale_type = scale_type.lower()

    if scale_type == "standard":
        return StandardScaler()
    if scale_type in ("minmax", "min-max"):
        return MinMaxScaler()

    raise ValueError("Invalid scale_type. Must be 'standard' or 'minmax'.")


def inspect_metadata(path):
    """
    Inspect the metadata of a .h5 file
    - inputs: path to the .h5 file
    - outputs: prints the keys and shapes of the datasets in the .h5 file, returns the list of input parameter names and output function names
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
    """
    Load a training .h5 file
    - inputs: path to the .h5 file
    - outputs: numpy array X (inputs) of shape (n_samples, n_features), numpy array Y (outputs) of shape (n_samples, n_features), numpy array wvl (wavelengths)
    """
    with h5py.File(path, "r") as f:
        Y = f["LUTdata"][:]      # outputs
        X = f["LUTheader"][:]    # inputs
        wvl = f["wvl"][:]        # wavelengths

    return X, Y, wvl


def load_test_csv(path):
    """
    Load a test .csv file
    - inputs: path to the .csv file
    - outputs: numpy array X (inputs) of shape (n_samples, n_features)
    """
    df = pd.read_csv(path, header=None)
    X = df.to_numpy()
    return X.T


def load_csv_last_id(path):
    """
    Load a CSV file and return the last id used for logging results
    - inputs: path to the CSV file
    - outputs: last id used in the CSV file, or 0 if the file does not exist or is empty
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
    wavelengths = wavelengths.squeeze()
    Y_resh = Y.reshape(-1, Y.shape[1] // len(wavelengths), len(wavelengths))

    # first split: train (80%) and temp (20%)
    X_tr, X_temp, Y_tr, Y_temp = train_test_split(X, Y_resh, test_size=0.2, shuffle=True, random_state=42)

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


def apply_pca(y_tr, y_val, n_components=10, kernel=None, gamma=1e-2, alpha=0.1, degree=3):
    """
    Apply PCA or KernelPCA to each function separately, retaining n_components components.
    - y_tr: training outputs of shape (n_samples, n_functions, n_wavelengths)
    - y_val: validation outputs of shape (n_samples, n_functions, n_wavelengths)
    - n_components: number of PCA components to retain
    - kernel: if None, use regular PCA, otherwise specify the kernel type for KernelPCA (e.g., 'rbf', 'poly', etc.)
    - returns: list of PCA objects, list of transformed training outputs, list of transformed validation outputs
    """

    print(f"---------- Applying {'KernelPCA' if kernel is not None else 'PCA'} with n_components={n_components} to each function separately... ----------")
    pca_list = []
    y_tr_pca = np.zeros((y_tr.shape[0], y_tr.shape[1], n_components))
    y_val_pca = np.zeros((y_val.shape[0], y_val.shape[1], n_components))

    for i in range(globals.N_FUNCTIONS):
        if kernel is not None:
            pca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, fit_inverse_transform=True, alpha=alpha, degree=degree)
        else:
            pca = PCA(n_components=n_components)
            
        y_tr_pca[:, i, :] = pca.fit_transform(y_tr[:, i, :]) # fit training here
        y_val_pca[:, i, :] = pca.transform(y_val[:, i, :]) # transform validation with the same PCA fitted on training
        pca_list.append(pca)

    # print amount of explained variance and number of components for each function
    if kernel is None:
        total_explained_variance = 0
        print("  Regular PCA used, displaying results:")
        for i, pca in enumerate(pca_list):
            explained_variance = pca.explained_variance_ratio_.sum()
            total_explained_variance += explained_variance
            print(f"  Function {i+1}: Explained variance = {explained_variance:.4f}")
            print(f"  Number of components retained: {pca.n_components_}")
            print()

        print(f"  Total explained variance = {total_explained_variance:.4f}")

    print("---------- PCA application completed. ----------\n")

    return pca_list, y_tr_pca, y_val_pca


def scale_input_data(x_tr, x_val, scale_type="standard"):
    """
    Scale the data using either standard scaling or min-max scaling
    - inputs: training inputs, validation inputs, scaling type
    - outputs: scaler, scaled training inputs, scaled validation inputs
    """
    print(f"---------- Scaling input data using {scale_type} scaling... ----------")

    scaler = make_scaler(scale_type)
    x_scaler = scaler.fit(x_tr)
    x_tr_scaled = x_scaler.transform(x_tr)
    x_val_scaled = x_scaler.transform(x_val)

    print("---------- Input data scaling completed. ----------\n")

    return scaler, x_tr_scaled, x_val_scaled


def scale_output_data(y_tr, y_val, scale_type="standard"):
    """
    Scale the output data using the provided scalers
    - inputs: training outputs of shape (n_samples, n_functions, n_wavelengths), validation outputs of shape (n_samples, n_functions, n_wavelengths), scaling type
    - outputs: list of scalers used for each output function, scaled training outputs of shape (n_samples, n_functions, n_wavelengths), scaled validation outputs of shape (n_samples, n_functions, n_wavelengths)
    """
    print(f"---------- Scaling output data using {scale_type} scaling... ----------")

    y_scalers = []
    y_tr_scaled = np.zeros_like(y_tr)
    y_val_scaled = np.zeros_like(y_val)
    for i in range(globals.N_FUNCTIONS):
        scaler = make_scaler(scale_type)
        y_tr_scaled[:, i, :] = scaler.fit_transform(y_tr[:, i, :])
        y_val_scaled[:, i, :] = scaler.transform(y_val[:, i, :])
        y_scalers.append(scaler)

    print("---------- Output data scaling completed. ----------\n")

    return y_scalers, y_tr_scaled, y_val_scaled


def build_mask(wavelengths):
    """
    Build a boolean mask to exclude certain wavelength ranges from evaluation
    - inputs: numpy array of wavelengths
    - outputs: boolean mask where True indicates wavelengths to include in evaluation
    """
    wavelengths = np.asarray(wavelengths).squeeze()

    # wavelengths to exclude from error calculation: 931-945 nm, 1100-1160 nm, 1300-1500 nm, 1750-1980 nm, and >2420 nm
    mask = (
        ((wavelengths < 931) | (wavelengths > 945)) &
        ((wavelengths < 1100) | (wavelengths > 1160)) &
        ((wavelengths < 1300) | (wavelengths > 1500)) &
        ((wavelengths < 1750) | (wavelengths > 1980)) &
        (wavelengths < 2420)
    )
    return mask


def _reduce_metric(error, wavelengths, axis=None):
    """
    Reduce a metric error array using the same axis options used by MRE and MAE.
    """
    mask = build_mask(wavelengths)
    if mask.ndim != 1 or mask.shape[0] != error.shape[-1]:
        raise ValueError(
            "wavelengths must contain one value for each wavelength in y_true/y_pred."
        )

    if axis is None:
        return np.mean(error[:, :, mask])
    if axis == 2:
        return np.mean(error[:, :, mask], axis=(0, 2))
    if axis == 1:
        return np.mean(error, axis=(0, 1))
    if axis == 0:
        return np.mean(error, axis=0)

    raise ValueError("Invalid axis value. Must be None, 0, 1, or 2.")


def mre_score(y_true, y_pred, wavelengths, axis=None, epsilon=1e-8):
    """
    Mean Relative Error (MRE) metric
    - inputs: true values, predicted values, wavelengths, axis on which to compute the metric, epsilon (small constant to avoid division by zero)
    - output: MRE score, either as a global scalar or as an array depending on the axis parameter

    axis options:
        None -> global scalar
        2    -> per function
        1    -> per wavelength
        0    -> per function and per wavelength
    """
    error = np.abs(y_pred - y_true) / (np.abs(y_true) + epsilon)
    return _reduce_metric(error, wavelengths, axis=axis)


def mae_score(y_true, y_pred, wavelengths, axis=None):
    """
    Mean Absolute Error (MAE) metric
    - inputs: true values, predicted values, wavelengths, axis on which to compute the metric
    - output: MAE score, either as a global scalar or as an array depending on the axis parameter

    axis options:
        None -> global scalar
        2    -> per function
        1    -> per wavelength
        0    -> per function and per wavelength
    """
    error = np.abs(y_pred - y_true)
    return _reduce_metric(error, wavelengths, axis=axis)


def calculate_coverage(y_true, y_pred, y_std, n_std=2):
    """
    Calculates the percentage of true values falling within the GP uncertainty bands.
    - inputs: true values, predicted values, predicted standard deviation, n_std (number of standard deviations for the confidence interval)
    - outputs: global coverage percentage, coverage percentage per function
    """
    # define bounds: (n_samples, n_functions, n_wavelengths)
    lower_bound = y_pred - n_std * y_std
    upper_bound = y_pred + n_std * y_std

    # boolean mask: True if the value is within the interval
    is_inside = (y_true >= lower_bound) & (y_true <= upper_bound)

    # global coverage
    global_coverage = np.mean(is_inside) * 100

    # coverage per function
    # average across axis 0 (samples) and 2 (wavelengths)
    per_function_coverage = np.mean(is_inside, axis=(0, 2)) * 100

    return global_coverage, per_function_coverage
