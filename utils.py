from pathlib import Path
import time
import copy
import h5py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

import globals

# ----------------------------
# Utilities for loading data, preprocessing, and evaluation
# ----------------------------


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
    y_tr_pca_list = []
    y_val_pca_list = []

    for i in range(globals.N_FUNCTIONS):
        if kernel is not None:
            pca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, fit_inverse_transform=True, alpha=alpha, degree=degree)
        else:
            pca = PCA(n_components=n_components)
            
        # shape of single function: (n_samples, n_wavelengths)
        y_tr_pca = pca.fit_transform(y_tr[:, i, :])     # fit training here
        y_val_pca = pca.transform(y_val[:, i, :])       # transform validation with the same PCA fitted on training
        
        pca_list.append(pca)
        y_tr_pca_list.append(y_tr_pca)
        y_val_pca_list.append(y_val_pca)

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

    return pca_list, y_tr_pca_list, y_val_pca_list


def scale_input_data(x_tr, x_val, scale_type="standard"):
    """
    Scale the data using either standard scaling or min-max scaling
    - inputs: training inputs, validation inputs, scaling type
    - outputs: scaler, scaled training inputs, scaled validation inputs
    """
    print(f"---------- Scaling input data using {scale_type} scaling... ----------")

    scaler = StandardScaler() if scale_type == "standard" else MinMaxScaler()
    x_scaler = scaler.fit(x_tr)
    x_tr_scaled = x_scaler.transform(x_tr)
    x_val_scaled = x_scaler.transform(x_val)

    print("---------- Input data scaling completed. ----------\n")

    return scaler, x_tr_scaled, x_val_scaled


def scale_output_data(y_tr_list, y_val_list, scale_type="standard"):
    """
    Scale the output data using the provided scalers
    - inputs: list of training outputs, list of validation outputs, scaling type
    - outputs: list of scalers used for each output function, list of scaled training outputs, list of scaled validation outputs
    """
    print(f"---------- Scaling output data using {scale_type} scaling... ----------")

    y_scalers = []
    y_tr_scaled_list = []
    y_val_scaled_list = []
    for i in range(globals.N_FUNCTIONS):
        scaler = StandardScaler() if scale_type == "standard" else MinMaxScaler()
        y_tr_scaled = scaler.fit_transform(y_tr_list[i])
        y_val_scaled = scaler.transform(y_val_list[i])
        y_scalers.append(scaler)
        y_tr_scaled_list.append(y_tr_scaled)
        y_val_scaled_list.append(y_val_scaled)

    print("---------- Output data scaling completed. ----------\n")

    return y_scalers, y_tr_scaled_list, y_val_scaled_list


def build_mask(wavelengths):
    """
    Build a boolean mask to exclude certain wavelength ranges from evaluation
    - inputs: numpy array of wavelengths
    - outputs: boolean mask where True indicates wavelengths to include in evaluation
    """
    # wavelengths to exclude from error calculation: 931-945 nm, 1100-1160 nm, 1300-1500 nm, 1750-1980 nm, and >2420 nm
    mask = (
        ((wavelengths < 931) | (wavelengths > 945)) &
        ((wavelengths < 1100) | (wavelengths > 1160)) &
        ((wavelengths < 1300) | (wavelengths > 1500)) &
        ((wavelengths < 1750) | (wavelengths > 1980)) &
        (wavelengths < 2420)
    )
    return mask


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
    
    mask = build_mask(wavelengths)

    if axis is None:
        mre = np.mean(
            np.abs(y_pred[:, :, mask] - y_true[:, :, mask]) / (np.abs(y_true[:, :, mask]) + epsilon)
        )
    elif axis == 2:
        mre = np.mean(
            np.abs(y_pred[:, :, mask] - y_true[:, :, mask]) / (np.abs(y_true[:, :, mask]) + epsilon),
            axis=(0, 2)
        )
    elif axis == 1:
        mre = np.mean(
            np.abs(y_pred - y_true) / (np.abs(y_true) + epsilon),
            axis=(0, 1)
        )
    elif axis == 0:
        mre = np.mean(
            np.abs(y_pred - y_true) / (np.abs(y_true) + epsilon),
            axis=0
        )
    else:
        raise ValueError("Invalid axis value. Must be None, 0, 1, or 2.")
    
    return mre


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

    mask = build_mask(wavelengths)

    if axis is None:
        mae = np.mean(
            np.abs(y_pred[:, :, mask] - y_true[:, :, mask])
        )
    elif axis == 2:
        mae = np.mean(
            np.abs(y_pred[:, :, mask] - y_true[:, :, mask]),
            axis=(0, 2)
        )
    elif axis == 1:
        mae = np.mean(
            np.abs(y_pred - y_true),
            axis=(0, 1)
        )
    elif axis == 0:
        mae = np.mean(
            np.abs(y_pred - y_true),
            axis=0
        )
    else:
        raise ValueError("Invalid axis value. Must be None, 0, 1, or 2.")
    
    return mae


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