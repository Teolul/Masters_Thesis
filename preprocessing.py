"""PCA and feature scaling helpers for model inputs and outputs."""

import numpy as np
import globals
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def apply_pca(y_tr, y_val, n_components=10, verbose=True):
    """Apply PCA separately to each function, retaining the requested components.

    Args:
        y_tr: Training outputs with shape ``(n_samples, n_functions, n_wavelengths)``.
        y_val: Validation outputs with shape ``(n_samples, n_functions, n_wavelengths)``.
        n_components: Number of PCA components to retain.
        verbose: Whether to print explained variance and component counts.

    Returns:
        PCA objects and transformed training and validation outputs, in that order.
    """

    print(f"---------- Applying PCA with n_components={n_components} to each function separately... ----------")
    pca_list = []
    y_tr_pca = np.zeros((y_tr.shape[0], y_tr.shape[1], n_components))
    y_val_pca = np.zeros((y_val.shape[0], y_val.shape[1], n_components))

    for i in range(globals.N_FUNCTIONS):
        pca = PCA(n_components=n_components)

        y_tr_pca[:, i, :] = pca.fit_transform(y_tr[:, i, :]) # fit training here
        y_val_pca[:, i, :] = pca.transform(y_val[:, i, :]) # transform validation with the same PCA fitted on training
        pca_list.append(pca)

    # print amount of explained variance and number of components for each function
    if verbose:
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
    """Scale input arrays using standard or min-max scaling.

    Args:
        x_tr: Training input array used to fit the scaler.
        x_val: Validation input array transformed with the fitted scaler.
        scale_type: Scaling choice; ``"standard"`` selects standard scaling,
            and other values select min-max scaling.

    Returns:
        The fitted scaler, scaled training inputs, and scaled validation inputs.
    """
    print(f"---------- Scaling input data using {scale_type} scaling... ----------")

    scaler = StandardScaler() if scale_type == "standard" else MinMaxScaler()
    x_scaler = scaler.fit(x_tr)
    x_tr_scaled = x_scaler.transform(x_tr)
    x_val_scaled = x_scaler.transform(x_val)

    print("---------- Input data scaling completed. ----------\n")

    return scaler, x_tr_scaled, x_val_scaled


def scale_input_data_feature_selection(x_tr, x_val, scale_type="standard"):
    """Scale feature-selected inputs separately for each transfer function.

    Args:
        x_tr: Training arrays, one ``(n_samples, n_features)`` array per function.
        x_val: Validation arrays corresponding to ``x_tr``.
        scale_type: Scaling choice; ``"standard"`` selects standard scaling,
            and other values select min-max scaling.

    Returns:
        Lists of fitted scalers, scaled training arrays, and scaled validation arrays.
    """
    print(f"---------- Scaling feature-selected input data using {scale_type} scaling... ----------")

    scalers = []
    x_tr_scaled = []
    x_val_scaled = []

    for i, (x_tr_func, x_val_func) in enumerate(zip(x_tr, x_val)):
        scaler = StandardScaler() if scale_type == "standard" else MinMaxScaler()
        scaler.fit(x_tr_func)
        x_tr_func_scaled = scaler.transform(x_tr_func)
        x_val_func_scaled = scaler.transform(x_val_func)
        scalers.append(scaler)
        x_tr_scaled.append(x_tr_func_scaled)
        x_val_scaled.append(x_val_func_scaled)
        print(f"Function {i}: {x_tr_func.shape[1]} features")

    print("---------- Feature-selected input data scaling completed. ----------\n")

    return scalers, x_tr_scaled, x_val_scaled


def scale_output_data(y_tr, y_val, scale_type="standard"):
    """Scale outputs with a separate scaler for each function.

    Args:
        y_tr: Training outputs with shape ``(n_samples, n_functions, n_wavelengths)``.
        y_val: Validation outputs with shape ``(n_samples, n_functions, n_wavelengths)``.
        scale_type: Scaling choice; ``"standard"`` selects standard scaling,
            and other values select min-max scaling.

    Returns:
        Per-function scalers, scaled training outputs, and scaled validation outputs.
    """
    print(f"---------- Scaling output data using {scale_type} scaling... ----------")

    y_scalers = []
    y_tr_scaled = np.zeros_like(y_tr)
    y_val_scaled = np.zeros_like(y_val)
    for i in range(globals.N_FUNCTIONS):
        scaler = StandardScaler() if scale_type == "standard" else MinMaxScaler()
        y_tr_scaled[:, i, :] = scaler.fit_transform(y_tr[:, i, :])
        y_val_scaled[:, i, :] = scaler.transform(y_val[:, i, :])
        y_scalers.append(scaler)

    print("---------- Output data scaling completed. ----------\n")

    return y_scalers, y_tr_scaled, y_val_scaled


