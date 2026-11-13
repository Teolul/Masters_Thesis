"""Gaussian process prediction transformation helpers."""

import numpy as np

def inverse_mean_transform(y_red_scaled, scaler, pca):
    """Restore predicted means to the original output space.

    Args:
        y_red_scaled: Predicted means after scaling and optional PCA reduction.
        scaler: Fitted output scaler to invert.
        pca: Fitted PCA transform to invert, or ``None`` when PCA was not used.

    Returns:
        Predicted means after the available inverse transforms.
    """
    y_red = scaler.inverse_transform(y_red_scaled)
    if pca != None:
        y_red = pca.inverse_transform(y_red)
    return y_red


def inverse_std_transform(std_red_scaled, scaler, pca):
    """Restore predicted standard deviations to the original output space.

    Args:
        std_red_scaled: Predicted standard deviations after scaling and optional PCA reduction.
        scaler: Fitted output scaler whose scale parameters are used for inversion.
        pca: Fitted PCA transform used to propagate diagonal variances, or ``None``.

    Returns:
        Predicted standard deviations in the original output space.
    """
    if str(scaler) == "StandardScaler()":
        Y_std_red = std_red_scaled * scaler.scale_
    else: # MinMaxScaler
        Y_std_red = std_red_scaled * (scaler.data_max_ - scaler.data_min_)

    if pca != None:
        W = pca.components_
        latent_var = Y_std_red**2
        Y_var_red = latent_var @ (W**2)
        Y_std_red = np.sqrt(Y_var_red)

    return Y_std_red


def gp_predict_per_function(X, gpr_list, pca_list, x_scaler, y_scalers,
                            output_template=None):
    """Predict all transfer functions with one fitted GP per function.

    Args:
        X: Input samples in the original input space.
        gpr_list: Fitted Gaussian process model for each transfer function.
        pca_list: Per-function PCA transforms, or None when PCA was not used.
        x_scaler: Fitted input scaler, or None when inputs are not scaled.
        y_scalers: Per-function output scalers.
        output_template: Optional array whose shape and dtype should be retained.

    Returns:
        Prediction and predictive standard-deviation arrays with shape
        ``(samples, functions, wavelengths)`` in the original output space.
    """
    X_scaled = x_scaler.transform(X) if x_scaler is not None else X
    predictions = []
    standard_deviations = []

    for function_idx, gp in enumerate(gpr_list):
        pred_scaled, std_scaled = gp.predict(X_scaled, return_std=True)
        pca = pca_list[function_idx] if pca_list is not None else None
        predictions.append(
            inverse_mean_transform(pred_scaled, y_scalers[function_idx], pca)
        )
        standard_deviations.append(
            inverse_std_transform(std_scaled, y_scalers[function_idx], pca)
        )

    output_shape = (len(X), len(gpr_list), predictions[0].shape[1])
    if output_template is None:
        predictions_out = np.zeros(output_shape)
        std_out = np.zeros(output_shape)
    else:
        predictions_out = np.zeros_like(output_template)
        std_out = np.zeros_like(output_template)
    for function_idx, (pred, std) in enumerate(zip(predictions, standard_deviations)):
        predictions_out[:, function_idx, :] = pred
        std_out[:, function_idx, :] = std
    return predictions_out, std_out


def gp_predict_components(X, gp_list, x_scaler, y_scaler, pca):
    """Predict one transfer function represented by one GP per component.

    Args:
        X: Input samples in the original input space.
        gp_list: Fitted Gaussian process model for each component.
        x_scaler: Fitted input scaler, or None when inputs are not scaled.
        y_scaler: Fitted output scaler for the component predictions.
        pca: Fitted PCA transform used to reconstruct the transfer function.

    Returns:
        Prediction and predictive standard-deviation arrays in the original
        output space.
    """
    X_scaled = x_scaler.transform(X) if x_scaler is not None else X
    component_predictions = [gp.predict(X_scaled, return_std=True) for gp in gp_list]
    pred_scaled = np.column_stack([mean for mean, _ in component_predictions])
    std_scaled = np.column_stack([std for _, std in component_predictions])
    return (
        inverse_mean_transform(pred_scaled, y_scaler, pca),
        inverse_std_transform(std_scaled, y_scaler, pca),
    )


def gp_predict_combined(X, combined_emulator, function_names, pca_list,
                        x_scaler, y_scalers):
    """Predict all functions from a dictionary of component-level GP models.

    Args:
        X: Input samples in the original input space.
        combined_emulator: Mapping from function name to component GP models.
        function_names: Transfer-function names in output order.
        pca_list: Per-function PCA transforms.
        x_scaler: Fitted input scaler, or None when inputs are not scaled.
        y_scalers: Per-function output scalers.

    Returns:
        Prediction and predictive standard-deviation arrays with shape
        ``(samples, functions, wavelengths)`` in the original output space.
    """
    predictions = []
    standard_deviations = []
    for function_idx, function_name in enumerate(function_names):
        pred, std = gp_predict_components(
            X,
            combined_emulator[function_name],
            x_scaler,
            y_scalers[function_idx],
            pca_list[function_idx] if pca_list is not None else None,
        )
        predictions.append(pred)
        standard_deviations.append(std)
    return np.stack(predictions, axis=1), np.stack(standard_deviations, axis=1)


