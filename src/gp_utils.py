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


