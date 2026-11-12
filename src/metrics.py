"""Evaluation metrics and atmospheric reflectance calculations."""

import numpy as np

def retrieve_reflectance(Ltoa: np.ndarray, Y_pred: np.ndarray, sza: np.ndarray, n_wvl: int) -> np.ndarray:
    """Retrieve surface reflectance from TOA radiance and transfer functions.

    Calculates surface reflectance ($\rho$) from Top-Of-Atmosphere (TOA)
    radiance using the provided transfer functions.

    Args:
        Ltoa: TOA radiance with shape ``(n_samples, n_wvl)``.
        Y_pred: Predicted transfer-function blocks with shape
            ``(n_samples, 6 * n_wvl)``.
        sza: Solar zenith angle in degrees, broadcast-compatible with the
            sample axis.
        n_wvl: Number of spectral wavelengths.

    Returns:
        Surface reflectance with shape ``(n_samples, n_wvl)``.
    """
    # split transfer-function data
    L0 = Y_pred[:, 0:n_wvl]
    E_dir = Y_pred[:, n_wvl:2*n_wvl]
    E_dif = Y_pred[:, 2*n_wvl:3*n_wvl]
    Sa = Y_pred[:, 3*n_wvl:4*n_wvl]
    T_dir = Y_pred[:, 4*n_wvl:5*n_wvl]
    T_dif = Y_pred[:, 5*n_wvl:6*n_wvl]

    # compute total irradiance and total transmittance
    # Latex: E_{total} = E_{dir} \cdot \cos(SZA) + E_{dif}
    E_total = E_dir * np.cos(np.radians(sza)) + E_dif
    T_total = T_dir + T_dif

    # retrieve surface reflectance
    # Latex: \rho = \frac{\pi (L_{toa} - L_0)}{E_{total} T_{total} + \pi (L_{toa} - L_0) S_a}
    numerator = np.pi * (Ltoa - L0)
    denominator = E_total * T_total + np.pi * (Ltoa - L0) * Sa
    rho_ret = numerator / denominator
    return rho_ret


def mre_amlec(rho_ref: np.ndarray, rho_ret: np.ndarray, wvl: np.ndarray, axis: int | None = None) -> float | np.ndarray:
    """Calculate Scenario A MRE with the established absorption-band handling.

    Args:
        rho_ref: Reference reflectance with shape ``(n_wvl,)``.
        rho_ret: Retrieved reflectance with shape ``(n_samples, n_wvl)``.
        wvl: Wavelengths with shape ``(1, n_wvl)``.
        axis: ``None`` for the global mean or ``1`` for per-wavelength values.
            The absorption-band mask is applied to the global mean only.

    Returns:
        Mean relative error as a fraction (not multiplied by 100), or an array
        of per-wavelength values when ``axis=1``.
    """
    re = np.abs(rho_ret - rho_ref) / rho_ref

    if axis is None:
        mask = ~(
            ((wvl > 931) & (wvl < 945)) |
            ((wvl > 1100) & (wvl < 1160)) |
            ((wvl > 1300) & (wvl < 1500)) |
            ((wvl > 1750) & (wvl < 1980)) |
            (wvl > 2420)
        )
        re_mean = np.nanmean(re, axis=0, keepdims=True)
        return np.nanmean(re_mean.flatten()[mask.flatten()])
    elif axis == 1:
        return np.nanmean(re, axis=0)
    else:
        raise ValueError("Invalid axis value. Must be None or 1.")


def mae_amlec(rho_ref: np.ndarray, rho_ret: np.ndarray, wvl: np.ndarray, axis: int | None = None) -> float | np.ndarray:
    """Calculate Scenario A MAE with the established absorption-band handling.

    Args:
        rho_ref: Reference reflectance with shape ``(n_wvl,)``.
        rho_ret: Retrieved reflectance with shape ``(n_samples, n_wvl)``.
        wvl: Wavelengths with shape ``(1, n_wvl)``.
        axis: ``None`` for the global mean or ``1`` for per-wavelength values.
            The absorption-band mask is applied to the global mean only.

    Returns:
        Mean absolute error or an array of per-wavelength values when
        ``axis=1``.
    """
    ae = np.abs(rho_ret - rho_ref)

    if axis is None:
        mask = ~(
            ((wvl > 931) & (wvl < 945)) |
            ((wvl > 1100) & (wvl < 1160)) |
            ((wvl > 1300) & (wvl < 1500)) |
            ((wvl > 1750) & (wvl < 1980)) |
            (wvl > 2420)
        )
        ae_mean = np.nanmean(ae, axis=0, keepdims=True)
        return np.nanmean(ae_mean.flatten()[mask.flatten()])
    elif axis == 1:
        return np.nanmean(ae, axis=0)
    else:
        raise ValueError("Invalid axis value. Must be None or 1.")


def propagate_reflectance_uncertainty(Ltoa, Y_pred, Y_std, sza, n_wvl):
    """Propagate transfer-function uncertainty into reflectance uncertainty.

    Uses analytical delta-method propagation, assuming independence between
    the six transfer functions.

    Args:
        Ltoa: TOA radiance with shape ``(n_samples, n_wvl)``.
        Y_pred: Predicted transfer-function means with shape
            ``(n_samples, 6 * n_wvl)``.
        Y_std: Predicted standard deviations in the same layout as ``Y_pred``.
        sza: Solar zenith angles with shape ``(n_samples, 1)``.
        n_wvl: Number of wavelengths.

    Returns:
        Reflectance standard deviations with shape ``(n_samples, n_wvl)``.
    """
    L0    = Y_pred[:, 0:n_wvl]
    Edir  = Y_pred[:, n_wvl:2*n_wvl]
    Edif  = Y_pred[:, 2*n_wvl:3*n_wvl]
    Sa    = Y_pred[:, 3*n_wvl:4*n_wvl]
    Tdir  = Y_pred[:, 4*n_wvl:5*n_wvl]
    Tdif  = Y_pred[:, 5*n_wvl:6*n_wvl]

    s_L0   = Y_std[:, 0:n_wvl]
    s_Edir = Y_std[:, n_wvl:2*n_wvl]
    s_Edif = Y_std[:, 2*n_wvl:3*n_wvl]
    s_Sa   = Y_std[:, 3*n_wvl:4*n_wvl]
    s_Tdir = Y_std[:, 4*n_wvl:5*n_wvl]
    s_Tdif = Y_std[:, 5*n_wvl:6*n_wvl]

    cos_sza = np.cos(np.radians(sza))  # (n_samples, 1), broadcasts against (n_samples, n_wvl)

    E_total = Edir * cos_sza + Edif
    T_total = Tdir + Tdif
    N = np.pi * (Ltoa - L0)
    D = E_total * T_total + N * Sa
    rho = N / D

    d_L0   = -np.pi * E_total * T_total / D**2
    d_Sa   = -rho**2
    d_Edir = -rho * T_total * cos_sza / D
    d_Edif = -rho * T_total / D
    d_Tdir = -rho * E_total / D
    d_Tdif = -rho * E_total / D

    var_rho = (
        (d_L0   * s_L0)**2 +
        (d_Edir * s_Edir)**2 +
        (d_Edif * s_Edif)**2 +
        (d_Sa   * s_Sa)**2 +
        (d_Tdir * s_Tdir)**2 +
        (d_Tdif * s_Tdif)**2
    )

    return np.sqrt(var_rho)


def build_mask(wavelengths):
    """Build the wavelength inclusion mask used by evaluation metrics.

    Args:
        wavelengths: Array of wavelength values.

    Returns:
        Boolean mask where ``True`` marks wavelengths included in evaluation.
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
    """Calculate mean relative error using the function's axis-specific aggregation.

    Args:
        y_true: Reference values.
        y_pred: Predicted values matching ``y_true``.
        wavelengths: Wavelength array used to construct the evaluation mask.
        axis: Aggregation mode: ``None`` for a global scalar, ``2`` per function,
            ``1`` per wavelength, or ``0`` per function and wavelength. The
            wavelength exclusion mask is applied for ``None`` and ``2`` only;
            the ``1`` and ``0`` branches include all wavelengths.
        epsilon: Small constant added to the absolute reference values.

    Returns:
        MRE as a scalar or array according to ``axis``.
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
    """Calculate mean absolute error using the function's axis-specific aggregation.

    Args:
        y_true: Reference values.
        y_pred: Predicted values matching ``y_true``.
        wavelengths: Wavelength array used to construct the evaluation mask.
        axis: Aggregation mode: ``None`` for a global scalar, ``2`` per function,
            ``1`` per wavelength, or ``0`` per function and wavelength. The
            wavelength exclusion mask is applied for ``None`` and ``2`` only;
            the ``1`` and ``0`` branches include all wavelengths.

    Returns:
        MAE as a scalar or array according to ``axis``.
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
    """Calculate the percentage of targets inside prediction uncertainty bands.

    Args:
        y_true: Reference values.
        y_pred: Predicted means matching ``y_true``.
        y_std: Predicted standard deviations matching ``y_true``.
        n_std: Number of standard deviations defining each interval bound.

    Returns:
        Global coverage percentage and coverage percentage per function.
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


