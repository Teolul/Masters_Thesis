"""
Python conversion of script_validation.m

Challenge evaluation script.

For every participant "model" and every scenario/track combination, this
script:
  1. Loads the model's emulated atmospheric transfer functions (L0, E, Sa, T)
     from its submitted .h5 file.
  2. Uses those transfer functions to either
       - retrieve surface reflectance from reference TOA radiance (scenario A), or
       - simulate TOA radiance from a reference reflectance spectrum (scenario B).
  3. Compares the result against the true reference data with a mean relative
     error (MRE) metric.
  4. Ranks all models per scenario/track, then combines the ranks into a
     single weighted final ranking.

NOTE on paths: the original MATLAB script used hard-coded Windows paths
("D:\\...\\"). Update the `BASE_PATH` variable below to point at your local
results/scenario folders before running.

NOTE on HDF5 axis order: MATLAB's h5read and Python's h5py can disagree on
the axis order of multi-dimensional datasets written by MATLAB, because
HDF5 stores arrays in row-major (C) order while MATLAB is column-major.
If shapes/indices below look "transposed" compared to what you expect from
MATLAB, try `.T` on the array right after reading it.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import h5py


# --------------------------------------------------------------------------
# Path configuration -- adapt for your machine
# --------------------------------------------------------------------------
BASE_PATH = r"C:\Users\Matteo\Desktop\Scuola\MastersThesis\AMLEC-1\rtm_emulation\results" + "\\"


def read_lutdata(file_path, dataset="/LUTdata"):
    """Read a dataset from an HDF5 file (equivalent to MATLAB's h5read)."""
    with h5py.File(file_path, "r") as f:
        return np.asarray(f[dataset][()], dtype=float)


# --------------------------------------------------------------------------
# Support functions (equivalent to the MATLAB local functions)
# --------------------------------------------------------------------------
def retrieve_reflectance(Ltoa, Y, sza, n):
    """
    Perform atmospheric correction: retrieve surface reflectance from
    TOA radiance `Ltoa` given the emulated transfer functions `Y`.

    Y is stacked as 6 blocks of n rows each: [L0; E_direct; E_diffuse; Sa; T1; T2]
    """
    L0 = Y[0:n, :]
    E = Y[n:2 * n, :] * np.cos(np.deg2rad(sza)) + Y[2 * n:3 * n, :]
    Sa = Y[3 * n:4 * n, :]
    T = Y[4 * n:5 * n, :] + Y[5 * n:6 * n, :]

    rho = np.pi * (Ltoa - L0) / (E * T + np.pi * (Ltoa - L0) * Sa)
    return rho


def toa_radiance(rho, Y, sza, n):
    """
    Forward-simulate TOA radiance from a known surface reflectance `rho`
    given the emulated transfer functions `Y`.
    """
    L0 = Y[0:n, :]
    E = Y[n:2 * n, :] * np.cos(np.deg2rad(sza)) + Y[2 * n:3 * n, :]
    Sa = Y[3 * n:4 * n, :]
    T = Y[4 * n:5 * n, :] + Y[5 * n:6 * n, :]

    Ltoa = L0 + (1 / np.pi) * E * T * rho / (1 - Sa * rho)
    return Ltoa


def error_metric_A(rho_ref, rho_ret, wvl):
    """Mean relative error (%) in retrieved surface reflectance, scenario A."""
    re = 100 * np.abs(rho_ret - rho_ref) / rho_ref
    re = np.nanmean(re, axis=1)  # average over samples

    # Spectral average, skipping strong atmospheric absorption bands
    idx = ~(
        ((wvl > 931) & (wvl < 945))
        | ((wvl > 1100) & (wvl < 1160))
        | ((wvl > 1300) & (wvl < 1500))
        | ((wvl > 1750) & (wvl < 1980))
        | (wvl > 2420)
    )
    return np.nanmean(re[idx])


def error_metric_B(Ltoa_ref, Ltoa_ret):
    """Mean relative error (%) in simulated TOA radiance, scenario B."""
    re = 100 * np.abs(Ltoa_ref - Ltoa_ret) / Ltoa_ref
    re = np.nanmean(re, axis=1)  # average over samples
    return np.nanmean(re)  # spectral average


def compute_final_ranks(rnk):
    """
    Compute the weighted average ranking and the standard competition
    ranking ("1224" style: ties share the lowest rank, next rank skips
    accordingly) for a given ranking matrix.

    Parameters
    ----------
    rnk : (n_models, 4) array
        Per-scenario/track individual ranks. Columns are assumed to be
        [AC-Interp, AC-Extra, CO2-Interp, CO2-Extra].

    Returns
    -------
    rnk_avg : (n_models,) array
        Weighted average rank for each model.
    final_ranks : (n_models,) int array
        Standard competition rank based on rnk_avg.
    """
    n = rnk.shape[0]

    # Interpolation weighted 0.65, extrapolation 0.35; two scenarios contribute equally
    weights = np.array([0.325, 0.175, 0.325, 0.175])  # [AC-I, AC-E, CO2-I, CO2-E]

    rnk_avg = rnk @ weights

    idx_sorted = np.argsort(rnk_avg, kind="stable")
    sorted_scores = rnk_avg[idx_sorted]
    final_ranks = np.zeros(n, dtype=int)

    i = 0
    while i < n:
        tie_value = sorted_scores[i]
        tied = np.where(np.abs(sorted_scores - tie_value) < 1e-8)[0]
        tied = tied[tied >= i]
        k = tied.size

        final_ranks[idx_sorted[tied]] = i + 1  # 1-based rank, as in MATLAB
        i += k

    return rnk_avg, final_ranks


def rank_columnwise(values):
    """
    Equivalent of MATLAB's `[~, idx] = sort(x); [~, rnk] = sort(idx);` trick,
    which converts values into 1-based ranks (NaNs are sorted last by
    default in both MATLAB and NumPy).
    """
    order = np.argsort(values, kind="stable")
    ranks = np.argsort(order, kind="stable") + 1
    return ranks


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------
def main():
    # Find how many .h5 submission files we have, and how many distinct models
    h5_files = glob.glob(os.path.join(BASE_PATH, "*.h5"))
    file_names = [os.path.basename(f) for f in h5_files]

    pattern = r"^(.*)_[AB][1-3]\.h5$"
    model_names = sorted(
        {
            m.group(1)
            for name in file_names
            if (m := re.match(pattern, name)) is not None
        }
    )
    n = len(model_names)

    # Scenario-track configuration
    S = ["A", "B"]
    tracks = [1, 2]
    track_names = ["refInterp", "refExtrap", "refReal"]
    m = len(S) * len(tracks)  # number of scenario-track combinations

    mre = np.full((n, m), np.nan)
    rnk = np.full((n, m), np.nan)

    # ----------------------------------------------------------------
    # Evaluate error metrics
    # ----------------------------------------------------------------
    for s_idx, s in enumerate(S):
        for t_idx, t in enumerate(tracks):
            j = t_idx + s_idx * len(tracks)  # scenario-track index
            print(f"Scenario {s} - track: {track_names[t_idx]}")

            # Reference data (TOA radiance, header/geometry, wavelengths)
            ref_file = BASE_PATH.replace(
                "results\\", f"scenario{s}\\reference\\{track_names[t_idx]}.h5"
            )
            L = read_lutdata(ref_file, "/LUTdata")
            sza_full = read_lutdata(ref_file, "/LUTheader")
            # NOTE: row indices below assume the same header row layout/orientation
            # as the original MATLAB code. Check axis order (see module docstring).
            sza = sza_full[6, :] if s_idx == 0 else sza_full[4, :]
            wvl = read_lutdata(ref_file, "/wvl").flatten()
            n_wvl = wvl.size

            # Reference surface reflectance (note: original script always reads
            # this from "scenarioA", regardless of the current scenario `s` --
            # preserved here as-is; double check whether this is intentional)
            refldb_file = BASE_PATH.replace("results\\", "scenarioA\\reference\\refldb.txt")
            rho_ref_raw = np.loadtxt(refldb_file, delimiter=",")
            spline = CubicSpline(rho_ref_raw[:, 0], rho_ref_raw[:, 1])
            rho_ref = spline(wvl)

            # Evaluate each model on the current scenario-track
            for i, model in enumerate(model_names):
                file_name = f"{model}_{s}{t}.h5"
                file_path = os.path.join(BASE_PATH, file_name)
                print(f"Model {i + 1} ({file_name})")

                if os.path.exists(file_path):
                    yq = read_lutdata(file_path, "/LUTdata")

                    if s_idx == 0:  # scenario A
                        rho_ret = retrieve_reflectance(L, yq, sza, n_wvl)
                        mre[i, j] = error_metric_A(rho_ref, rho_ret, wvl)
                    else:  # scenario B
                        Ltoa_ret = toa_radiance(rho_ref, yq, sza, n_wvl)
                        mre[i, j] = error_metric_B(L, Ltoa_ret)

            # Individual ranking for this scenario-track (NaNs -> worst rank)
            rnk[:, j] = rank_columnwise(mre[:, j])
            rnk[np.isnan(mre[:, j]), j] = n

    # ----------------------------------------------------------------
    # Combine into final ranking
    # ----------------------------------------------------------------
    rnk_avg, final_score = compute_final_ranks(rnk)

    results = pd.DataFrame(
        {
            "model": model_names,
            "final_rank": final_score,
            "avg_rank": rnk_avg,
        }
    )
    for j in range(m):
        results[f"mre_{j + 1}"] = mre[:, j]

    print(results)
    return results


if __name__ == "__main__":
    main()