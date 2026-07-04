from pathlib import Path
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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


def train_val_test_split(X, Y, wavelengths, verbose=True):
    """
    Perform train, validation and test splits on given data.
    - X: inputs of shape (n_samples, n_inputs)
    - Y: outputs of shape (n_samples, n_functions, n_wavelengths)
    - wavelengths: numpy array of wavelengths
    - verbose: print additional info or not
    - returns: training, validation and test inputs and outputs
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


def apply_pca(y_tr, y_val, n_components=10):
    """
    Apply PCA to each function separately, retaining n_components components.
    - y_tr: training outputs of shape (n_samples, n_functions, n_wavelengths)
    - y_val: validation outputs of shape (n_samples, n_functions, n_wavelengths)
    - n_components: number of PCA components to retain
    - returns: list of PCA objects, list of transformed training outputs, list of transformed validation outputs
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

    scaler = StandardScaler() if scale_type == "standard" else MinMaxScaler()
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
        scaler = StandardScaler() if scale_type == "standard" else MinMaxScaler()
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


# ============================= PLOTTING UTILITIES =============================
def show_fit_val_summary(results_df, save_path="nn_saves/nn_results_analysis.png"):
    # ── average fit time per dataset size ────────────────────────────────────
    avg_fit_time = (
        results_df.groupby("dataset_size")["fit_time"]
        .mean()
        .rename("avg_fit_time")
        .reset_index()
    )
    print("=" * 60)
    print("Average Fit Time by Dataset Size")
    print("=" * 60)
    print(avg_fit_time.to_string(index=False))


    # ── average best_val_mre per dataset size AND model ───────────────────────
    avg_val_mre = (
        results_df.groupby(["dataset_size", "model"])["best_val_mre"]
        .mean()
        .rename("avg_best_val_mre")
        .reset_index()
    )
    print("\n" + "=" * 60)
    print("Average Best Val MRE by Dataset Size & Model")
    print("=" * 60)
    print(avg_val_mre.to_string(index=False))


    # ── average best_val_mae per dataset size AND model ───────────────────────
    avg_val_mae = (
        results_df.groupby(["dataset_size", "model"])["best_val_mae"]
        .mean()
        .rename("avg_best_val_mae")
        .reset_index()
    )
    print("\n" + "=" * 60)
    print("Average Best Val MAE by Dataset Size & Model")
    print("=" * 60)
    print(avg_val_mae.to_string(index=False))


    # ── average best_val_mre and best_val_mae per model (across all dataset sizes) ──
    avg_metrics_by_model = (
        results_df.groupby("model")[["best_val_mre", "best_val_mae"]]
        .mean()
        .rename(columns={
            "best_val_mre": "avg_best_val_mre",
            "best_val_mae": "avg_best_val_mae"
        })
        .reset_index()
    )

    print("\n" + "=" * 60)
    print("Average Best Val MRE & MAE by Model (Across Dataset Sizes)")
    print("=" * 60)
    print(avg_metrics_by_model.to_string(index=False))


    # ── average best_val_mre and best_val_mae per dataset size (across models) ──
    avg_metrics_by_dataset_size = (
        results_df.groupby("dataset_size")[["best_val_mre", "best_val_mae"]]
        .mean()
        .rename(columns={
            "best_val_mre": "avg_best_val_mre",
            "best_val_mae": "avg_best_val_mae"
        })
        .reset_index()
    )

    print("\n" + "=" * 60)
    print("Average Best Val MRE & MAE by Dataset Size (Across Models)")
    print("=" * 60)
    print(avg_metrics_by_dataset_size.to_string(index=False))


    # ── visualisation ─────────────────────────────────────────────────────────
    models       = sorted(results_df["model"].unique())
    dataset_sizes = sorted(results_df["dataset_size"].unique())

    # pivot tables for grouped bars
    mre_pivot = avg_val_mre.pivot(index="dataset_size", columns="model", values="avg_best_val_mre")
    mae_pivot = avg_val_mae.pivot(index="dataset_size", columns="model", values="avg_best_val_mae")
    fit_pivot = avg_fit_time.set_index("dataset_size")[["avg_fit_time"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Neural Network Training Results — Summary", fontsize=14, y=1.02)

    x      = np.arange(len(dataset_sizes))
    n_mdl  = len(models)
    width  = 0.18
    offsets = np.linspace(-(n_mdl - 1) / 2, (n_mdl - 1) / 2, n_mdl) * width


    def _grouped_bars(ax, pivot, ylabel, title):
        for i, mdl in enumerate(models):
            vals = pivot[mdl].values if mdl in pivot.columns else np.zeros(len(dataset_sizes))
            bars = ax.bar(x + offsets[i], vals, width, label=mdl,
                        alpha=0.88, linewidth=0)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.015,
                        f"{h:.4f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in dataset_sizes])
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7.5, loc="upper right")


    # panel 1 — pal MRE
    _grouped_bars(axes[0], mre_pivot, "Avg Best Val MRE", "Validation MRE by Model & Dataset Size")

    # panel 2 — pal MAE
    _grouped_bars(axes[1], mae_pivot, "Avg Best Val MAE", "Validation MAE by Model & Dataset Size")

    # panel 3 — fit time (single series, no model split)
    fit_vals = [fit_pivot.loc[s, "avg_fit_time"] for s in dataset_sizes]
    bars = axes[2].bar(x, fit_vals, width=0.45, alpha=0.88, linewidth=0)
    for bar in bars:
        h = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2, h * 1.015,
                    f"{h:.1f}s", ha="center", va="bottom", fontsize=8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([str(s) for s in dataset_sizes])
    axes[2].set_xlabel("Dataset Size")
    axes[2].set_ylabel("Avg Fit Time (s)")
    axes[2].set_title("Average Fit Time by Dataset Size", fontweight="bold", pad=10)
    axes[2].yaxis.grid(True)
    axes[2].set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved to {save_path}")


def show_top_results(results_df, top_n=5):
    """
    Show the top N results based on best_val_mre
    - inputs: results dataframe, number of top results to show
    - outputs: prints the top N results
    """
    print(f"\nTop {top_n} Results by Best Validation MRE:")
    print("=" * 60)
    top_results = results_df.nsmallest(top_n, "best_val_mre")
    print(top_results.to_string(index=False))


def show_barplot_results(results_df, save_path="nn_saves/nn_results_analysis.png"):
    """
    Show a bar plot of best_val_mre for each parameter combination, sorted in ascending order.
    - inputs: results dataframe, path to save the plot
    - outputs: displays the bar plot and saves it to the specified path
    """
    # sort by best_val_mre ascending
    results_sorted = results_df.sort_values("best_val_mre", ascending=True)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(results_sorted)), results_sorted["best_val_mre"], color="skyblue")
    plt.xlabel("Parameter Combination ID")
    plt.ylabel("Val MRE")
    plt.title("Val MRE for Each Parameter Combination (Sorted)")
    plt.xticks(range(len(results_sorted)), results_sorted["experiment_id"], rotation=90)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nBar plot saved to {save_path}")


def show_test_results_mre(y_test, y_pred, wavelengths):
    """
    Show the test results, including MRE and MRE per function and per wavelength.
    - inputs: true test outputs, predicted test outputs, wavelengths
    - outputs: prints MRE scores and displays plots of MRE per wavelength and per function
    """
    mre = mre_score(y_test, y_pred, wavelengths)
    print("Testing MRE:", mre)

    mre_per_func = mre_score(y_test, y_pred, wavelengths, axis=2)
    for i in range(globals.N_FUNCTIONS):
        print(f"{globals.function_names_plots[i]} MRE: {mre_per_func[i]:.4f}")

    mre_per_wvl = mre_score(y_test, y_pred, wavelengths, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    axes[0].plot(wavelengths, mre_per_wvl)
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("MRE")
    axes[0].set_title("MRE per Wavelength")
    axes[0].grid()
    axes[1].plot(wavelengths, mre_per_wvl)
    axes[1].set_ylim(0, 0.6)
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("MRE")
    axes[1].set_title("MRE per Wavelength (Zoomed)")
    axes[1].grid()
    plt.tight_layout()
    plt.show()

    # MRE per wavelength again but in log scale to better visualize small values
    mre_per_wvl_log = np.log10(mre_per_wvl + 1e-10)  # add small value to avoid log(0)
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, mre_per_wvl_log)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Log10(MRE)")
    plt.title("Log10(MRE) per Wavelength")
    plt.grid()
    plt.tight_layout()
    plt.show()

    mre_per_func_wvl = mre_score(y_test, y_pred, wavelengths, axis=0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i in range(globals.N_FUNCTIONS):
        axes[i].plot(wavelengths, mre_per_func_wvl[i])
        axes[i].set_ylim(0, 1.25)
        axes[i].set_xlabel("Wavelength (nm)")
        axes[i].set_ylabel("MRE")
        axes[i].set_title(f"MRE for {globals.function_names_plots[i]} per wavelength")
        axes[i].grid()
    plt.tight_layout()
    plt.show()

    # MRE per function again but in log scale to better visualize small values
    mre_per_func_wvl_log = np.log10(mre_per_func_wvl + 1e-10)  # add small value to avoid log(0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i in range(globals.N_FUNCTIONS):
        axes[i].plot(wavelengths, mre_per_func_wvl_log[i])
        axes[i].set_xlabel("Wavelength (nm)")
        axes[i].set_ylabel("Log10(MRE)")
        axes[i].set_title(f"Log10(MRE) for {globals.function_names_plots[i]} per wavelength")
        axes[i].grid()
    plt.tight_layout()
    plt.show()


def show_test_results_mae(y_test, y_pred, wavelengths):
    mae = mae_score(y_test, y_pred, wavelengths)
    print("Testing MAE:", mae)

    mae_per_func = mae_score(y_test, y_pred, wavelengths, axis=2)
    for i in range(globals.N_FUNCTIONS):
        print(f"{globals.function_names_plots[i]} MAE: {mae_per_func[i]:.4f}")

    mae_per_wvl = mae_score(y_test, y_pred, wavelengths, axis=1)
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, mae_per_wvl)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("MAE")
    plt.title("MAE per wavelength")
    plt.grid()
    plt.show()

    mae_per_func_wvl = mae_score(y_test, y_pred, wavelengths, axis=0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i in range(globals.N_FUNCTIONS):
        axes[i].plot(wavelengths, mae_per_func_wvl[i])
        axes[i].set_xlabel("Wavelength (nm)")
        axes[i].set_ylabel("MAE")
        axes[i].set_title(f"MAE for {globals.function_names_plots[i]} per wavelength")
        axes[i].grid()
    plt.tight_layout()
    plt.show()