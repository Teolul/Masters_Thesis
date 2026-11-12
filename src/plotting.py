"""Plotting helpers for model results and diagnostic comparisons."""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import globals
from metrics import (
    mae_amlec,
    mae_score,
    mre_amlec,
    mre_score,
)

def show_fit_val_summary(results_df, title="Results Summary", save_path="nn_saves/nn_results_analysis.png"):
    """Summarize validation metrics and fit times with grouped plots.

    Args:
        results_df: Results dataframe containing validation metrics and fit times.
        title: Figure title.
        save_path: Destination path for the saved figure.

    Returns:
        None. Prints summary tables and displays and saves the figure.
    """
    if results_df["experiment_id"].str.startswith("DotProduct_", na=False).any():
        plot_results_df = results_df[
            ~results_df["experiment_id"].str.startswith(("DotProduct_", "Matern_iso"), na=False)
        ]
    else:
        plot_results_df = results_df.copy()

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
        plot_results_df.groupby(["dataset_size", "model"])["best_val_mre"]
        .mean()
        .rename("avg_val_mre")
        .reset_index()
    )
    print("\n" + "=" * 60)
    print("Average Val MRE by Dataset Size & Model")
    print("=" * 60)
    print(avg_val_mre.to_string(index=False))


    # ── average best_val_mae per dataset size AND model ───────────────────────
    avg_val_mae = (
        plot_results_df.groupby(["dataset_size", "model"])["best_val_mae"]
        .mean()
        .rename("avg_val_mae")
        .reset_index()
    )
    print("\n" + "=" * 60)
    print("Average Val MAE by Dataset Size & Model")
    print("=" * 60)
    print(avg_val_mae.to_string(index=False))


    # ── average best_val_mre and best_val_mae per model (across all dataset sizes) ──
    avg_metrics_by_model = (
        plot_results_df.groupby("model")[["best_val_mre", "best_val_mae"]]
        .mean()
        .rename(columns={
            "best_val_mre": "avg_val_mre",
            "best_val_mae": "avg_val_mae"
        })
        .reset_index()
    )

    print("\n" + "=" * 60)
    print("Average Val MRE & MAE by Model (Across Dataset Sizes)")
    print("=" * 60)
    print(avg_metrics_by_model.to_string(index=False))


    # ── average best_val_mre and best_val_mae per dataset size (across models) ──
    avg_metrics_by_dataset_size = (
        plot_results_df.groupby("dataset_size")[["best_val_mre", "best_val_mae"]]
        .mean()
        .rename(columns={
            "best_val_mre": "avg_val_mre",
            "best_val_mae": "avg_val_mae"
        })
        .reset_index()
    )

    print("\n" + "=" * 60)
    print("Average Val MRE & MAE by Dataset Size (Across Models)")
    print("=" * 60)
    print(avg_metrics_by_dataset_size.to_string(index=False))


    # ── visualisation ─────────────────────────────────────────────────────────
    models       = sorted(plot_results_df["model"].unique())
    dataset_sizes = sorted(plot_results_df["dataset_size"].unique())

    # pivot tables for grouped bars
    mre_pivot = avg_val_mre.pivot(index="dataset_size", columns="model", values="avg_val_mre")
    mae_pivot = avg_val_mae.pivot(index="dataset_size", columns="model", values="avg_val_mae")
    fit_pivot = avg_fit_time.set_index("dataset_size")[["avg_fit_time"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    fig.suptitle(title, fontsize=14, y=1.02)

    x      = np.arange(len(dataset_sizes))
    n_mdl  = len(models)
    width  = 0.15
    offsets = np.linspace(-(n_mdl - 1) / 2, (n_mdl - 1) / 2, n_mdl) * width


    def _grouped_bars(ax, pivot, ylabel, title, factor=100):
        for i, mdl in enumerate(models):
            vals = pivot[mdl].values if mdl in pivot.columns else np.zeros(len(dataset_sizes))
            bars = ax.bar(x + offsets[i], vals, width, label=mdl,
                        alpha=0.88, linewidth=0)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.015,
                        f"{h*factor:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in dataset_sizes])
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8.5, loc="upper right")


    # panel 1 — pal MRE
    _grouped_bars(axes[0], mre_pivot, "Avg Val MRE (%)", "Validation MRE by Model & Dataset Size")

    # panel 2 — pal MAE
    _grouped_bars(axes[1], mae_pivot, "Avg Val MAE", "Validation MAE by Model & Dataset Size", factor=1)

    # panel 3 — fit time (single series, no model split)
    fit_vals = [fit_pivot.loc[s, "avg_fit_time"] for s in dataset_sizes]
    bars = axes[2].bar(x, fit_vals, width=0.45, alpha=0.88, linewidth=0)
    for bar in bars:
        h = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2, h * 1.015,
                    f"{h:.1f}s", ha="center", va="bottom", fontsize=8.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([str(s) for s in dataset_sizes])
    axes[2].set_xlabel("Dataset Size")
    axes[2].set_ylabel("Avg Fit Time (s)")
    axes[2].set_title("Average Fit Time by Dataset Size", fontweight="bold", pad=10)
    axes[2].yaxis.grid(True)
    axes[2].set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved to {save_path}")


def show_test_summary(results_df, title="Results Summary", save_path="nn_saves/nn_results_analysis.png"):
    """Summarize test metrics and inference times with grouped plots.

    Args:
        results_df: Results dataframe containing test metrics and inference times.
        title: Figure title.
        save_path: Destination path for the saved figure.

    Returns:
        None. Prints summary tables and displays and saves the figure.
    """
    if results_df["experiment_id"].str.startswith("DotProduct_", na=False).any():
        plot_results_df = results_df[
            ~results_df["experiment_id"].str.startswith(("DotProduct_", "Matern_iso", "GP"), na=False)
        ]
    else:
        plot_results_df = results_df.copy()

    # ── average inference time per dataset size ──────────────────────────────
    if "gp_kernel_full_name" in results_df.columns:
        results_df_time = results_df[~results_df["experiment_id"].str.startswith(("combined", "GP"), na=False)]
    else:
        results_df_time = results_df.copy()

    avg_inference_time = (
        results_df_time.groupby("dataset_size")["inference_time"]
        .mean()
        .rename("avg_inference_time")
        .reset_index()
    )
    print("=" * 60)
    print("Average Inference Time by Dataset Size")
    print("=" * 60)
    print(avg_inference_time.to_string(index=False))

    # ── average best_val_mre per dataset size AND model ───────────────────────
    avg_test_mre = (
        plot_results_df.groupby(["dataset_size", "model"])["test_mre"]
        .mean()
        .rename("avg_test_mre")
        .reset_index()
    )
    print("\n" + "=" * 60)
    print("Average Test MRE by Dataset Size & Model")
    print("=" * 60)
    print(avg_test_mre.to_string(index=False))


    # ── average best_val_mae per dataset size AND model ───────────────────────
    avg_test_mae = (
        plot_results_df.groupby(["dataset_size", "model"])["test_mae"]
        .mean()
        .rename("avg_test_mae")
        .reset_index()
    )
    print("\n" + "=" * 60)
    print("Average Test MAE by Dataset Size & Model")
    print("=" * 60)
    print(avg_test_mae.to_string(index=False))


    # ── average best_val_mre and best_val_mae per model (across all dataset sizes) ──
    avg_metrics_by_model = (
        plot_results_df.groupby("model")[["test_mre", "test_mae"]]
        .mean()
        .rename(columns={
            "test_mre": "avg_test_mre",
            "test_mae": "avg_test_mae"
        })
        .reset_index()
    )

    print("\n" + "=" * 60)
    print("Average Test MRE & MAE by Model (Across Dataset Sizes)")
    print("=" * 60)
    print(avg_metrics_by_model.to_string(index=False))


    # ── average test_mre and test_mae per dataset size (across models) ──
    avg_metrics_by_dataset_size = (
        plot_results_df.groupby("dataset_size")[["test_mre", "test_mae"]]
        .mean()
        .rename(columns={
            "test_mre": "avg_test_mre",
            "test_mae": "avg_test_mae"
        })
        .reset_index()
    )

    print("\n" + "=" * 60)
    print("Average Test MRE & MAE by Dataset Size (Across Models)")
    print("=" * 60)
    print(avg_metrics_by_dataset_size.to_string(index=False))


    # ── visualisation ─────────────────────────────────────────────────────────
    models       = sorted(plot_results_df["model"].unique())
    dataset_sizes = sorted(plot_results_df["dataset_size"].unique())

    # pivot tables for grouped bars
    mre_pivot = avg_test_mre.pivot(index="dataset_size", columns="model", values="avg_test_mre")
    mae_pivot = avg_test_mae.pivot(index="dataset_size", columns="model", values="avg_test_mae")
    inf_pivot = avg_inference_time.set_index("dataset_size")[["avg_inference_time"]]

    fig, axes = plt.subplots(1, 3 if "amlec" not in save_path else 2, figsize=(18, 9))
    fig.suptitle(title, fontsize=14, y=1.02)

    x      = np.arange(len(dataset_sizes))
    n_mdl  = len(models)
    width  = 0.12
    offsets = np.linspace(-(n_mdl - 1) / 2, (n_mdl - 1) / 2, n_mdl) * width


    def _grouped_bars(ax, pivot, ylabel, title, factor=100, decimals=2):
        for i, mdl in enumerate(models):
            vals = pivot[mdl].values if mdl in pivot.columns else np.zeros(len(dataset_sizes))
            bars = ax.bar(x + offsets[i], vals, width, label=mdl,
                        alpha=0.88, linewidth=0)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.015,
                        f"{h*factor:.{decimals}f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in dataset_sizes])
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*factor:.{decimals}f}"))
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7.5, loc="upper right")


    # panel 1 — pal MRE
    _grouped_bars(axes[0], mre_pivot, "Avg Test MRE (%)", "Test MRE by Model & Dataset Size")

    # panel 2 — pal MAE
    _grouped_bars(axes[1], mae_pivot, "Avg Test MAE", "Test MAE by Model & Dataset Size", factor=1, decimals=4 if "amlec" in save_path else 2)

    if "amlec" not in save_path:
        # panel 3 — inference time (single series, no model split)
        inf_vals = [inf_pivot.loc[s, "avg_inference_time"] for s in dataset_sizes]
        bars = axes[2].bar(x, inf_vals, width=0.45, alpha=0.88, linewidth=0)
        for bar in bars:
            h = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width() / 2, h * 1.015,
                        f"{h:.3f}s", ha="center", va="bottom", fontsize=8.5)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([str(s) for s in dataset_sizes])
        axes[2].set_xlabel("Dataset Size")
        axes[2].set_ylabel("Avg Inference Time (s)")
        axes[2].set_title("Average Inference Time by Dataset Size", fontweight="bold", pad=10)
        axes[2].yaxis.grid(True)
        axes[2].set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved to {save_path}")


def show_top_results(results_df, top_n=5):
    """Print the rows with the lowest available validation or test MRE.

    Args:
        results_df: Results dataframe containing a validation or test MRE column.
        top_n: Number of rows to print.

    Returns:
        None. Prints the selected rows.
    """
    print(f"\nTop {top_n} Results by Best MRE:")
    print("=" * 60)
    
    mre_to_consider = ""
    if "best_val_mre" not in results_df.columns:
        mre_to_consider = "test_mre"
    elif "test_mre" not in results_df.columns:
        mre_to_consider = "best_val_mre"

    top_results = results_df.nsmallest(top_n, mre_to_consider)
    print(top_results.to_string(index=False))


def show_barplot_results(results_df, title="Bar Plot of Results", save_path="nn_saves/nn_results_analysis.png"):
    """Plot the available validation or test MRE values by experiment.

    Args:
        results_df: Results dataframe containing experiment IDs and MRE values.
        title: Figure title.
        save_path: Destination path for the saved figure.

    Returns:
        None. Displays and saves the figure.
    """
    # sort by best_val_mre ascending
    if results_df["experiment_id"].str.startswith("DotProduct_", na=False).any():
        plot_results_df = results_df[
            ~results_df["experiment_id"].str.startswith(("DotProduct_", "Matern_iso"), na=False)
        ]
    else:
        plot_results_df = results_df.copy()

    mre_to_consider = ""
    if "best_val_mre" not in results_df.columns:
        mre_to_consider = "test_mre"
    elif "test_mre" not in results_df.columns:
        mre_to_consider = "best_val_mre"

    results_sorted = plot_results_df.sort_values(mre_to_consider, ascending=True)

    plt.figure(figsize=(12, 6), dpi=300)
    plt.bar(range(len(results_sorted)), results_sorted[mre_to_consider], color="skyblue")
    plt.xlabel("Parameter Combination ID")
    plt.ylabel("MRE (%)")
    plt.title(title)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.{2}f}"))
    plt.xticks(range(len(results_sorted)), results_sorted["experiment_id"], rotation=90)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nBar plot saved to {save_path}")


def show_test_results_mre(y_test, y_pred, wavelengths, exp_id="EXP_ID", save_path="nn_saves/testing_results/"):
    """Report test MRE and plot spectral error summaries.

    Args:
        y_test: Reference test outputs.
        y_pred: Predicted test outputs.
        wavelengths: Wavelength array corresponding to the final output axis.
        exp_id: Experiment identifier used in plot titles and filenames.
        save_path: Directory prefix used when saving the plots.

    Returns:
        Overall MRE and per-function MRE, or ``None`` for per-function MRE
        when the AMLEC path is selected.
    """
    if "amlec" in save_path:
        mre = mre_amlec(y_test, y_pred, wavelengths)
    else:
        mre = mre_score(y_test, y_pred, wavelengths)
    print("Testing MRE:", mre)

    if "amlec" not in save_path:
        mre_per_func = mre_score(y_test, y_pred, wavelengths, axis=2)
        for i in range(globals.N_FUNCTIONS):
            print(f"{globals.function_names_plots[i]} MRE: {mre_per_func[i]:.4f}")
    else:
        mre_per_func = None

    # ---MRE PER WAVELENGTH AND ZOOMED---
    if "amlec" in save_path:
        mre_per_wvl = mre_amlec(y_test, y_pred, wavelengths, axis=1)
    else:
        mre_per_wvl = mre_score(y_test, y_pred, wavelengths, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    plt.suptitle(exp_id, fontsize=16, y=1.02)
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
    plt.savefig(save_path + f"{exp_id}_mre_wavelengths.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---MRE PER WAVELENGTH LOG---
    mre_per_wvl_log = np.log10(mre_per_wvl + 1e-10)  # add small value to avoid log(0)
    plt.figure(figsize=(10, 6))
    plt.suptitle(exp_id, fontsize=16, y=1.02)
    plt.plot(wavelengths, mre_per_wvl_log)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Log10(MRE)")
    plt.title("Log10(MRE) per Wavelength")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path + f"{exp_id}_mre_wavelengths_log.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---MRE PER WAVELENGTH LOG ZOOMED---
    mre_per_wvl_log = np.log10(mre_per_wvl + 1e-10)  # add small value to avoid log(0)
    plt.figure(figsize=(10, 6))
    plt.suptitle(exp_id, fontsize=16, y=1.02)
    plt.plot(wavelengths, mre_per_wvl_log)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Log10(MRE)")
    plt.ylim(-3.5, 1.5)
    plt.title("Log10(MRE) per Wavelength (Zoomed)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path + f"{exp_id}_mre_wavelengths_log_zoomed.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---MRE PER WAVELENGTH LOG ABSORPTIONS---
    mre_per_wvl_log = np.log10(mre_per_wvl + 1e-10)  # add small value to avoid log(0)
    plt.figure(figsize=(10, 6))
    plt.suptitle(exp_id, fontsize=16, y=1.02)
    plt.plot(wavelengths, mre_per_wvl_log)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Log10(MRE)")
    plt.xlim(600, 1250)
    plt.ylim(-3.5, 0)
    plt.title("Log10(MRE) per Wavelength (Absorptions)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path + f"{exp_id}_mre_wavelengths_log_absorptions.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---MRE PER FUNCTION PER WAVELENGTH ZOOMED---
    if "amlec" not in save_path:
        mre_per_func_wvl = mre_score(y_test, y_pred, wavelengths, axis=0)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.suptitle(exp_id, fontsize=16, y=1.02)
        axes = axes.flatten()
        for i in range(globals.N_FUNCTIONS):
            axes[i].plot(wavelengths, mre_per_func_wvl[i])
            axes[i].set_ylim(0, 1.25)
            axes[i].set_xlabel("Wavelength (nm)")
            axes[i].set_ylabel("MRE")
            axes[i].set_title(f"MRE for {globals.function_names_plots[i]} per wavelength")
            axes[i].grid()
        plt.tight_layout()
        plt.savefig(save_path + f"{exp_id}_mre_functions.png", dpi=300, bbox_inches="tight")
        plt.show()
    
        # ---MRE PER FUNCTION PER WAVELENGTH LOG---
        mre_per_func_wvl_log = np.log10(mre_per_func_wvl + 1e-10)  # add small value to avoid log(0)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.suptitle(exp_id, fontsize=16, y=1.02)
        axes = axes.flatten()
        for i in range(globals.N_FUNCTIONS):
            axes[i].plot(wavelengths, mre_per_func_wvl_log[i])
            axes[i].set_xlabel("Wavelength (nm)")
            axes[i].set_ylabel("Log10(MRE)")
            axes[i].set_ylim(-6, 6)
            axes[i].set_title(f"Log10(MRE) for {globals.function_names_plots[i]} per wavelength")
            axes[i].grid()
        plt.tight_layout()
        plt.savefig(save_path + f"{exp_id}_mre_functions_log.png", dpi=300, bbox_inches="tight")
        plt.show()
    
        # ---MRE PER FUNCTION PER WAVELENGTH LOG ZOOMED---
        mre_per_func_wvl_log = np.log10(mre_per_func_wvl + 1e-10)  # add small value to avoid log(0)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.suptitle(exp_id, fontsize=16, y=1.02)
        axes = axes.flatten()
        for i in range(globals.N_FUNCTIONS):
            axes[i].plot(wavelengths, mre_per_func_wvl_log[i])
            axes[i].set_xlabel("Wavelength (nm)")
            axes[i].set_ylabel("Log10(MRE)")
            axes[i].set_ylim(-5, 1.5)
            axes[i].set_title(f"Log10(MRE) for {globals.function_names_plots[i]} per wavelength (Zoomed)")
            axes[i].grid()
        plt.tight_layout()
        plt.savefig(save_path + f"{exp_id}_mre_functions_log_zoomed.png", dpi=300, bbox_inches="tight")
        plt.show()
    
    file_path = save_path + "mre_per_wvl_log.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            mre_dict = pickle.load(f)
    else:
        mre_dict = {}
    mre_dict[exp_id] = mre_per_wvl_log
    with open(file_path, "wb") as f:
        pickle.dump(mre_dict, f)

    return mre, mre_per_func


def show_test_results_mae(y_test, y_pred, wavelengths, exp_id="EXP_ID", save_path="nn_saves/testing_results/"):
    """Report test MAE and plot spectral error summaries.

    Args:
        y_test: Reference test outputs.
        y_pred: Predicted test outputs.
        wavelengths: Wavelength array corresponding to the final output axis.
        exp_id: Experiment identifier used in plot titles and filenames.
        save_path: Directory prefix used when saving the plots.

    Returns:
        Overall MAE and per-function MAE, or ``None`` for per-function MAE
        when the AMLEC path is selected.
    """
    """
    Show the test results, including MAE and MAE per function and per wavelength.
    - inputs: true test outputs, predicted test outputs, wavelengths
    - outputs: prints MAE scores and displays plots of MAE per wavelength and per function
    """
    if "amlec" in save_path:
        mae = mae_amlec(y_test, y_pred, wavelengths)
    else:
        mae = mae_score(y_test, y_pred, wavelengths)
    print("Testing MAE:", mae)

    if "amlec" not in save_path:
        mae_per_func = mae_score(y_test, y_pred, wavelengths, axis=2)
        for i in range(globals.N_FUNCTIONS):
            print(f"{globals.function_names_plots[i]} MAE: {mae_per_func[i]:.4f}")
    else:
        mae_per_func = None

    if "amlec" in save_path:
        mae_per_wvl = mae_amlec(y_test, y_pred, wavelengths, axis=1)
    else:
        mae_per_wvl = mae_score(y_test, y_pred, wavelengths, axis=1)
    plt.figure(figsize=(10, 5))
    plt.suptitle(exp_id, fontsize=16, y=1.02)
    plt.plot(wavelengths, mae_per_wvl)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("MAE")
    plt.title("MAE per wavelength")
    plt.grid()
    plt.savefig(save_path + f"{exp_id}_mae_wavelengths.png", dpi=300, bbox_inches="tight")
    plt.show()

    if "amlec" not in save_path:
        mae_per_func_wvl = mae_score(y_test, y_pred, wavelengths, axis=0)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.suptitle(exp_id, fontsize=16, y=1.02)
        axes = axes.flatten()
        for i in range(globals.N_FUNCTIONS):
            axes[i].plot(wavelengths, mae_per_func_wvl[i])
            axes[i].set_xlabel("Wavelength (nm)")
            axes[i].set_ylabel("MAE")
            axes[i].set_title(f"MAE for {globals.function_names_plots[i]} per wavelength")
            axes[i].grid()
        plt.tight_layout()
        plt.savefig(save_path + f"{exp_id}_mae_functions.png", dpi=300, bbox_inches="tight")
        plt.show()

    file_path = save_path + "mae_per_wvl.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            mae_dict = pickle.load(f)
    else:
        mae_dict = {}
    mae_dict[exp_id] = mae_per_wvl
    with open(file_path, "wb") as f:
        pickle.dump(mae_dict, f)

    return mae, mae_per_func


def show_predicted_vs_true(y_test, y_pred, y_std, wavelengths, exp_id="EXP_ID", save_path="nn_saves/testing_results/"):
    """Plot predicted and reference outputs with prediction uncertainty.

    Args:
        y_test: Reference test outputs.
        y_pred: Predicted test outputs.
        y_std: Predicted standard deviations corresponding to ``y_pred``.
        wavelengths: Wavelength array corresponding to the final output axis.
        exp_id: Experiment identifier used in plot titles and filenames.
        save_path: Directory prefix used when saving the plots.

    Returns:
        None. Displays and saves the plots.
    """
    """
    Show the mean predicted vs true functions across all samples, with optional uncertainty bands.
    - inputs: true test outputs, predicted test outputs, predicted standard deviation (optional), wavelengths
    - outputs: displays plots of mean predicted vs true functions for each function
    """
    # --- compute mean true and predicted functions across all samples ---
    Y_true_mean = np.mean(y_test, axis=0)        # shape: (6, 4205)
    Y_pred_mean = np.mean(y_pred, axis=0)       # shape: (6, 4205)
    if y_std is not None:
        Y_std_mean = np.mean(y_std, axis=0)        # shape: (6, 4205)

    # plot mean true vs predicted
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Mean Predicted vs True Functions — {exp_id}", fontsize=16, y=1.02)

    for i in range(globals.N_FUNCTIONS):  # iterate over functions
        plt.subplot(2, 3, i + 1)

        plt.plot(wavelengths, Y_pred_mean[i], label="Mean Predicted")
        if y_std is not None:
            # with Gaussian prior, 2 standard deviations should cover ~95% of the true function values
            plt.fill_between(wavelengths, Y_pred_mean[i] - 2 * Y_std_mean[i], Y_pred_mean[i] + 2 * Y_std_mean[i], color="blue", alpha=0.2, label="Predicted Std Dev")
        plt.plot(wavelengths, Y_true_mean[i], label="Mean True")

        plt.title(f"{globals.function_names_plots[i]}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Function Value (" + globals.function_units_plots[i] + ")")
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path + f"{exp_id}_predicted_vs_true.png", dpi=300, bbox_inches="tight")
    plt.show()


def show_residuals(y_test, y_pred, wavelengths, exp_id="EXP_ID", save_path="nn_saves/testing_results/"):
    """Plot output residuals across functions and wavelengths.

    Args:
        y_test: Reference test outputs.
        y_pred: Predicted test outputs.
        wavelengths: Wavelength array corresponding to the final output axis.
        exp_id: Experiment identifier used in plot titles and filenames.
        save_path: Directory prefix used when saving the plots.

    Returns:
        None. Displays and saves the plots.
    """
    """
    Show the residuals of the predictions on the test set.
    - inputs: true test outputs, predicted test outputs, wavelengths
    - outputs: displays plots of mean residuals for each function
    """
    # --- residuals for all samples ---
    residuals = y_pred - y_test   # shape: (n_samples, 6, 4205)

    # mean residual across samples
    mean_residuals = np.mean(residuals, axis=0)  # shape: (6, 4205)

    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Residuals — {exp_id}", fontsize=16, y=1.02)
    for i in range(globals.N_FUNCTIONS):
        plt.subplot(2, 3, i + 1)

        plt.plot(wavelengths, mean_residuals[i])
        plt.axhline(0, linestyle="--")

        plt.title(f"Mean Residuals - {globals.function_names_plots[i]}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Prediction Error")

    plt.tight_layout()
    plt.savefig(save_path + f"{exp_id}_residuals.png", dpi=300, bbox_inches="tight")
    plt.show()


def show_predicted_vs_true_reflectance(rho_ref, rho_ret, rho_std, wavelengths, exp_id="EXP_ID", save_path="gp_saves/amlec_testing_results/single_kernel/"):
    """Plot reference and retrieved reflectance with uncertainty.

    Args:
        rho_ref: Reference surface reflectance.
        rho_ret: Retrieved surface reflectance.
        rho_std: Standard deviation of retrieved reflectance.
        wavelengths: Wavelength array corresponding to the spectral axis.
        exp_id: Experiment identifier used in plot titles and filenames.
        save_path: Directory prefix used when saving the plots.

    Returns:
        None. Displays and saves the plots.
    """
    """
    Show the reference vs retrieved reflectance across all samples, with optional uncertainty bands.
    - inputs: reference reflectance (rho_ref, shape [n_wvl]), retrieved reflectance
      (rho_ret, shape [n_samples, n_wvl]), propagated std dev (rho_std, optional,
      shape [n_samples, n_wvl]), wavelengths
    - outputs: displays a single plot of mean retrieved vs reference reflectance
    """
    # --- compute mean retrieved reflectance across all samples ---
    rho_ret_mean = np.mean(rho_ret, axis=0)   # shape: (n_wvl,)
    if rho_std is not None:
        rho_std_mean = np.mean(rho_std, axis=0)   # shape: (n_wvl,)

    plt.figure(figsize=(10, 6))
    plt.suptitle(f"Mean Retrieved vs Reference Reflectance — {exp_id}", fontsize=16, y=1.02)

    plt.plot(wavelengths, rho_ret_mean, label="Mean Retrieved")
    if rho_std is not None:
        # with Gaussian prior, 2 standard deviations should cover ~95% of the true reflectance values
        plt.fill_between(wavelengths, rho_ret_mean - 2 * rho_std_mean, rho_ret_mean + 2 * rho_std_mean, color="blue", alpha=0.2, label="Retrieved Std Dev")
    plt.plot(wavelengths, rho_ref, label="Reference")

    plt.title("Reflectance")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path + f"{exp_id}_predicted_vs_true_reflectance.png", dpi=300, bbox_inches="tight")
    plt.show()


def show_residuals_reflectance(rho_ref, rho_ret, wavelengths, exp_id="EXP_ID", save_path="gp_saves/amlec_testing_results/single_kernel/"):
    """Plot residuals between reference and retrieved reflectance.

    Args:
        rho_ref: Reference surface reflectance.
        rho_ret: Retrieved surface reflectance.
        wavelengths: Wavelength array corresponding to the spectral axis.
        exp_id: Experiment identifier used in plot titles and filenames.
        save_path: Directory prefix used when saving the plots.

    Returns:
        None. Displays and saves the plots.
    """
    """
    Show the residuals of the retrieved reflectance on the test set.
    - inputs: reference reflectance (rho_ref, shape [n_wvl]), retrieved reflectance
      (rho_ret, shape [n_samples, n_wvl]), wavelengths
    - outputs: displays a single plot of mean reflectance residuals
    """
    # --- residuals for all samples ---
    residuals = rho_ret - rho_ref   # shape: (n_samples, n_wvl)

    # mean residual across samples
    mean_residuals = np.mean(residuals, axis=0)  # shape: (n_wvl,)

    plt.figure(figsize=(10, 6))
    plt.suptitle(f"Reflectance Residuals — {exp_id}", fontsize=16, y=1.02)

    plt.plot(wavelengths, mean_residuals)
    plt.axhline(0, linestyle="--")

    plt.title("Mean Residuals - Reflectance")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Prediction Error")
    plt.ylim(-0.1, 0.1)

    plt.tight_layout()
    plt.savefig(save_path + f"{exp_id}_residuals_reflectance.png", dpi=300, bbox_inches="tight")
    plt.show()


def show_multiple_experiments_errors(errors_per_wvl, wavelengths, save_path="./", xlims=[], ylims=[], ylabel="Error", title="Errors"):
    """Plot wavelength-dependent errors for multiple experiments.

    Args:
        errors_per_wvl: Error arrays to plot, one series per experiment.
        wavelengths: Wavelength array corresponding to the error values.
        save_path: Destination path for the saved figure.
        xlims: Optional wavelength-axis limits.
        ylims: Optional error-axis limits.
        ylabel: Label for the error axis.
        title: Figure title.

    Returns:
        None. Displays and saves the figure.
    """
    plt.figure(figsize=(10, 6), dpi=300)
    for exp_id, err_per_wvl in errors_per_wvl.items():
        plt.plot(wavelengths, err_per_wvl, label=exp_id)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(ylabel)
    if len(xlims) > 0:
        plt.xlim(xlims[0], xlims[1])
        mask = (wavelengths >= xlims[0]) & (wavelengths <= xlims[1])
    if len(ylims) > 0:
        plt.ylim(ylims[0], ylims[1])
    else:
        errors_in_region = np.concatenate([err_per_wvl[mask] for err_per_wvl in errors_per_wvl.values()])
        plt.ylim(np.min(errors_in_region) - 0.1, np.max(errors_in_region) + 0.1)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()




