"""Backward-compatible imports for the thesis utility functions."""

# ----------------------------
# Utilities for loading data, preprocessing, and evaluation
# ----------------------------

from data_utils import (
    inspect_metadata,
    load_train_h5,
    load_test_csv,
    load_csv_last_id,
    train_val_test_split,
)
from preprocessing import (
    apply_pca,
    scale_input_data,
    scale_input_data_feature_selection,
    scale_output_data,
)
from metrics import (
    retrieve_reflectance,
    mre_amlec,
    mae_amlec,
    propagate_reflectance_uncertainty,
    build_mask,
    mre_score,
    mae_score,
    calculate_coverage,
)
from plotting import (
    show_fit_val_summary,
    show_test_summary,
    show_top_results,
    show_barplot_results,
    show_test_results_mre,
    show_test_results_mae,
    show_predicted_vs_true,
    show_residuals,
    show_predicted_vs_true_reflectance,
    show_residuals_reflectance,
    show_multiple_experiments_errors,
)
from gp_utils import inverse_mean_transform, inverse_std_transform
from nn_experiments import (
    ARCHITECTURES,
    ENCODER_VERSIONS,
    SCALE_TYPES,
    FULL_DS_MODELS,
    REDUCED_DS_MODELS,
    BATCH_SIZE,
    N_EPOCHS,
    PATIENCE,
    nn_create_datasets,
    nn_prepare_all_experiments,
    nn_get_loaders_and_scalers,
    nn_calculate_metrics,
    nn_run_experiment,
    nn_run_all_experiments,
)
