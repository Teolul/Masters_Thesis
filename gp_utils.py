import time
import copy
import numpy as np
import utils
from sklearn.gaussian_process import GaussianProcessRegressor

import globals

# ----------------------------
# Utilities for the Gaussian Process pipeline
# ----------------------------

def fit_gp(x_tr_scaled, y_tr_reduced_scaled, gp_kernel):
    """
    Fit a Gaussian Process Regressor for each output function. Reduced and scaled data is used to speed up fitting and improve performance
    - inputs: training inputs, training outputs, GP kernel to use for fitting
    - outputs: list of fitted GaussianProcessRegressor models for each output function and the time taken to fit all GPs
    """

    print("---------- Fitting Gaussian Process Regressor for each output function... ----------")

    gpr_list = []
    start_time_fit = time.time()
    for i in range(globals.N_FUNCTIONS):
        # more than 0 restarts causes a huge increase in fitting time without significant improvement in performance
        gpr = GaussianProcessRegressor(kernel=copy.deepcopy(gp_kernel), n_restarts_optimizer=0, random_state=42)
        gpr.fit(x_tr_scaled, y_tr_reduced_scaled[:, i, :])
        print("Learned kernel:", gpr.kernel_)
        gpr_list.append(gpr)
    end_time_fit = time.time()
    print(f"Time taken to fit GPs: {end_time_fit - start_time_fit:.2f} seconds")

    print("---------- Gaussian Process fitting completed. ----------\n")

    return gpr_list, end_time_fit - start_time_fit


def validate_gp(gpr_list, x_val_scaled, y_val, wavelengths, y_scalers, pca_list):
    """
    Validate the fitted Gaussian Process models on the validation set
    - inputs: list of fitted GaussianProcessRegressor models, validation inputs, validation outputs, wavelengths, list of scalers used for each output function, list of PCA objects for each output function
    - outputs: numpy array of predicted outputs on the validation set in the original space (n_samples, 6, 4205)
    """

    print("---------- Validating Gaussian Process models on the validation set... ----------")

    y_val_pred = np.zeros_like(y_val)  # (n_samples, 6, 4205)
    for i in range(globals.N_FUNCTIONS):
        y_pred_red_scaled = gpr_list[i].predict(x_val_scaled) # predictions in reduced and scaled space
        y_pred_red = y_scalers[i].inverse_transform(y_pred_red_scaled)  # inverse scaling
        y_pred = pca_list[i].inverse_transform(y_pred_red)  # inverse PCA
        y_val_pred[:, i, :] = y_pred

    print("y_val_pred shape:", y_val_pred.shape)

    # MRE calculation
    mre = utils.mre_score(y_val, y_val_pred, wavelengths)
    print(f"----- Validation MRE: {mre} -----")

    print("---------- Validation completed. ----------\n")

    return mre