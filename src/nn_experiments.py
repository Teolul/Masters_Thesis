"""Neural network dataset preparation and experiment runners."""

from pathlib import Path
import random
import re
import copy
import itertools
import time
import pickle
import inspect
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
import torch

import globals
import nn_models
import nn_dataset
from preprocessing import (
    apply_pca,
    scale_input_data,
    scale_input_data_feature_selection,
    scale_output_data,
)
from metrics import mae_score, mre_score

# ==================== EXPERIMENT GRID ====================
ARCHITECTURES = {
    "EmulatorSet1": nn_models.EmulatorSet1,
    "EmulatorSet2": nn_models.EmulatorSet2,
    "EmulatorSet3": nn_models.EmulatorSet3,
    "EmulatorSet4": nn_models.EmulatorSet4,
    "EmulatorSet5": nn_models.EmulatorSet5,
    "EmulatorSet6": nn_models.EmulatorSet6
}

ENCODER_VERSIONS = [
    "single",
    "multi",
]

SCALE_TYPES = [
    "minmax", 
    "standard",
]

# which model families use the full dataset vs. the reduced one
FULL_DS_MODELS    = {"EmulatorSet1", "EmulatorSet5", "EmulatorSet6"}
REDUCED_DS_MODELS = {"EmulatorSet2", "EmulatorSet3", "EmulatorSet4"}

BATCH_SIZE = 64 #4, 16, 64
N_EPOCHS = 100
PATIENCE = 25


def nn_create_datasets(X_tr, X_val, Y_tr, Y_val, X_test, Y_test, verbose=True):
    """Create PyTorch datasets for training, validation, and test data.

    Args:
        X_tr: Training inputs, as an array or a list of per-function arrays.
        X_val: Validation inputs in the same representation as ``X_tr``.
        Y_tr: Training targets.
        Y_val: Validation targets.
        X_test: Test inputs in the same representation as ``X_tr``.
        Y_test: Test targets.
        verbose: Whether to print dataset lengths and one item’s shapes.

    Returns:
        Training, validation, and test dataset objects, in that order.
    """
    if not isinstance(X_tr, list):
        train_ds = nn_dataset.MyDataset(X_tr, Y_tr)
        val_ds = nn_dataset.MyDataset(X_val, Y_val)
        test_ds = nn_dataset.MyDataset(X_test, Y_test)
    else:
        train_ds = nn_dataset.MyDataset2(X_tr, Y_tr)
        val_ds = nn_dataset.MyDataset2(X_val, Y_val)
        test_ds = nn_dataset.MyDataset2(X_test, Y_test)

    if verbose:
        print("Train dataset length:", len(train_ds))
        print("Val dataset length:", len(val_ds))
        print("Test dataset length:", len(test_ds))

        # get item check
        x, y = train_ds.__getitem__(0)
        if isinstance(x, (list, tuple)):
            print("Input shapes:", [item.shape for item in x])
        else:
            print("Input shape:", x.shape)
        print("Output shape:", y.shape)
        print()

    return train_ds, val_ds, test_ds


def nn_build_model(model_name, encoder_version, config, scale_type, device,
                   wavelengths, region_configs=None):
    """Create a registered model using the constructor options it supports.

    Args:
        model_name: Registered key from ``ARCHITECTURES``.
        encoder_version: Encoder configuration for models that accept one.
        config: Prepared experiment configuration containing fitted input scalers.
        scale_type: Scaling configuration key used to select input scalers.
        device: PyTorch device for the model.
        wavelengths: Wavelength array passed to compatible model constructors.
        region_configs: Optional spectral region configuration.

    Returns:
        The initialized model on ``device``.
    """
    if model_name not in ARCHITECTURES:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {list(ARCHITECTURES)}.")

    model_class = ARCHITECTURES[model_name]
    parameters = inspect.signature(model_class.__init__).parameters
    options = {
        "encoder_type": encoder_version,
        "wavelengths": wavelengths,
        "region_configs": region_configs,
    }
    if "input_dims" in parameters:
        scalers = config["x_scalers"][scale_type]
        if not isinstance(scalers, (list, tuple)):
            raise ValueError(
                f"{model_name} requires per-function feature inputs; prepare the selected-feature dataset first."
            )
        options["input_dims"] = [scaler.n_features_in_ for scaler in scalers]

    model = model_class(**{name: value for name, value in options.items() if name in parameters})
    return model.to(device)


def nn_forward(model, inputs):
    """Run a model and return its prediction tensor.

    Args:
        model: Initialized neural network model.
        inputs: Batch input tensor or nested collection of tensors.

    Returns:
        Prediction tensor, discarding optional auxiliary model outputs.
    """
    outputs = model(inputs)
    return outputs[0] if isinstance(outputs, tuple) else outputs


def _move_batch_to_device(value, device):
    """Move a tensor or a nested input collection to a PyTorch device.

    Args:
        value: Tensor, list, or tuple containing batch data.
        device: Destination PyTorch device.

    Returns:
        The input structure with each tensor moved to ``device``.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_batch_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_batch_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_batch_to_device(item, device) for key, item in value.items()}
    return value


def nn_prepare_all_experiments(X_tr, X_val, X_test, Y_tr, Y_val, Y_test, n_pca_components=10):
    """Prepare scaled datasets and transformation objects for model experiments.

    Args:
        X_tr: Training inputs, as an array or a list of per-function arrays.
        X_val: Validation inputs corresponding to ``X_tr``.
        X_test: Test inputs corresponding to ``X_tr``.
        Y_tr: Training outputs.
        Y_val: Validation outputs.
        Y_test: Test outputs.
        n_pca_components: Number of PCA components used for reduced-output models.

    Returns:
        Configuration dictionary containing scalers, PCA objects, datasets, and
        training hyperparameters for the experiment runners.
    """
    config = {
        "x_scalers":               {},
        "y_scalers":               {},
        "y_scalers_reduced":       {},
        "pca_lists":               {},
        "train_ds_scaled":         {},
        "val_ds_scaled":           {},
        "test_ds_scaled":          {},
        "train_ds_reduced_scaled": {},
        "val_ds_reduced_scaled":   {},
        "test_ds_reduced_scaled":  {},
    }

    # PCA is fit on raw outputs — independent of scale type, so compute once
    pca_list, Y_tr_reduced, Y_val_reduced = apply_pca(Y_tr, Y_val, n_components=n_pca_components)
    Y_test_reduced = np.zeros((Y_test.shape[0], globals.N_FUNCTIONS, n_pca_components))
    for i in range(globals.N_FUNCTIONS):
        Y_test_reduced[:, i, :] = pca_list[i].transform(Y_test[:, i, :])

    for scale_type in SCALE_TYPES:
        print(f"\n── Preparing [{scale_type}] ──────────────────────────────")

        # --- inputs ---

        if not isinstance(X_tr, list):
            x_scaler, X_tr_scaled, X_val_scaled = scale_input_data(
                X_tr, X_val, scale_type=scale_type
            )
            X_test_scaled = x_scaler.transform(X_test)
        else:
            x_scaler, X_tr_scaled, X_val_scaled = scale_input_data_feature_selection(
                X_tr, X_val, scale_type=scale_type
            )
            X_test_scaled = [x_scaler[i].transform(X_test[i]) for i in range(globals.N_FUNCTIONS)]

        # --- full outputs ---
        y_scalers, Y_tr_scaled, Y_val_scaled = scale_output_data(
            Y_tr, Y_val, scale_type=scale_type
        )
        Y_test_scaled = np.zeros_like(Y_test)
        for i in range(globals.N_FUNCTIONS):
            Y_test_scaled[:, i, :] = y_scalers[i].transform(Y_test[:, i, :])

        # --- reduced outputs ---
        y_scalers_reduced, Y_tr_reduced_scaled, Y_val_reduced_scaled = scale_output_data(
            Y_tr_reduced, Y_val_reduced, scale_type=scale_type
        )
        Y_test_reduced_scaled = np.zeros((Y_test.shape[0], globals.N_FUNCTIONS, n_pca_components))
        for i in range(globals.N_FUNCTIONS):
            Y_test_reduced_scaled[:, i, :] = y_scalers_reduced[i].transform(Y_test_reduced[:, i, :])

        # --- datasets ---
        train_ds_scaled, val_ds_scaled, test_ds_scaled = nn_create_datasets(
            X_tr_scaled, X_val_scaled, Y_tr_scaled, Y_val_scaled, X_test_scaled, Y_test_scaled, verbose=False
        )
        train_ds_reduced_scaled, val_ds_reduced_scaled, test_ds_reduced_scaled = nn_create_datasets(
            X_tr_scaled, X_val_scaled, Y_tr_reduced_scaled, Y_val_reduced_scaled,
            X_test_scaled, Y_test_reduced_scaled, verbose=False
        )

        # --- store everything under the scale_type key ---
        config["x_scalers"][scale_type]               = x_scaler
        config["y_scalers"][scale_type]               = y_scalers
        config["y_scalers_reduced"][scale_type]       = y_scalers_reduced
        config["pca_lists"][scale_type]               = pca_list        # same for both
        config["train_ds_scaled"][scale_type]         = train_ds_scaled
        config["val_ds_scaled"][scale_type]           = val_ds_scaled
        config["test_ds_scaled"][scale_type]          = test_ds_scaled
        config["train_ds_reduced_scaled"][scale_type] = train_ds_reduced_scaled
        config["val_ds_reduced_scaled"][scale_type]   = val_ds_reduced_scaled
        config["test_ds_reduced_scaled"][scale_type]  = test_ds_reduced_scaled

    # training hyperparameters
    config["batch_size"] = BATCH_SIZE
    config["n_epochs"]   = N_EPOCHS
    config["patience"]   = PATIENCE

    return config


# ==================== DATASET / SCALER ROUTER ====================
def nn_get_loaders_and_scalers(model_name, scale_type, config):
    """Select experiment datasets and create training and validation loaders.

    Args:
        model_name: Name of the model family being run.
        scale_type: Scaling configuration key.
        config: Prepared experiment configuration.

    Returns:
        Training and validation datasets and loaders, output scalers, and the
        PCA objects used for reduced-output models.
    """
    if model_name in FULL_DS_MODELS:
        train_ds = config["train_ds_scaled"][scale_type]
        val_ds   = config["val_ds_scaled"][scale_type]
        y_scalers = config["y_scalers"][scale_type]
        pca_list = None
    else:
        train_ds = config["train_ds_reduced_scaled"][scale_type]
        val_ds   = config["val_ds_reduced_scaled"][scale_type]
        y_scalers = config["y_scalers_reduced"][scale_type]
        pca_list = config["pca_lists"][scale_type]

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"], num_workers=0, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=config["batch_size"], num_workers=0, shuffle=False)
    return train_ds, val_ds, train_dl, val_dl, y_scalers, pca_list


def nn_calculate_metrics(y_pred, Y_batch, wavelengths, y_scalers, pca_list):
    """Calculate MRE and MAE after restoring predictions to original units.

    Args:
        y_pred: Batch predictions, potentially scaled or PCA-reduced.
        Y_batch: Batch targets in the corresponding representation.
        wavelengths: Wavelength array used by the evaluation metrics.
        y_scalers: Per-function fitted scalers, or ``None`` when not scaled.
        pca_list: Per-function fitted PCA objects, or ``None`` when not reduced.

    Returns:
        Batch MRE and MAE values in the original output space.
    """
    is_scaled  = y_scalers is not None
    is_reduced = pca_list  is not None

    if is_scaled or is_reduced:
        # prepare tensors to hold the restored predictions and targets in original space
        y_pred_og_shape = torch.zeros((y_pred.size(0), globals.N_FUNCTIONS, len(wavelengths)), device=y_pred.device)
        y_true_og_shape = torch.zeros((Y_batch.size(0), globals.N_FUNCTIONS, len(wavelengths)), device=Y_batch.device)

        # inverse transform the scaling and PCA to get back to original space if needed
        for i in range(globals.N_FUNCTIONS):
            y_pred_restored = y_pred[:, i, :].cpu().detach().numpy()
            y_true_restored = Y_batch[:, i, :].cpu().detach().numpy()
            if is_scaled:
                y_pred_restored = y_scalers[i].inverse_transform(y_pred_restored)
                y_true_restored = y_scalers[i].inverse_transform(y_true_restored)
            if is_reduced:
                y_pred_restored = pca_list[i].inverse_transform(y_pred_restored)
                y_true_restored = pca_list[i].inverse_transform(y_true_restored)
            y_pred_og_shape[:, i, :] = torch.from_numpy(y_pred_restored).to(y_pred_og_shape.device)
            y_true_og_shape[:, i, :] = torch.from_numpy(y_true_restored).to(y_true_og_shape.device)

        batch_train_mre_unscaled = mre_score(y_true_og_shape.cpu().detach().numpy(), y_pred_og_shape.cpu().detach().numpy(), wavelengths)
        batch_train_mae_unscaled = mae_score(y_true_og_shape.cpu().detach().numpy(), y_pred_og_shape.cpu().detach().numpy(), wavelengths)
    else:
        batch_train_mre_unscaled = mre_score(Y_batch.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), wavelengths)
        batch_train_mae_unscaled = mae_score(Y_batch.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), wavelengths)

    return batch_train_mre_unscaled, batch_train_mae_unscaled


# ==================== SINGLE EXPERIMENT ====================
def nn_run_experiment(model_name, encoder_version, scale_type, config, device, wavelengths, region_configs=None, seed=42):
    """Train one neural network configuration and save its model and history.

    Args:
        model_name: Model family key from ``ARCHITECTURES``.
        encoder_version: Encoder configuration passed to the selected model.
        scale_type: Scaling configuration key.
        config: Prepared datasets, scalers, and training settings.
        device: PyTorch device used for model training.
        wavelengths: Wavelength array used for evaluation and model setup.
        region_configs: Optional spectral region configuration for supported models.
        seed: Random seed used by the experiment.

    Returns:
        The trained model with best validation MRE, its history, and result record.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed == 42:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    dataset_size = int(re.search(r'\d+', Path(globals.CURRENT_TRAIN_FILE).stem).group())
    exp_id = f"{model_name}_{encoder_version}_{scale_type}_{dataset_size}"
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {exp_id}")
    print(f"{'='*60}")

    # --- build model ---
    model = nn_build_model(
        model_name, encoder_version, config, scale_type, device, wavelengths, region_configs
    )

    # --- data ---
    train_ds, val_ds, train_dl, val_dl, y_scalers, pca_list = nn_get_loaders_and_scalers(
        model_name, scale_type, config
    )

    # --- optimiser / loss / scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5
    )

    # --- early stopping state ---
    n_epochs        = config.get("n_epochs", 100)
    patience        = config.get("patience", 25)
    best_val_mre    = float("inf")
    patience_counter = 0
    best_model_wts  = copy.deepcopy(model.state_dict())
    history         = defaultdict(list)

    start_time = time.time()

    for epoch in range(n_epochs):
        # ---------- TRAIN ----------
        model.train()
        epoch_train_loss = epoch_train_mre = epoch_train_mae = 0.0
        # accumulate preds and targets to compute metrics once at the end of the loop to avoid aggregation artifacts
        all_preds   = []
        all_targets = []

        for X_batch, Y_batch in tqdm(train_dl, desc=f"[{exp_id}] E{epoch+1} Train", leave=False):
            X_batch = _move_batch_to_device(X_batch, device)
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad()
            y_pred = nn_forward(model, X_batch)
            loss   = criterion(y_pred, Y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * Y_batch.size(0)
            all_preds.append(y_pred.detach().cpu())
            all_targets.append(Y_batch.cpu())

        all_preds   = torch.cat(all_preds,   dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        epoch_train_mre, epoch_train_mae = nn_calculate_metrics(all_preds, all_targets, wavelengths, y_scalers, pca_list)
        epoch_train_loss /= len(train_ds)

        # ---------- VALIDATE ----------
        model.eval()
        epoch_val_loss = epoch_val_mre = epoch_val_mae = 0.0
        all_preds   = []
        all_targets = []

        with torch.no_grad():
            for X_batch, Y_batch in tqdm(val_dl, desc=f"[{exp_id}] E{epoch+1} Val", leave=False):
                X_batch = _move_batch_to_device(X_batch, device)
                Y_batch = Y_batch.to(device)
                y_pred = nn_forward(model, X_batch)
                loss   = criterion(y_pred, Y_batch)

                epoch_val_loss += loss.item() * Y_batch.size(0)
                all_preds.append(y_pred.cpu())
                all_targets.append(Y_batch.cpu())

        all_preds   = torch.cat(all_preds,   dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        epoch_val_mre, epoch_val_mae = nn_calculate_metrics(all_preds, all_targets, wavelengths, y_scalers, pca_list)
        epoch_val_loss /= len(val_ds)

        scheduler.step(epoch_val_mre)

        # record
        history["train_loss"].append(epoch_train_loss)
        history["train_mre"].append(epoch_train_mre)
        history["train_mae"].append(epoch_train_mae)
        history["val_loss"].append(epoch_val_loss)
        history["val_mre"].append(epoch_val_mre)
        history["val_mae"].append(epoch_val_mae)

        print(
            f"  E{epoch+1:03d} | "
            f"train loss {epoch_train_loss:.5f}  mre {epoch_train_mre:.5f}  mae {epoch_train_mae:.5f} | "
            f"val loss {epoch_val_loss:.5f}  mre {epoch_val_mre:.5f}  mae {epoch_val_mae:.5f}"
        )

        # ---------- EARLY STOPPING ----------
        if epoch_val_mre < best_val_mre:
            best_val_mre   = epoch_val_mre
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("  --> best val MRE — weights saved")
        else:
            patience_counter += 1
            print(f"  --> no improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("  !!! early stopping !!!")
                break

    elapsed = time.time() - start_time

    # reload best weights
    model.load_state_dict(best_model_wts)

    # persist model
    nn_save_dir = globals.NN_RESULTS_ROOT
    if seed == 42:
        model_path = nn_save_dir / "models" / f"{exp_id}.pth"
    else:
        model_path = nn_save_dir / "ensemble_models" / f"{exp_id}_{seed}.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # save history
    if seed == 42:
        history_path = nn_save_dir / "histories" / f"{exp_id}_history.pkl"
    else:
        history_path = nn_save_dir / "histories" / f"{exp_id}_{seed}_history.pkl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    # build result row
    idx_best = int(np.argmin(history["val_mre"]))
    result = {
        "experiment_id":  exp_id,
        "model":          model_name,
        "encoder":        encoder_version,
        "scale_type":     scale_type,
        "dataset_size":   dataset_size,
        "fit_time":       elapsed,
        "best_epoch":     idx_best + 1,
        "best_train_loss": history["train_loss"][idx_best],
        "best_val_loss":   history["val_loss"][idx_best],
        "best_train_mre":  history["train_mre"][idx_best],
        "best_val_mre":    history["val_mre"][idx_best],
        "best_train_mae":  history["train_mae"][idx_best],
        "best_val_mae":    history["val_mae"][idx_best],
    }
    return model, history, result


# ==================== FULL EXPERIMENT LOOP ====================
def nn_run_all_experiments(config, device, wavelengths, region_configs=None, seed=42,
                           model_names=None, encoder_versions=None, scale_types=None):
    """Run the configured model, encoder, and scaling combinations.

    Args:
        config: Prepared datasets, scalers, and training settings.
        device: PyTorch device used for model training.
        wavelengths: Wavelength array used for evaluation and model setup.
        region_configs: Optional spectral region configuration for supported models.
        seed: Random seed used by each experiment.
        model_names: Optional subset of registered model names to run.
        encoder_versions: Optional subset of encoder versions to run.
        scale_types: Optional subset of scaling configurations to run.

    Returns:
        A sorted results dataframe, the last successfully assigned model, and
        its history.
    """

    if seed == 42:
        results_path = globals.NN_RESULTS_ROOT / "validation" / "nn_val_results.csv"
    else:
        results_path = globals.NN_RESULTS_ROOT / "validation" / "nn_val_results_uncertainty.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    all_results  = []

    selected_models = list(ARCHITECTURES) if model_names is None else list(model_names)
    selected_encoders = ENCODER_VERSIONS if encoder_versions is None else list(encoder_versions)
    selected_scales = SCALE_TYPES if scale_types is None else list(scale_types)
    if not selected_models or not selected_encoders or not selected_scales:
        raise ValueError("Select at least one model, encoder version, and scale type.")
    unknown_models = set(selected_models) - set(ARCHITECTURES)
    if unknown_models:
        raise ValueError(f"Unknown model names: {sorted(unknown_models)}")
    grid = list(itertools.product(selected_models, selected_encoders, selected_scales))
    print(f"Total experiments to run: {len(grid)}")

    for model_name, encoder_version, scale_type in grid:
        try:
            model, history, result = nn_run_experiment(
                model_name, encoder_version, scale_type,
                config, device, wavelengths, region_configs, seed
            )
            all_results.append(result)

            # append to CSV after every experiment so a crash doesn't lose data
            row_df = pd.DataFrame([result])
            row_df.to_csv(
                results_path,
                mode="a",
                header=not results_path.exists(),
                index=False,
            )
            print(f"  Saved results for {result['experiment_id']}\n")

        except Exception as e:
            print(f"  [ERROR] {model_name}__{encoder_version}__{scale_type} failed: {e}")
            continue

    summary = pd.DataFrame(all_results).sort_values("best_val_mre")
    print("\n===== EXPERIMENT SUMMARY (sorted by val MRE) =====")
    print(summary[["experiment_id", "best_val_mre", "best_val_mae", "best_epoch", "fit_time"]].to_string(index=False))
    return summary, model, history



