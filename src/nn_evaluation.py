"""Neural network model loading and prediction helpers for evaluation."""

import torch

import globals
from nn_experiments import FULL_DS_MODELS, nn_build_model, nn_forward


def _record_value(record, name):
    """Read a field from a mapping-like experiment result.

    Args:
        record: Experiment result mapping or pandas Series.
        name: Name of the result field.

    Returns:
        The corresponding field value.
    """
    return record[name]


def nn_load_model(record, config, device, wavelengths, region_configs=None, seed=None):
    """Build a registered model and load its saved experiment weights.

    Args:
        record: Experiment result containing model, encoder, scale, and ID fields.
        config: Prepared experiment configuration with fitted input scalers.
        device: PyTorch device used for inference.
        wavelengths: Wavelength coordinates passed to compatible model constructors.
        region_configs: Optional spectral region configuration.
        seed: Optional ensemble seed selecting an uncertainty model checkpoint.

    Returns:
        Loaded model in evaluation mode.
    """
    model_name = _record_value(record, "model")
    encoder_version = _record_value(record, "encoder")
    scale_type = _record_value(record, "scale_type")
    experiment_id = _record_value(record, "experiment_id")
    model = nn_build_model(
        model_name, encoder_version, config, scale_type, device, wavelengths, region_configs
    )

    if seed is None:
        relative_path = f"nn_saves/model_saves/{experiment_id}.pth"
    else:
        relative_path = f"nn_saves/model_saves_uncertainty/{experiment_id}_{seed}.pth"
    state_dict = torch.load(globals.nn_input_path(relative_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def nn_predict(model, X, x_scalers, y_scalers, pca_list, wavelengths, device):
    """Predict samples and restore outputs to the original target space.

    Args:
        model: Loaded neural network model.
        X: Raw input array or per-function input arrays.
        x_scalers: Fitted input scaler or per-function scalers, or None.
        y_scalers: Per-function output scalers, or None.
        pca_list: Per-function PCA transforms, or None.
        wavelengths: Wavelength coordinates defining the restored output length.
        device: PyTorch device used for inference.

    Returns:
        Prediction array with shape ``(samples, functions, wavelengths)`` in the
        original output space.
    """
    if isinstance(x_scalers, (list, tuple)):
        if not isinstance(X, (list, tuple)) or len(X) != len(x_scalers):
            raise ValueError("Per-function input scalers require one input array per function.")
        inputs = [
            torch.as_tensor(
                scaler.transform(values) if scaler is not None else values,
                dtype=torch.float32,
                device=device,
            )
            for values, scaler in zip(X, x_scalers)
        ]
    else:
        scaled_inputs = x_scalers.transform(X) if x_scalers is not None else X
        inputs = torch.as_tensor(scaled_inputs, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        predictions = nn_forward(model, inputs)
        if y_scalers is not None or pca_list is not None:
            restored = torch.zeros(
                (predictions.size(0), globals.N_FUNCTIONS, len(wavelengths)),
                device=predictions.device,
            )
            for function_idx in range(globals.N_FUNCTIONS):
                values = predictions[:, function_idx, :].cpu().numpy()
                if y_scalers is not None:
                    values = y_scalers[function_idx].inverse_transform(values)
                if pca_list is not None:
                    values = pca_list[function_idx].inverse_transform(values)
                restored[:, function_idx, :] = torch.from_numpy(values).to(restored.device)
            predictions = restored

    return predictions.cpu().numpy()


def nn_test_model(record, X, config, device, wavelengths, region_configs=None):
    """Load an experiment model and predict the specified test inputs.

    Args:
        record: Experiment result row identifying the model and preprocessing.
        X: Raw test inputs in the representation expected by that model.
        config: Prepared experiment configuration containing fitted transforms.
        device: PyTorch device used for inference.
        wavelengths: Wavelength coordinates for output restoration.
        region_configs: Optional spectral region configuration.

    Returns:
        Predictions in the original output space as a NumPy array.
    """
    scale_type = _record_value(record, "scale_type")
    model_name = _record_value(record, "model")
    if model_name in FULL_DS_MODELS:
        y_scalers = config["y_scalers"][scale_type]
        pca_list = None
    else:
        y_scalers = config["y_scalers_reduced"][scale_type]
        pca_list = config["pca_lists"][scale_type]

    model = nn_load_model(record, config, device, wavelengths, region_configs)
    return nn_predict(
        model,
        X,
        config["x_scalers"][scale_type],
        y_scalers,
        pca_list,
        wavelengths,
        device,
    )


def nn_load_ensemble(record, seeds, config, device, wavelengths, region_configs=None):
    """Load saved ensemble members for one experiment configuration.

    Args:
        record: Experiment result row shared by the ensemble members.
        seeds: Seed values identifying the saved ensemble checkpoints.
        config: Prepared experiment configuration with fitted input scalers.
        device: PyTorch device used for inference.
        wavelengths: Wavelength coordinates passed to compatible constructors.
        region_configs: Optional spectral region configuration.

    Returns:
        Loaded models in the same order as ``seeds``.
    """
    return [
        nn_load_model(record, config, device, wavelengths, region_configs, seed=seed)
        for seed in seeds
    ]


def nn_ensemble_predict(models, X, record, config, wavelengths, device):
    """Predict with ensemble members and return mean and standard deviation.

    Args:
        models: Loaded ensemble models.
        X: Raw test inputs in the representation expected by the model.
        record: Experiment result row identifying its preprocessing configuration.
        config: Prepared experiment configuration containing fitted transforms.
        wavelengths: Wavelength coordinates for output restoration.
        device: PyTorch device used for inference.

    Returns:
        Mean and sample standard deviation predictions in the original output space.
    """
    scale_type = _record_value(record, "scale_type")
    model_name = _record_value(record, "model")
    if model_name in FULL_DS_MODELS:
        y_scalers = config["y_scalers"][scale_type]
        pca_list = None
    else:
        y_scalers = config["y_scalers_reduced"][scale_type]
        pca_list = config["pca_lists"][scale_type]

    predictions = [
        nn_predict(
            model,
            X,
            config["x_scalers"][scale_type],
            y_scalers,
            pca_list,
            wavelengths,
            device,
        )
        for model in models
    ]
    stacked = torch.stack([torch.from_numpy(values) for values in predictions], dim=0)
    return stacked.mean(dim=0).numpy(), stacked.std(dim=0).numpy()
