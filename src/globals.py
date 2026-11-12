from pathlib import Path
import os

# Project and dataset roots are anchored to this file so notebook working directories do not affect input data selection.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT.parent / "AMLEC-1" / "rtm_emulation"
# Historical thesis outputs are retained as read-only reference material.
LEGACY_RESULTS_ROOT = PROJECT_ROOT / "thesis_results"
RESULTS_ROOT = PROJECT_ROOT / "research_results"
NN_RESULTS_ROOT = RESULTS_ROOT / "neural_networks"
GP_RESULTS_ROOT = RESULTS_ROOT / "gaussian_processes"


def nn_output_path(relative_path):
    """Create and return a path for a new neural network result.

    Args:
        relative_path: Relative result path, optionally beginning with ``nn_saves``.

    Returns:
        An absolute path under ``research_results/neural_networks``.
    """
    path = _result_subpath(relative_path, "nn_saves")
    output_path = NN_RESULTS_ROOT / path
    is_directory = str(relative_path).endswith(("/", "\\"))
    (output_path if is_directory else output_path.parent).mkdir(parents=True, exist_ok=True)
    output = str(output_path)
    return output + os.sep if is_directory else output


def gp_output_path(relative_path):
    """Create and return a path for a new Gaussian process result.

    Args:
        relative_path: Relative result path, optionally beginning with ``gp_saves``.

    Returns:
        An absolute path under ``research_results/gaussian_processes``.
    """
    path = _result_subpath(relative_path, "gp_saves")
    output_path = GP_RESULTS_ROOT / path
    is_directory = str(relative_path).endswith(("/", "\\"))
    (output_path if is_directory else output_path.parent).mkdir(parents=True, exist_ok=True)
    output = str(output_path)
    return output + os.sep if is_directory else output

# Scenario A
TRAIN_FILE_A_500 = ROOT / "scenarioA" / "train" / "train500.h5"
TRAIN_FILE_A_2000 = ROOT / "scenarioA" / "train" / "train2000.h5"
TRAIN_FILE_A_10000 = ROOT / "scenarioA" / "train" / "train10000.h5"
TEST_FILE_A_INTERP = ROOT / "scenarioA" / "reference" / "refInterp.h5"
TEST_FILE_A_EXTRAP = ROOT / "scenarioA" / "reference" / "refExtrap.h5"
REF_LDB = ROOT / "scenarioA" / "reference" / "refldb.txt"

# "current" selection (mutable state)
CURRENT_TRAIN_FILE = TRAIN_FILE_A_10000
CURRENT_TEST_FILE = TEST_FILE_A_INTERP

N_FUNCTIONS = 6
N_INPUTS = 9

# plotting info
param_names_plots = ['Ozone (atm-cm)', 'Columnar Water Vapor (g/cm^2)', 'Aerosol Optical Depth (-)', 'Asymmetry Parameter (-)', 'Angstrom Exponent (-)',
                     'Single Scattering Albedo (-)', 'Solar Zenith Angle (deg)', 'View Zenith Angle (deg)', 'Relative Azimuth Angle (deg)']
param_names_plots_abb = ['O3', 'CWV', 'AOD', 'Asy. Param.', 'Ang. Exp.', 'SSA', 'SZA', 'VZA', 'RAA']

function_names_plots = ['Path Radiance', 'Direct Solar Irradiance', 'Diffuse Solar Irradiance', 'Spherical Albedo', 'Direct Transmittance', 'Diffuse Transmittance']
function_names_plots_abb = ['Path Rad.', 'Direct Irrad.', 'Diffuse Irrad.', 'Sph. Alb.', 'Direct Trans.', 'Diffuse Trans.']
function_names_results = ['path_rad', "dir_irr", "diff_irr", "sph_alb", "dir_trans", "diff_trans"]

function_units_plots = ["mW·m⁻²·sr⁻¹·nm⁻¹", "mW·m⁻²·nm⁻¹", "mW·m⁻²·nm⁻¹", "-", "-", "-"]


def _result_subpath(relative_path, legacy_prefix):
    """Map a legacy result path to its organized research-result location.

    Args:
        relative_path: Legacy result path.
        legacy_prefix: Leading directory name to remove from the legacy path.

    Returns:
        Relative path with known legacy directory names mapped to current names.
    """
    path = Path(relative_path)
    if path.parts and path.parts[0] == legacy_prefix:
        path = Path(*path.parts[1:])
    aliases = {
        "model_saves": "models",
        "model_saves_uncertainty": "ensemble_models",
        "validation_results": "validation",
        "testing_results": "testing",
        "amlec_testing_results": "amlec_testing",
        "other_plots": "plots",
        "nodimred_resources": "dimensionality_reduction",
    }
    if (
        len(path.parts) > 1
        and path.parts[0] in {"model_saves", "model_saves_uncertainty"}
        and path.name.endswith("_history.pkl")
    ):
        return Path("histories", path.name)
    if path.parts and path.parts[0] in aliases:
        path = Path(aliases[path.parts[0]], *path.parts[1:])
    return path


def nn_input_path(relative_path):
    """Find a neural network result in the research or legacy results tree.

    Args:
        relative_path: Relative result path, optionally beginning with ``nn_saves``.

    Returns:
        The research result path when present, otherwise the corresponding legacy path.
    """
    path = _result_subpath(relative_path, "nn_saves")
    research_path = NN_RESULTS_ROOT / path
    if research_path.exists():
        return research_path
    legacy_path = LEGACY_RESULTS_ROOT / Path(relative_path)
    return legacy_path


def gp_input_path(relative_path):
    """Find a Gaussian process result in the research or legacy results tree.

    Args:
        relative_path: Relative result path, optionally beginning with ``gp_saves``.

    Returns:
        The research result path when present, otherwise the corresponding legacy path.
    """
    path = _result_subpath(relative_path, "gp_saves")
    research_path = GP_RESULTS_ROOT / path
    if research_path.exists():
        return research_path
    legacy_path = LEGACY_RESULTS_ROOT / Path(relative_path)
    return legacy_path
