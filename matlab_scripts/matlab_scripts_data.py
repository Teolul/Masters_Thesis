"""
Python conversion of scripts_data.m

This script has two independent parts (separated by %% in the original
MATLAB file, kept here as two functions you can run separately):

  1. `save_test_inputs_to_csv()` - dumps the header/query points of a
     reference .h5 file to a .csv file.
  2. `compute_baseline_predictions()` - fits a baseline "polyfit" emulator
     on a training LUT, uses it to predict at the test query points, and
     writes the predictions to a results .h5 file (with a runtime attribute).

IMPORTANT - missing helper functions:
This script calls two MATLAB functions, `readLUThdr` and `myinterpn`, that
were NOT included in the files you uploaded. I've added Python stand-ins
below (`read_lut_header` and `myinterpn`) that reproduce the parts of their
behavior implied by how they're used in the original script:

  - `readLUThdr(file)` -> returns a struct with a `.LUTheader` field. Since
    script_validation.m separately does `h5read(file, '/LUTheader')`
    directly, `read_lut_header` below simply reads that dataset. If your
    real `readLUThdr.m` does more (e.g. parses variable names/ranges from
    attributes), you'll need to port that logic in too.

  - `myinterpn(X, Y, [], 'polyfit')` -> called once to *fit* a model
    (returns `B`), and again as `myinterpn(Xq, 'polyfit', B)` to *predict*
    at new points `Xq` using the fitted model `B`. This is a custom
    look-up-table interpolator/regressor and its exact math isn't in the
    uploaded files, so it's stubbed out below with a placeholder
    polynomial-regression implementation. Replace `myinterpn` with a direct
    port of your actual function for correct results.
"""

import numpy as np
import h5py


BASE_PATH = r"C:\Users\Matteo\Desktop\Scuola\MastersThesis\AMLEC-1\rtm_emulation"


# --------------------------------------------------------------------------
# Stand-ins for helper functions not present in the uploaded files
# --------------------------------------------------------------------------
def read_lut_header(file_path):
    """
    Stand-in for MATLAB's readLUThdr(file). Reads the '/LUTheader' dataset
    and returns it as a dict with a 'LUTheader' key, mirroring how the
    MATLAB struct is used (hdr.LUTheader) in the original script.

    NOTE: if your real readLUThdr.m parses additional metadata (variable
    names, units, ranges, etc. from HDF5 attributes), add that here too.
    """
    with h5py.File(file_path, "r") as f:
        lutheader = np.asarray(f["/LUTheader"][()], dtype=float)
    return {"LUTheader": lutheader}


def myinterpn(*args):
    """
    PLACEHOLDER for MATLAB's myinterpn.m -- not included in the uploaded
    files, so its exact interpolation/regression method is unknown.

    Original call patterns:
        [~, B] = myinterpn(X, Y, [], 'polyfit')   # fit
        Yq      = myinterpn(Xq, 'polyfit', B)     # predict

    Below is a simple multivariate polynomial least-squares regression as
    a functional placeholder, matching the (fit) / (predict) call pattern.
    Replace with a direct port of your real myinterpn.m for correct results.
    """
    if len(args) == 4:
        # Fit mode: myinterpn(X, Y, _, 'polyfit')
        X, Y, _, method = args
        assert method == "polyfit", f"Unsupported method: {method}"

        # Simple linear (degree-1) design matrix as a placeholder
        design = np.hstack([np.ones((X.shape[0], 1)), X])
        # Solve Y ~ design @ B  (least squares) for each output row in Y.T
        B, *_ = np.linalg.lstsq(design, Y, rcond=None)
        return None, B  # first output unused in the original script (`~`)

    elif len(args) == 3:
        # Predict mode: myinterpn(Xq, 'polyfit', B)
        Xq, method, B = args
        assert method == "polyfit", f"Unsupported method: {method}"

        design = np.hstack([np.ones((Xq.shape[0], 1)), Xq])
        Yq = design @ B
        return Yq

    else:
        raise ValueError("Unexpected number of arguments to myinterpn")


# --------------------------------------------------------------------------
# Part 1: save test dataset inputs into a .csv file
# --------------------------------------------------------------------------
def save_test_inputs_to_csv(scenario="B"):
    file_path = f"{BASE_PATH}\\scenario{scenario}\\reference\\refInterp.h5"
    hdr = read_lut_header(file_path)
    X = hdr["LUTheader"]
    print(X.shape)

    csv_path = file_path.replace(".h5", "_NEW.csv")
    print(csv_path)
    np.savetxt(csv_path, X, delimiter=",")
    return X


# --------------------------------------------------------------------------
# Part 2: calculate baseline (polyfit) and emulator predictions
# --------------------------------------------------------------------------
def compute_baseline_predictions(scenario="A"):
    # Training data
    train_file = f"{BASE_PATH}\\scenario{scenario}\\train\\train2000.h5"
    hdr = read_lut_header(train_file)
    X = hdr["LUTheader"]
    with h5py.File(train_file, "r") as f:
        Y = np.asarray(f["/LUTdata"][()], dtype=float)

    # Query points (test inputs saved earlier)
    xq_path = f"{BASE_PATH}\\scenario{scenario}\\reference\\refInterp.csv"
    Xq = np.loadtxt(xq_path, delimiter=",").T

    # Fit baseline model, then predict at query points
    _, B = myinterpn(X, Y, None, "polyfit")

    import time
    t0 = time.time()
    Yq = myinterpn(Xq, "polyfit", B)
    elapsed = time.time() - t0
    print(f"Prediction time: {elapsed:.3f} s")

    # Write predictions + runtime attribute to results .h5 file
    out_file = f"{BASE_PATH}\\results\\baseline_{scenario}1_NEW.h5"
    with h5py.File(out_file, "w") as f:
        dset = f.create_dataset("/LUTdata", data=Yq.astype(np.float32))
        f.attrs["runtime"] = 0.458  # kept identical to the original hard-coded value

    return Yq


if __name__ == "__main__":
    save_test_inputs_to_csv(scenario="B")
    compute_baseline_predictions(scenario="A")