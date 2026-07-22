"""
Python conversion of checkFileValidity.m

Validates that a participant's submission file (an HDF5 look-up-table file)
is well-formed and comparable to a reference file before it gets scored.
"""

import os
import re
import h5py


def check_file_validity(target_file: str, reference_file: str) -> bool:
    """
    Check that `target_file` is a valid submission, comparable to `reference_file`.

    A file is considered valid if:
      1. It exists.
      2. It has a .h5 / .hdf5 extension.
      3. Its filename (without extension) matches "<model>_<S><track>",
         e.g. "myModel_A1", "myModel_B2" (S is 'A' or 'B', track is '1' or '2').
      4. It contains a dataset called "/LUTdata".
      5. "/LUTdata" is a 2-D array.
      6. The reference file exists, also contains a 2-D "/LUTdata" dataset,
         and its second dimension matches the target file's second dimension.

    Returns
    -------
    bool
        True if every check passes, False otherwise (a message is printed
        explaining the first failure encountered).
    """

    # 1. Target file must exist
    if not os.path.isfile(target_file):
        print("Target file does not exist.")
        return False

    # 2. Extension must be .h5 or .hdf5
    target_name, target_ext = os.path.splitext(os.path.basename(target_file))
    target_ext = target_ext.lower()
    if target_ext not in (".h5", ".hdf5"):
        print("Target file is not a .h5 or .hdf5 file.")
        return False

    # 3. Filename must match "<model>_<S><track>" (S = A or B, track = 1 or 2)
    name_pattern = r"^[a-zA-Z0-9]+_(A|B)[12]$"
    if re.match(name_pattern, target_name) is None:
        print("Target file name is not in the required format <model>_<Sx>.")
        return False

    # 4-5. "/LUTdata" must exist in target file and be 2-D
    try:
        with h5py.File(target_file, "r") as f:
            target_lutdata = f["/LUTdata"][()]
    except (OSError, KeyError):
        print("LUTdata variable not found in the target file.")
        return False

    if target_lutdata.ndim != 2:
        print("LUTdata in the target file is not a 2D matrix.")
        return False

    # 6a. Reference file must exist
    if not os.path.isfile(reference_file):
        print("Reference file does not exist.")
        return False

    # 6b. "/LUTdata" must exist in reference file and be 2-D
    try:
        with h5py.File(reference_file, "r") as f:
            reference_lutdata = f["/LUTdata"][()]
    except (OSError, KeyError):
        print("LUTdata variable not found in the reference file.")
        return False

    if reference_lutdata.ndim != 2:
        print("LUTdata in the reference file is not a 2D matrix.")
        return False

    # 6c. Second dimension must match between target and reference
    if target_lutdata.shape[1] != reference_lutdata.shape[1]:
        print("Second dimension of LUTdata in target file does not match reference file.")
        return False

    # All checks passed
    print("Target file is valid.")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python check_file_validity.py <target_file.h5> <reference_file.h5>")
        sys.exit(1)

    is_valid = check_file_validity(sys.argv[1], sys.argv[2])
    sys.exit(0 if is_valid else 1)