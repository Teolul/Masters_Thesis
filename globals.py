from pathlib import Path

# root
ROOT = Path("../AMLEC-1/rtm_emulation")

# Scenario A
TRAIN_FILE_A_500 = ROOT / "scenarioA" / "train" / "train500.h5"
TRAIN_FILE_A_2000 = ROOT / "scenarioA" / "train" / "train2000.h5"
TRAIN_FILE_A_10000 = ROOT / "scenarioA" / "train" / "train10000.h5"
TEST_FILE_A_INTERP = ROOT / "scenarioA" / "reference" / "refInterp.csv"
TEST_FILE_A_EXTRAP = ROOT / "scenarioA" / "reference" / "refExtrap.csv"

# Scenario B
TRAIN_FILE_B_500 = ROOT / "scenarioB" / "train" / "train500.h5"
TRAIN_FILE_B_2000 = ROOT / "scenarioB" / "train" / "train2000.h5"
TEST_FILE_B_INTERP = ROOT / "scenarioB" / "reference" / "refInterp.csv"
TEST_FILE_B_EXTRAP = ROOT / "scenarioB" / "reference" / "refExtrap.csv"

# "current" selection (mutable state)
CURRENT_TRAIN_FILE = TRAIN_FILE_A_10000
CURRENT_TEST_FILE = TEST_FILE_A_INTERP

N_FUNCTIONS = 6
N_INPUTS = 9