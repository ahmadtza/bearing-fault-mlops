import h5py
import numpy as np


# ============================================================
# 1. Configuration
# ============================================================

file_path = "data/MachineData_export.mat"


# ============================================================
# 2. Open MATLAB v7.3 file
# ============================================================

print("=" * 80)
print("LOADING RAW MACHINE DATA")
print("=" * 80)


with h5py.File(file_path, "r") as file:

    print("\nVariables inside MATLAB file:")

    for key in file.keys():
        print(f" - {key}")


    # ========================================================
    # 3. Load variables
    #
    # MATLAB v7.3 / HDF5 orientation may appear transposed
    # ========================================================

    ch1 = np.array(file["ch1"])
    ch2 = np.array(file["ch2"])
    ch3 = np.array(file["ch3"])

    run_id = np.array(
        file["run_id"]
    ).flatten()

    label_code = np.array(
        file["label_code"]
    ).flatten()


# ============================================================
# 4. Display raw HDF5 shapes
# ============================================================

print("\n" + "=" * 80)
print("RAW HDF5 SHAPES")
print("=" * 80)

print("ch1:", ch1.shape)
print("ch2:", ch2.shape)
print("ch3:", ch3.shape)

print("run_id:", run_id.shape)
print("label_code:", label_code.shape)


# ============================================================
# 5. Correct MATLAB/HDF5 orientation if necessary
# ============================================================

# We want:
#
# rows    = recordings = 40
# columns = signal samples = 70000
#
# Final desired shape:
#
# (40, 70000)

if ch1.shape == (70000, 40):

    ch1 = ch1.T
    ch2 = ch2.T
    ch3 = ch3.T


# ============================================================
# 6. Verify final shapes
# ============================================================

print("\n" + "=" * 80)
print("FINAL PYTHON SHAPES")
print("=" * 80)

print("ch1:", ch1.shape)
print("ch2:", ch2.shape)
print("ch3:", ch3.shape)


# ============================================================
# 7. Dataset information
# ============================================================

print("\n" + "=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print(
    f"Number of recordings : {ch1.shape[0]}"
)

print(
    f"Samples per recording: {ch1.shape[1]}"
)

print(
    f"Number of channels   : 3"
)


# ============================================================
# 8. Run IDs
# ============================================================

print("\nRun IDs:")

print(
    run_id.astype(int)
)


# ============================================================
# 9. Labels
# ============================================================

print("\nLabel codes:")

print(
    label_code.astype(int)
)


# ============================================================
# 10. Class distribution
# ============================================================

unique_labels, counts = np.unique(
    label_code.astype(int),
    return_counts=True
)


print("\nClass distribution:")

for label, count in zip(
    unique_labels,
    counts
):

    if label == 0:
        condition = "After (Normal)"
    else:
        condition = "Before (Anomalous)"

    print(
        f"{label} = {condition}: {count}"
    )


# ============================================================
# 11. Check NaN and Inf
# ============================================================

print("\n" + "=" * 80)
print("DATA QUALITY")
print("=" * 80)


for name, signal in [
    ("ch1", ch1),
    ("ch2", ch2),
    ("ch3", ch3)
]:

    print(f"\n{name}")

    print(
        "NaN:",
        np.isnan(signal).sum()
    )

    print(
        "Inf:",
        np.isinf(signal).sum()
    )

    print(
        "Minimum:",
        signal.min()
    )

    print(
        "Maximum:",
        signal.max()
    )

    print(
        "Mean:",
        signal.mean()
    )

    print(
        "Std:",
        signal.std()
    )


# ============================================================
# 12. Inspect several individual recordings
# ============================================================

print("\n" + "=" * 80)
print("EXAMPLE RECORDINGS")
print("=" * 80)


example_runs = [
    0,      # Run 1  - Before
    19,     # Run 20 - Before
    20,     # Run 21 - After
    39      # Run 40 - After
]


for index in example_runs:

    print(
        f"\nRun {int(run_id[index])}"
    )

    if label_code[index] == 0:
        condition = "After (Normal)"
    else:
        condition = "Before (Anomalous)"

    print(
        f"Condition: {condition}"
    )

    print(
        f"ch1 first 5 samples: "
        f"{ch1[index, :5]}"
    )

    print(
        f"ch2 first 5 samples: "
        f"{ch2[index, :5]}"
    )

    print(
        f"ch3 first 5 samples: "
        f"{ch3[index, :5]}"
    )


# ============================================================
# 13. Final validation
# ============================================================

assert ch1.shape == (40, 70000)
assert ch2.shape == (40, 70000)
assert ch3.shape == (40, 70000)

assert len(run_id) == 40
assert len(label_code) == 40

assert np.sum(label_code == 1) == 20
assert np.sum(label_code == 0) == 20


print("\n" + "=" * 80)
print("RAW DATA VALIDATION PASSED")
print("=" * 80)

print(
    "40 independent recordings were loaded successfully."
)