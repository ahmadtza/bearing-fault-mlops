from scipy.io import loadmat
import numpy as np


file_path = "data/MachineData.mat"


print("=" * 80)
print("LOADING MachineData.mat")
print("=" * 80)


# ============================================================
# 1. Load MATLAB file
# ============================================================

mat_data = loadmat(
    file_path,
    struct_as_record=False,
    squeeze_me=True
)


# ============================================================
# 2. Show top-level variables
# ============================================================

print("\nTOP-LEVEL VARIABLES")
print("-" * 80)

for key in mat_data.keys():

    # MATLAB internal variables start with __
    if not key.startswith("__"):

        value = mat_data[key]

        print(f"\nVariable: {key}")
        print(f"Python type: {type(value)}")

        if isinstance(value, np.ndarray):

            print(f"Shape: {value.shape}")
            print(f"Dtype: {value.dtype}")

        else:

            try:
                print(f"Shape: {value.shape}")
            except AttributeError:
                pass


# ============================================================
# 3. Detailed inspection
# ============================================================

print("\n" + "=" * 80)
print("DETAILED CONTENT")
print("=" * 80)


for key, value in mat_data.items():

    if key.startswith("__"):
        continue

    print(f"\n{'-' * 80}")
    print(f"VARIABLE: {key}")
    print(f"{'-' * 80}")

    print("Type:")
    print(type(value))

    if isinstance(value, np.ndarray):

        print("\nShape:")
        print(value.shape)

        print("\nDtype:")
        print(value.dtype)

        print("\nFirst few values:")

        try:
            print(value.flatten()[:10])
        except Exception as error:
            print("Could not display values:", error)

    else:

        print("\nContent:")
        print(value)