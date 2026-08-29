import os
import scipy.io
import numpy as np


# ============================================================
# 1. Configuration
# ============================================================

file_path = "data/MachineData.mat"


# ============================================================
# 2. Basic file information
# ============================================================

print("=" * 80)
print("MACHINEDATA.MAT - BASIC INFORMATION")
print("=" * 80)

print(f"\nFile: {file_path}")

print(
    f"File size: "
    f"{os.path.getsize(file_path) / (1024 ** 2):.2f} MB"
)


# ============================================================
# 3. Read MATLAB variable directory
#
# This does NOT fully load the 62 MB file.
# ============================================================

print("\n" + "=" * 80)
print("MATLAB VARIABLE DIRECTORY")
print("=" * 80)

try:

    variables = scipy.io.whosmat(
        file_path
    )

    for variable in variables:

        name, shape, matlab_class = variable

        print(
            f"\nName         : {name}"
        )

        print(
            f"Shape        : {shape}"
        )

        print(
            f"MATLAB class : {matlab_class}"
        )

except Exception as error:

    print(
        "\nwhosmat error:"
    )

    print(error)


# ============================================================
# 4. Try loading without squeezing
# ============================================================

print("\n" + "=" * 80)
print("LOADMAT - RAW MODE")
print("=" * 80)

try:

    raw_data = scipy.io.loadmat(
        file_path,
        struct_as_record=True,
        squeeze_me=False
    )

    print("\nTop-level keys:")

    for key in raw_data.keys():

        if key.startswith("__"):
            continue

        obj = raw_data[key]

        print("\n" + "-" * 70)

        print(
            f"Variable: {key}"
        )

        print(
            f"Python type: {type(obj)}"
        )

        if hasattr(obj, "shape"):

            print(
                f"Shape: {obj.shape}"
            )

        if hasattr(obj, "dtype"):

            print(
                f"Dtype: {obj.dtype}"
            )


except Exception as error:

    print(
        "\nRaw load error:"
    )

    print(error)


# ============================================================
# 5. Try simplified-cell loading
# ============================================================

print("\n" + "=" * 80)
print("LOADMAT - SIMPLIFIED MODE")
print("=" * 80)

try:

    simple_data = scipy.io.loadmat(
        file_path,
        simplify_cells=True
    )

    print("\nTop-level keys:")

    for key, value in simple_data.items():

        if key.startswith("__"):
            continue

        print("\n" + "-" * 70)

        print(
            f"Variable: {key}"
        )

        print(
            f"Python type: {type(value)}"
        )

        if hasattr(value, "shape"):

            print(
                f"Shape: {value.shape}"
            )

        if hasattr(value, "dtype"):

            print(
                f"Dtype: {value.dtype}"
            )

        if isinstance(value, dict):

            print(
                "Dictionary keys:"
            )

            print(
                list(value.keys())
            )


except Exception as error:

    print(
        "\nSimplified load error:"
    )

    print(error)


# ============================================================
# 6. Search raw binary file for useful text
# ============================================================

print("\n" + "=" * 80)
print("SEARCHING BINARY FILE FOR METADATA KEYWORDS")
print("=" * 80)


keywords = [
    b"trainData",
    b"label",
    b"Before",
    b"After",
    b"filename",
    b"file",
    b"FileName",
    b"run",
    b"Run",
    b"recording",
    b"Recording",
    b"timestamp",
    b"Timestamp",
    b"time",
    b"Time",
    b"member",
    b"Member",
    b"ensemble",
    b"Ensemble",
    b"condition",
    b"Condition"
]


with open(
    file_path,
    "rb"
) as file:

    binary_content = file.read()


for keyword in keywords:

    count = binary_content.count(
        keyword
    )

    if count > 0:

        print(
            f"{keyword.decode(errors='ignore'):15s}"
            f" : {count}"
        )


# ============================================================
# 7. MATLAB header
# ============================================================

print("\n" + "=" * 80)
print("MATLAB FILE HEADER")
print("=" * 80)


with open(
    file_path,
    "rb"
) as file:

    header = file.read(128)


print(
    header.decode(
        errors="ignore"
    )
)


print("\n" + "=" * 80)
print("INSPECTION COMPLETED")
print("=" * 80)