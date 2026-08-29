import h5py
import numpy as np


# ============================================================
# 1. Configuration
# ============================================================

file_path = "data/FeatureEntire.mat"


# ============================================================
# 2. Helper function for MATLAB char arrays
# ============================================================

def decode_char(dataset):

    values = dataset[()].flatten()

    return "".join(
        chr(int(value))
        for value in values
        if value != 0
    )


# ============================================================
# 3. Helper function for displaying an HDF5 object
# ============================================================

def inspect_object(file, obj, indent=""):

    print(f"{indent}Name: {obj.name}")
    print(f"{indent}Type: {type(obj).__name__}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    if isinstance(obj, h5py.Dataset):

        print(f"{indent}Shape: {obj.shape}")
        print(f"{indent}Dtype: {obj.dtype}")

        matlab_class = obj.attrs.get(
            "MATLAB_class"
        )

        print(
            f"{indent}MATLAB class: "
            f"{matlab_class}"
        )

        # MATLAB character array
        if matlab_class == b"char":

            try:
                text = decode_char(obj)

                print(
                    f"{indent}Decoded text: "
                    f"{text}"
                )

            except Exception as error:

                print(
                    f"{indent}Could not decode char: "
                    f"{error}"
                )

        # Numeric dataset
        elif obj.dtype != object:

            try:

                values = (
                    obj[()]
                    .flatten()
                )

                print(
                    f"{indent}First values:"
                )

                print(
                    values[:20]
                )

            except Exception as error:

                print(
                    f"{indent}Could not read values: "
                    f"{error}"
                )

        # Reference/cell dataset
        else:

            print(
                f"{indent}Reference/cell dataset"
            )

            try:

                references = (
                    obj[()]
                    .flatten()
                )

                for index, ref in enumerate(
                    references[:20]
                ):

                    print(
                        f"{indent}  Reference "
                        f"{index}:"
                    )

                    try:

                        referenced_object = file[ref]

                        print(
                            f"{indent}    -> "
                            f"{referenced_object.name}"
                        )

                    except Exception as error:

                        print(
                            f"{indent}    Could not "
                            f"resolve reference: {error}"
                        )

            except Exception as error:

                print(
                    f"{indent}Could not inspect "
                    f"references: {error}"
                )


    # --------------------------------------------------------
    # Group
    # --------------------------------------------------------

    elif isinstance(obj, h5py.Group):

        print(
            f"{indent}Members:"
        )

        print(
            f"{indent}{list(obj.keys())}"
        )

        matlab_class = obj.attrs.get(
            "MATLAB_class"
        )

        print(
            f"{indent}MATLAB class: "
            f"{matlab_class}"
        )


# ============================================================
# 4. Open FeatureEntire.mat
# ============================================================

with h5py.File(file_path, "r") as file:

    refs = file["#refs#"]


    # ========================================================
    # 5. Inspect selected objects
    # ========================================================

    objects_to_inspect = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "M"
    ]


    print("=" * 80)
    print("SELECTED MATLAB OBJECTS")
    print("=" * 80)


    for object_name in objects_to_inspect:

        print("\n" + "=" * 80)
        print(
            f"OBJECT: {object_name}"
        )
        print("=" * 80)

        obj = refs[object_name]

        inspect_object(
            file,
            obj
        )


    # ========================================================
    # 6. Inspect members of object M
    # ========================================================

    print("\n" + "=" * 80)
    print("OBJECT M MEMBERS")
    print("=" * 80)


    M = refs["M"]


    for member_name in M.keys():

        print("\n" + "-" * 80)

        print(
            f"M MEMBER: {member_name}"
        )

        print("-" * 80)

        member = M[member_name]

        inspect_object(
            file,
            member,
            indent="  "
        )


    # ========================================================
    # 7. Inspect categorical label structure
    # ========================================================

    print("\n" + "=" * 80)
    print("CATEGORICAL LABEL STRUCTURE")
    print("=" * 80)


    categorical = refs["6"]


    for member_name in categorical.keys():

        print("\n" + "-" * 80)

        print(
            f"6 MEMBER: {member_name}"
        )

        print("-" * 80)

        member = categorical[member_name]

        inspect_object(
            file,
            member,
            indent="  "
        )