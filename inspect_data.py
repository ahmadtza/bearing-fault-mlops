import h5py


file_path = "data/FeatureEntire.mat"


def decode_matlab_char(dataset):
    """
    Convert a MATLAB char dataset stored as uint16
    to a normal Python string.
    """
    values = dataset[()].flatten()

    return "".join(chr(int(x)) for x in values if x != 0)


with h5py.File(file_path, "r") as file:

    refs = file["#refs#"]

    print("=" * 70)
    print("DECODING OBJECT g")
    print("=" * 70)

    g = refs["g"]

    print("Shape:", g.shape)
    print()

    for index, ref in enumerate(g[:, 0]):

        obj = file[ref]

        print(f"g[{index}]")
        print("Object name:", obj.name)
        print("Shape:", obj.shape)
        print("MATLAB class:", obj.attrs.get("MATLAB_class"))

        if obj.attrs.get("MATLAB_class") == b"char":
            text = decode_matlab_char(obj)
            print("Decoded text:", text)

        print("-" * 50)

    print()
    print("=" * 70)
    print("DECODING OBJECT y")
    print("=" * 70)

    y = refs["y"]

    print("Shape:", y.shape)
    print()

    for index, ref in enumerate(y[:, 0]):

        obj = file[ref]

        print(f"y[{index}]")
        print("Object name:", obj.name)
        print("Shape:", obj.shape)
        print("MATLAB class:", obj.attrs.get("MATLAB_class"))

        if obj.attrs.get("MATLAB_class") == b"char":
            text = decode_matlab_char(obj)
            print("Decoded text:", text)

        print("-" * 50)