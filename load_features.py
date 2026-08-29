import h5py
import numpy as np
import pandas as pd


file_path = "data/FeatureEntire.mat"


# -------------------------------------------------
# Column names discovered from MATLAB table
# -------------------------------------------------

feature_names = [
    "ch1_CrestFactor",
    "ch1_Kurtosis",
    "ch1_RMS",
    "ch1_Std",
    "ch2_Mean",
    "ch2_RMS",
    "ch2_Skewness",
    "ch2_Std",
    "ch3_CrestFactor",
    "ch3_SINAD",
    "ch3_SNR",
    "ch3_THD"
]


# MATLAB HDF5 objects containing the 12 features
feature_objects = [
    "i", "j", "k", "l",
    "m", "n", "o", "p",
    "q", "r", "s", "t"
]


with h5py.File(file_path, "r") as file:

    refs = file["#refs#"]

    # ---------------------------------------------
    # 1. Read numerical features
    # ---------------------------------------------

    data = {}

    for name, obj_name in zip(feature_names, feature_objects):

        values = refs[obj_name][()].flatten()

        data[name] = values


    # ---------------------------------------------
    # 2. Read categorical labels
    # ---------------------------------------------

    label_codes = refs["f"][()].flatten()

    label_map = {
        1: "Before",
        2: "After"
    }

    labels = [
        label_map[int(code)]
        for code in label_codes
    ]


    # ---------------------------------------------
    # 3. Create Pandas DataFrame
    # ---------------------------------------------

    df = pd.DataFrame(data)

    # Put label as the first column
    df.insert(0, "label", labels)


# -------------------------------------------------
# Basic checks
# -------------------------------------------------

print("=" * 70)
print("DATAFRAME CREATED SUCCESSFULLY")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic numerical statistics:")
print(df.describe())