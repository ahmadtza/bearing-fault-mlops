from pathlib import Path

import h5py
import numpy as np


SOURCE_FILE = Path("data/MachineData_export.mat")
OUTPUT_FILE = Path(
    "tests/fixtures/bearing_acceptance_fixture.npz"
)

SELECTED_RUNS = [1, 25]


def load_selected_runs():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source file not found: {SOURCE_FILE}"
        )

    with h5py.File(SOURCE_FILE, "r") as f:
        run_ids = (
            np.asarray(f["run_id"])
            .flatten()
            .astype(np.int32)
        )

        label_codes = (
            np.asarray(f["label_code"])
            .flatten()
            .astype(np.int32)
        )

        fixture = {}

        for run_id in SELECTED_RUNS:
            matches = np.where(
                run_ids == run_id
            )[0]

            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one Run {run_id}, "
                    f"found {len(matches)}."
                )

            index = int(matches[0])
            label_code = int(
                label_codes[index]
            )

            ch1 = np.asarray(
                f["ch1"][index],
                dtype=np.float32,
            )

            ch2 = np.asarray(
                f["ch2"][index],
                dtype=np.float32,
            )

            ch3 = np.asarray(
                f["ch3"][index],
                dtype=np.float32,
            )

            if not (
                ch1.shape
                == ch2.shape
                == ch3.shape
            ):
                raise RuntimeError(
                    f"Channel shape mismatch "
                    f"for Run {run_id}."
                )

            if not (
                np.isfinite(ch1).all()
                and np.isfinite(ch2).all()
                and np.isfinite(ch3).all()
            ):
                raise RuntimeError(
                    f"NaN or Inf found "
                    f"in Run {run_id}."
                )

            fixture[
                f"run_{run_id}_ch1"
            ] = ch1

            fixture[
                f"run_{run_id}_ch2"
            ] = ch2

            fixture[
                f"run_{run_id}_ch3"
            ] = ch3

            fixture[
                f"run_{run_id}_label"
            ] = np.asarray(
                [label_code],
                dtype=np.int32,
            )

            print(
                f"Run {run_id}: "
                f"index={index}, "
                f"label={label_code}, "
                f"samples={len(ch1)}"
            )

    return fixture


def main():
    print("=" * 70)
    print("CREATE CI ACCEPTANCE FIXTURE")
    print("=" * 70)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fixture = load_selected_runs()

    np.savez_compressed(
        OUTPUT_FILE,
        **fixture,
    )

    size_mb = (
        OUTPUT_FILE.stat().st_size
        / 1024
        / 1024
    )

    print()
    print(f"Created : {OUTPUT_FILE}")
    print(f"Size    : {size_mb:.2f} MB")
    print()
    print("Fixture creation: PASS")


if __name__ == "__main__":
    main()
