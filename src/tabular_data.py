from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Tuple

import pandas as pd


TaskName = Literal["full", "aoa", "reynolds", "scarce"]


def _manifest_keys(task: TaskName) -> Tuple[str, str]:
    if task == "full":
        return "full_train", "full_test"
    if task == "aoa":
        return "aoa_train", "aoa_test"
    if task == "reynolds":
        return "reynolds_train", "reynolds_test"
    if task == "scarce":
        return "scarce_train", "scarce_test"
    raise ValueError(f"Unknown task: {task}")


def load_airfrans_tabular_split(
    csv_path: Path,
    manifest_path: Path,
    task: TaskName = "full",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the tabular CSV and split rows according to AirfRANS manifest."""

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    df = pd.read_csv(csv_path)
    if "name" not in df.columns:
        raise ValueError("CSV must contain a 'name' column")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    train_key, test_key = _manifest_keys(task)

    if train_key not in manifest:
        raise KeyError(f"manifest missing key: {train_key}")

    # Some AirfRANS releases don't include scarce_test; fall back to full_test.
    if test_key not in manifest:
        if task == "scarce" and "full_test" in manifest:
            test_key = "full_test"
        else:
            raise KeyError(f"manifest missing key: {test_key}")

    train_names = set(manifest[train_key])
    test_names = set(manifest[test_key])

    train_df = df[df["name"].isin(train_names)].copy()
    test_df = df[df["name"].isin(test_names)].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            f"Split produced empty set(s). train={len(train_df)} test={len(test_df)} "
            f"(task={task}, keys={train_key}/{test_key})"
        )

    return train_df, test_df
