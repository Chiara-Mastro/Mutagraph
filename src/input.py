from pathlib import Path
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT_FROM_SCRIPT))

from src.CONFIG import (
    STATUS_OBSERVED,
    STATUS_IMPUTE,
    STATUS_INTRINSIC_RESISTANCE,
    REQUIRED_COLUMNS,
    KEY_COLUMNS
)

def load_input_table(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    df = pd.read_csv(input_path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["Species", "Family", "Country"]:
        df[col] = df[col].astype("string").str.strip()

    df["Year"] = df["Year"].astype(int)

    status_raw = df["status"].copy()

    valid_statuses = {
        STATUS_OBSERVED,
        STATUS_IMPUTE,
        STATUS_INTRINSIC_RESISTANCE,
    }

    bad_status = ~df["status"].isin(valid_statuses)

    if bad_status.any():
        raise ValueError(
            "Unknown status values found: "
            f"{status_raw[bad_status].dropna().unique().tolist()} with status {df['status']}"
        )

    df["n_S"] = pd.to_numeric(df["n_S"], errors="coerce")
    df["n_total"] = pd.to_numeric(df["n_total"], errors="coerce")

    return df


def apply_year_filter(
    df: pd.DataFrame,
    min_year: int | None,
    max_year: int | None,
) -> pd.DataFrame:
    df = df.copy()

    if min_year is not None:
        df = df[df["Year"] >= min_year].copy()

    if max_year is not None:
        df = df[df["Year"] <= max_year].copy()

    return df


def aggregate_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates duplicated Species-Family-Country-Year rows if present.

    Semantics:
    - OBSERVED rows are summed: n_S and n_total are empirical counts.
    - INTRINSIC_RESISTANCE dominates IMPUTE if no observed row exists.
    - IMPUTE remains IMPUTE only when no observed or intrinsic row exists.

    If a key has both OBSERVED and INTRINSIC_RESISTANCE, this is inconsistent.
    """
    rows = []

    for key, g in df.groupby(KEY_COLUMNS, dropna=False):
        statuses = set(g["status"].dropna().unique())

        has_observed = STATUS_OBSERVED in statuses
        has_intrinsic = STATUS_INTRINSIC_RESISTANCE in statuses
        has_impute = STATUS_IMPUTE in statuses

        if has_observed and has_intrinsic:
            raise ValueError(
                f"Cell {key} has both OBSERVED and INTRINSIC_RESISTANCE status."
            )

        row = dict(zip(KEY_COLUMNS, key))

        if has_observed:
            obs = g[g["status"] == STATUS_OBSERVED]

            if obs[["n_S", "n_total"]].isna().any(axis=None):
                raise ValueError(f"Observed cell {key} has missing counts.")

            n_S = obs["n_S"].sum()
            n_total = obs["n_total"].sum()

            if n_S < 0 or n_total <= 0 or n_S > n_total:
                raise ValueError(
                    f"Invalid counts for observed cell {key}: "
                    f"n_S={n_S}, n_total={n_total}"
                )

            row.update({
                "status": STATUS_OBSERVED,
                "n_S": n_S,
                "n_total": n_total,
            })

        elif has_intrinsic:
            row.update({
                "status": STATUS_INTRINSIC_RESISTANCE,
                "n_S": np.nan,
                "n_total": np.nan,
            })

        elif has_impute:
            row.update({
                "status": STATUS_IMPUTE,
                "n_S": np.nan,
                "n_total": np.nan,
            })

        else:
            raise ValueError(f"Cell {key} has no valid status.")

        rows.append(row)

    return pd.DataFrame(rows)


def add_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    observed = df["status"].eq(STATUS_OBSERVED)
    impute = df["status"].eq(STATUS_IMPUTE)
    intrinsic = df["status"].eq(STATUS_INTRINSIC_RESISTANCE)

    df["is_observed"] = observed.astype(int)
    df["is_impute"] = impute.astype(int)
    df["is_intrinsic_resistance"] = intrinsic.astype(int)

    df["prop_S"] = np.nan
    df.loc[observed, "prop_S"] = (
        df.loc[observed, "n_S"] / df.loc[observed, "n_total"]
    )

    df["structural_prop_S"] = np.nan
    df.loc[intrinsic, "structural_prop_S"] = 0.0

    return df


def final_qc(df: pd.DataFrame) -> None:
    duplicated = df.duplicated(KEY_COLUMNS).sum()
    if duplicated > 0:
        raise ValueError(f"Output still has duplicated keys: {duplicated}")

    observed = df["status"].eq(STATUS_OBSERVED)

    bad_counts = (
        observed
        & (
            df["n_S"].isna()
            | df["n_total"].isna()
            | (df["n_S"] < 0)
            | (df["n_total"] <= 0)
            | (df["n_S"] > df["n_total"])
        )
    )

    if bad_counts.any():
        raise ValueError(f"Invalid observed counts in {bad_counts.sum()} rows.")

    non_observed = ~observed
    df.loc[non_observed, ["n_S", "n_total"]]

    bad_prop = (
        observed
        & (
            df["prop_S"].isna()
            | (df["prop_S"] < 0)
            | (df["prop_S"] > 1)
        )
    )

    if bad_prop.any():
        raise ValueError(f"Invalid prop_S in {bad_prop.sum()} rows.")

