import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT_FROM_SCRIPT))

from src.CONFIG import STATUS_OBSERVED

def load_observed_dataset(path: Path) -> pd.DataFrame:
    """
    Load standardized AMR dataset and keep only observed cells.

    The model is trained/evaluated only on observed rows because these are the
    only rows with real labels n_S and n_total. The true to_impute rows are not
    used in this script yet.
    """
    df = pd.read_csv(path)

    required = [
        "Species",
        "Family",
        "Country",
        "Year",
        "status",
        "n_S",
        "n_total",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    for col in ["Species", "Family", "Country", "status"]:
        df[col] = df[col].astype("string").str.strip()

    df["status"] = df["status"].str.lower()
    df["Year"] = df["Year"].astype(int)

    df["n_S"] = pd.to_numeric(df["n_S"], errors="coerce")
    df["n_total"] = pd.to_numeric(df["n_total"], errors="coerce")

    observed_df = df[df["status"].eq(STATUS_OBSERVED)].copy()

    if observed_df.empty:
        raise ValueError(
            f"No observed rows found. Check STATUS_OBSERVED={STATUS_OBSERVED!r} "
            "against dataset status values."
        )

    observed_df = observed_df.dropna(subset=["n_S", "n_total"]).copy()

    observed_df = observed_df[
        (observed_df["n_total"] > 0)
        & (observed_df["n_S"] >= 0)
        & (observed_df["n_S"] <= observed_df["n_total"])
    ].copy()

    observed_df["prop_S"] = observed_df["n_S"] / observed_df["n_total"]

    # Snapshot identifier: this is the unit that receives a latent embedding z.
    observed_df["snapshot_id"] = (
        observed_df["Country"].astype(str)
        + "||"
        + observed_df["Year"].astype(str)
    )

    observed_df = observed_df.reset_index(drop=True)

    return observed_df

def assign_cell_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.20,
    test_frac: float = 0.10,
    min_cells_per_snapshot: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Split observed cells within each Country-Year snapshot.

    For snapshots with enough observed cells, rows are randomly assigned to:
        train / val / test

    For very small snapshots, all cells are kept in train, because a 70/20/10
    split would produce meaningless validation/test sets.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.")

    rng = np.random.default_rng(seed)

    df = df.copy()
    df["split"] = "train"

    for snapshot_id, idx in df.groupby("snapshot_id").groups.items():
        idx = np.array(list(idx))
        n = len(idx)

        if n < min_cells_per_snapshot:
            df.loc[idx, "split"] = "train"
            continue

        shuffled_idx = idx.copy()
        rng.shuffle(shuffled_idx)

        n_train = int(np.floor(train_frac * n))
        n_val = int(np.floor(val_frac * n))

        train_idx = shuffled_idx[:n_train]
        val_idx = shuffled_idx[n_train:n_train + n_val]
        test_idx = shuffled_idx[n_train + n_val:]

        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"

    return df

def build_index_mapping(values):
    unique_values = sorted(pd.Series(values).dropna().unique().tolist())
    return {value: i for i, value in enumerate(unique_values)}

def add_integer_indices(df: pd.DataFrame):
    """
    Add integer IDs for snapshot, species, and family.

    These IDs are used by PyTorch embedding layers.
    """
    df = df.copy()

    snapshot_to_idx = build_index_mapping(df["snapshot_id"])
    species_to_idx = build_index_mapping(df["Species"])
    family_to_idx = build_index_mapping(df["Family"])

    df["snapshot_idx"] = df["snapshot_id"].map(snapshot_to_idx).astype(int)
    df["species_idx"] = df["Species"].map(species_to_idx).astype(int)
    df["family_idx"] = df["Family"].map(family_to_idx).astype(int)

    mappings = {
        "snapshot_to_idx": snapshot_to_idx,
        "species_to_idx": species_to_idx,
        "family_to_idx": family_to_idx,
    }

    sizes = {
        "n_snapshots": len(snapshot_to_idx),
        "n_species": len(species_to_idx),
        "n_families": len(family_to_idx),
    }

    return df, mappings, sizes

def load_predictions(path: Path, prediction_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [
        "Species",
        "Family",
        "Country",
        "Year",
        "status",
        "n_S",
        "n_total",
        "prop_S",
        prediction_col,
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    for col in ["Species", "Family", "Country", "status"]:
        df[col] = df[col].astype("string").str.strip()

    df["Year"] = df["Year"].astype(int)
    df["n_S"] = pd.to_numeric(df["n_S"], errors="coerce")
    df["n_total"] = pd.to_numeric(df["n_total"], errors="coerce")
    df["prop_S"] = pd.to_numeric(df["prop_S"], errors="coerce")
    df[prediction_col] = pd.to_numeric(df[prediction_col], errors="coerce")

    if df[prediction_col].isna().any():
        n_missing = df[prediction_col].isna().sum()
        raise ValueError(f"{prediction_col} contains {n_missing} missing predictions.")

    return df

def add_residual_columns(df: pd.DataFrame, prediction_col: str) -> pd.DataFrame:
    df = df.copy()

    # Positive residual: observed susceptibility is higher than predicted.
    # Negative residual: observed susceptibility is lower than predicted.
    df["residual"] = df["prop_S"] - df[prediction_col]
    df["abs_error"] = df["residual"].abs()
    df["squared_error"] = df["residual"] ** 2

    # Count-level residual: observed susceptible count minus expected susceptible count.
    # This is useful to identify groups where the absolute number of mispredicted
    # susceptible isolates is large, not only the proportion error.
    df["expected_n_S"] = df[prediction_col] * df["n_total"]
    df["count_residual"] = df["n_S"] - df["expected_n_S"]
    df["abs_count_residual"] = df["count_residual"].abs()

    return df

def summarize_by_group(df: pd.DataFrame, group_cols, min_cells: int) -> pd.DataFrame:
    summary = (
        df.groupby(group_cols)
        .agg(
            n_cells=("residual", "size"),
            n_tests=("n_total", "sum"),
            mean_observed_prop_S=("prop_S", "mean"),
            mean_predicted_prop_S=("p_hierarchical", "mean"),
            mean_residual=("residual", "mean"),
            median_residual=("residual", "median"),
            mean_abs_error=("abs_error", "mean"),
            weighted_abs_error=(
                "abs_error",
                lambda x: np.average(x, weights=df.loc[x.index, "n_total"]),
            ),
            rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
            weighted_rmse=(
                "squared_error",
                lambda x: np.sqrt(np.average(x, weights=df.loc[x.index, "n_total"])),
            ),
            total_count_residual=("count_residual", "sum"),
            total_abs_count_residual=("abs_count_residual", "sum"),
        )
        .reset_index()
    )

    summary = summary[summary["n_cells"] >= min_cells].copy()

    # Useful ranking column: large weighted error and many tests.
    summary["error_burden"] = (
        summary["weighted_abs_error"] * summary["n_tests"]
    )

    return summary.sort_values("error_burden", ascending=False)

def summarize_all_levels(df: pd.DataFrame, output_dir: Path, min_cells: int):
    groupings = {
        "country": ["Country"],
        "species": ["Species"],
        "family": ["Family"],
        "year": ["Year"],
        "country_year": ["Country", "Year"],
        "species_family": ["Species", "Family"],
        "country_species_family": ["Country", "Species", "Family"],
    }

    for name, group_cols in groupings.items():
        summary = summarize_by_group(
            df=df,
            group_cols=group_cols,
            min_cells=min_cells,
        )

        output_path = output_dir / f"baseline_residuals_by_{name}.csv"
        summary.to_csv(output_path, index=False)

        print(f"Saved {output_path}")

def save_top_errors(df: pd.DataFrame, output_dir: Path):
    cols = [
        "Species",
        "Family",
        "Country",
        "Year",
        "status",
        "n_S",
        "n_total",
        "prop_S",
        "p_hierarchical",
        "residual",
        "abs_error",
        "count_residual",
        "abs_count_residual",
    ]

    top_abs_error = df.sort_values("abs_error", ascending=False).head(200)
    top_count_error = df.sort_values("abs_count_residual", ascending=False).head(200)

    top_abs_error[cols].to_csv(
        output_dir / "baseline_top_200_absolute_prop_errors.csv",
        index=False,
    )

    top_count_error[cols].to_csv(
        output_dir / "baseline_top_200_absolute_count_errors.csv",
        index=False,
    )
