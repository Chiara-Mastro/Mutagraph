# FUNCTIONS FOR THE HISTORICAL BASELINE MODELS
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import overload, Sequence

import sys

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT_FROM_SCRIPT))

from src.CONFIG import (
     STATUS_OBSERVED,
     STATUS_IMPUTE,
     EPS,
     BASELINE_LEVELS
)

from src.training import beta_binomial_nll_from_prob


def get_baseline_levels(mode: str):
    if mode not in BASELINE_LEVELS:
        valid = list(BASELINE_LEVELS.keys())
        raise ValueError(
            f"Unknown baseline mode: {mode}. Valid modes: {valid}"
        )

    return BASELINE_LEVELS[mode]

def fit_global_phi_for_baseline(
    df,
    pred_col,
    init_log_phi=4.0,
    lr=0.5,
    max_iter=100,
):
    """
    Fit one global beta-binomial dispersion parameter for a fixed baseline.

    The baseline mean predictions are not changed.
    Only phi is fitted.

    Fit this on the train split, then reuse the fitted phi on val/test.
    """
    p = torch.tensor(df[pred_col].to_numpy(dtype=float), dtype=torch.float32)
    n_s = torch.tensor(df["n_S"].to_numpy(dtype=float), dtype=torch.float32)
    n_total = torch.tensor(df["n_total"].to_numpy(dtype=float), dtype=torch.float32)

    log_phi = torch.nn.Parameter(torch.tensor(float(init_log_phi)))

    optimizer = torch.optim.LBFGS(
        [log_phi],
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()

        loss = beta_binomial_nll_from_prob(
            p=p,
            n_s=n_s,
            n_total=n_total,
            log_phi=log_phi,
            reduction="mean_per_test",
        )

        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        final_loss = beta_binomial_nll_from_prob(
            p=p,
            n_s=n_s,
            n_total=n_total,
            log_phi=log_phi,
            reduction="mean_per_test",
        )

        phi = F.softplus(log_phi).item()

    return {
        "log_phi": float(log_phi.detach().cpu()),
        "phi": float(phi),
        "beta_binomial_nll_per_test": float(final_loss.detach().cpu()),
    }

def load_dataset(path: Path) -> pd.DataFrame: 
    """ Load the standardized AMR table and enforce the minimal schema required by the baseline script. 
    Expected unit of observation: Species × Family × Country × Year
    Expected status values:
        observed   -> empirical cell with n_S and n_total
        to_impute  -> missing but imputable cell, no empirical counts
        intrinsic  -> excluded from imputation, handled outside this baseline
    In this baseline we only evaluate on observed cells, but we keep the full dataset because the same fitted means are later used to produce predictions for status == to_impute.
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
        "prop_S",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    # Standardize string columns to avoid failed joins due to whitespace or
    # mixed string dtypes. This matters because all baselines are group means
    # computed through merges.
    for col in ["Species", "Family", "Country", "status"]:
        df[col] = df[col].astype("string").str.strip()
    # Keep status convention lowercase, assuming CONFIG uses lowercase too:
    # STATUS_OBSERVED = "observed"
    # STATUS_IMPUTE = "to_impute"
    df["status"] = df["status"].str.lower()
    df["Year"] = df["Year"].astype(int)
    # Counts are only meaningful for observed cells. Non-observed rows may have
    # missing values here, which is expected.
    df["n_S"] = pd.to_numeric(df["n_S"], errors="coerce")
    df["n_total"] = pd.to_numeric(df["n_total"], errors="coerce")
    observed = df["status"].eq(STATUS_OBSERVED)
    # Recompute prop_S from counts instead of trusting the input column.
    # This makes the target internally consistent with n_S and n_total, and
    # prevents stale prop_S values from surviving after preprocessing.
    df["prop_S"] = np.nan
    df.loc[observed, "prop_S"] = (
        df.loc[observed, "n_S"] / df.loc[observed, "n_total"]
    )
    return df

def split_observed(df, train_max_year, test_min_year):
    observed_df = df[df["status"].eq(STATUS_OBSERVED)].copy()

    train_df = observed_df[observed_df["Year"] <= train_max_year].copy()
    test_df = observed_df[observed_df["Year"] >= test_min_year].copy()

    if train_df.empty:
        raise ValueError("Training set is empty.")

    if test_df.empty:
        raise ValueError("Test set is empty.")

    return train_df, test_df

@overload
def smoothed_rate(
    df: pd.DataFrame,
    group_cols: None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    ...


@overload
def smoothed_rate(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> pd.DataFrame:
    ...
    
def smoothed_rate(df, group_cols=None, alpha=1.0, beta=1.0): 
    """ Estimate susceptibility probability using empirical counts plus a Beta prior. 
    For a group g, the estimate is: p_hat_g = (sum(n_S_g) + alpha) / (sum(n_total_g) + alpha + beta) 
    This is equivalent to a Beta-Binomial posterior mean with prior Beta(alpha, beta). 
    The smoothing avoids exact 0/1 probabilities in groups with small counts, which would otherwise create infinite or unstable log-loss values. 
    If group_cols is None, the function returns the global susceptibility rate estimated over the whole training set. """

    if group_cols is None:
        n_S = df["n_S"].sum()
        n_total = df["n_total"].sum()
        return (n_S + alpha) / (n_total + alpha + beta)

    out = (
        df.groupby(group_cols)
        .agg(
            train_n_S=("n_S", "sum"),
            train_n_total=("n_total", "sum"),
        )
        .reset_index()
    )
    out["p_hat"] = (
        (out["train_n_S"] + alpha) / (out["train_n_total"] + alpha + beta)
    )
    
    return out

# Add mean prediction to target DataFrame, the simplest of the baselines, which is just the smoothed mean of the training data for each group, or the global mean if no training data exists for that group.
def add_mean_prediction(train_df, target_df, group_cols, pred_col, global_p, alpha=1.0, beta=1.0):
    mean_df = smoothed_rate(
        train_df,
        group_cols=group_cols,
        alpha=alpha,
        beta=beta,
    )

    pred_df = target_df.merge(
        mean_df[group_cols + ["p_hat"]],
        on=group_cols,
        how="left",
    )

    pred_df[pred_col] = pred_df["p_hat"].fillna(global_p)
    pred_df = pred_df.drop(columns=["p_hat"])

    return pred_df

def add_hierarchical_prediction(train_df, target_df, global_p, alpha=1.0, beta=1.0):
    """
    Add a hierarchical fallback prediction.

    The idea is to use the most specific historical estimate available in the
    training set. If a test/imputation cell has never been observed at a
    specific level, the prediction falls back to a broader level.

    Fallback order:
        1. Country × Species × Family
        2. Species × Family
        3. Family
        4. Species
        5. Global mean

    This gives a simple but strong non-parametric baseline:
    it uses local historical information when available, but never fails for
    unseen combinations because it eventually falls back to the global rate.
    """
    pred_df = target_df.copy()

    hierarchy = [
        (["Country", "Species", "Family"], "p_country_species_family"),
        (["Species", "Family"], "p_species_family"),
        (["Family"], "p_family"),
        (["Species"], "p_species"),
    ]

    # If target_df already contains prediction columns from previous baselines,
    # drop them before merging. Otherwise pandas creates _x/_y columns and the
    # expected column names disappear, because apparently chaos needed helpers.
    hierarchy_cols = [col_name for _, col_name in hierarchy]
    pred_df = pred_df.drop(columns=hierarchy_cols + ["p_hierarchical"], errors="ignore")

    for group_cols, col_name in hierarchy:
        mean_df = smoothed_rate(
            train_df,
            group_cols=group_cols,
            alpha=alpha,
            beta=beta,
        ).rename(columns={"p_hat": col_name})

        pred_df = pred_df.merge(
            mean_df[group_cols + [col_name]],
            on=group_cols,
            how="left",
        )

    pred_df["p_hierarchical"] = pred_df["p_country_species_family"]

    for fallback_col in ["p_species_family", "p_family", "p_species"]:
        pred_df["p_hierarchical"] = pred_df["p_hierarchical"].fillna(
            pred_df[fallback_col]
        )

    pred_df["p_hierarchical"] = pred_df["p_hierarchical"].fillna(global_p)

    return pred_df

def evaluate_predictions(df, pred_col, model_name):
    """
    Evaluate predicted susceptibility probabilities on observed cells.

    Metrics:
        weighted_mae / weighted_rmse:
            Errors on prop_S weighted by n_total. These prioritize cells with
            more empirical support.

        unweighted_mae / unweighted_rmse:
            Errors where each cell contributes equally, regardless of sample size.

        binomial_ce_per_test:
            Count-level binomial cross-entropy per isolate/test, ignoring the
            combinatorial constant. This is useful for comparing probability
            forecasts because it rewards calibrated probabilities and penalizes
            overconfident wrong predictions.
    """
    y = df["prop_S"].to_numpy(dtype=float)
    p = df[pred_col].to_numpy(dtype=float)
    w = df["n_total"].to_numpy(dtype=float)
    k = df["n_S"].to_numpy(dtype=float)
    n = df["n_total"].to_numpy(dtype=float)

    # Avoid log(0) in the binomial cross-entropy.
    p = np.clip(p, EPS, 1.0 - EPS)

    weighted_mae = np.average(np.abs(y - p), weights=w)
    weighted_rmse = np.sqrt(np.average((y - p) ** 2, weights=w))

    binomial_ce_per_test = -np.sum(
        k * np.log(p) + (n - k) * np.log(1.0 - p)
    ) / np.sum(n)

    unweighted_mae = np.mean(np.abs(y - p))
    unweighted_rmse = np.sqrt(np.mean((y - p) ** 2))

    return {
        "model_name": model_name,
        "n_cells": int(len(df)),
        "n_tests": int(np.sum(n)),
        "weighted_mae": float(weighted_mae),
        "weighted_rmse": float(weighted_rmse),
        "unweighted_mae": float(unweighted_mae),
        "unweighted_rmse": float(unweighted_rmse),
        "binomial_ce_per_test": float(binomial_ce_per_test),
    }

# Build predictions for the test set using the training set and the specified alpha and beta parameters for smoothing.
def build_test_predictions(train_df, test_df, alpha=1.0, beta=1.0):
    global_p = smoothed_rate(
        train_df,
        group_cols=None,
        alpha=alpha,
        beta=beta,
    )

    pred_df = test_df.copy()

    pred_df["p_global"] = global_p

    pred_df = add_mean_prediction(
        train_df=train_df,
        target_df=pred_df,
        group_cols=["Species"],
        pred_col="p_species",
        global_p=global_p,
        alpha=alpha,
        beta=beta,
    )

    pred_df = add_mean_prediction(
        train_df=train_df,
        target_df=pred_df,
        group_cols=["Family"],
        pred_col="p_family",
        global_p=global_p,
        alpha=alpha,
        beta=beta,
    )

    pred_df = add_mean_prediction(
        train_df=train_df,
        target_df=pred_df,
        group_cols=["Species", "Family"],
        pred_col="p_species_family",
        global_p=global_p,
        alpha=alpha,
        beta=beta,
    )

    pred_df = add_mean_prediction(
        train_df=train_df,
        target_df=pred_df,
        group_cols=["Country", "Species", "Family"],
        pred_col="p_country_species_family",
        global_p=global_p,
        alpha=alpha,
        beta=beta,
    )

    pred_df = add_hierarchical_prediction(
        train_df=train_df,
        target_df=pred_df,
        global_p=global_p,
        alpha=alpha,
        beta=beta,
    )

    prediction_cols = [
    "p_global",
    "p_species",
    "p_family",
    "p_species_family",
    "p_country_species_family",
    "p_hierarchical",
    ]

    missing_prediction_counts = {
        col: int(pred_df[col].isna().sum())
        for col in prediction_cols
    }

    for col in prediction_cols:
        pred_df[col] = pred_df[col].fillna(global_p)

    return pred_df, global_p, missing_prediction_counts

# Evaluate all models and return a DataFrame with the results.
def evaluate_all(pred_df, missing_prediction_counts=None, split="test", evaluation_protocol="temporal_year_holdout"):
    """
    Evaluate all simple-baseline models.

    `split` and `evaluation_protocol` are written onto every row so this
    table can be safely combined downstream (e.g. in
    09_compare_model_metrics.py) without silently conflating this
    temporal (train_max_year/test_min_year) holdout with the frozen,
    random cell-level train/val/test split produced by
    03_train_base_latent_model.py and reused by every neural model script.
    Two evaluation sets answer different questions (forecasting future
    years vs. completing missing cells within known years), so their
    metrics are NOT a like-for-like comparison even when both happen to
    be labeled split="test".
    """
    model_cols = {
        "global_mean": "p_global",
        "species_mean": "p_species",
        "family_mean": "p_family",
        "species_family_mean": "p_species_family",
        "country_species_family_mean": "p_country_species_family",
        "hierarchical_fallback_mean": "p_hierarchical",
    }

    rows = []

    for model_name, pred_col in model_cols.items():
        row = evaluate_predictions(
            df=pred_df,
            pred_col=pred_col,
            model_name=model_name,
        )

        if missing_prediction_counts is not None:
            row["n_missing_before_global_fallback"] = missing_prediction_counts.get(pred_col, 0)

        row["split"] = split
        row["evaluation_protocol"] = evaluation_protocol

        rows.append(row)

    return pd.DataFrame(rows)

def predict_to_impute(df, train_df, global_p, alpha=1.0, beta=1.0):
    """
    Produce susceptibility predictions for imputable missing cells.

    These rows are not used for evaluation because their true n_S/n_total is
    unavailable. The goal is to generate a first simple completion table using
    the same hierarchical fallback baseline fitted on observed training cells.

    Intrinsic-resistance cells are intentionally excluded: they are not missing
    values to impute, but structural non-susceptibility constraints.
    """
    to_impute_df = df[df["status"].eq(STATUS_IMPUTE)].copy()

    if to_impute_df.empty:
        return to_impute_df

    pred_df = add_hierarchical_prediction(
        train_df=train_df,
        target_df=to_impute_df,
        global_p=global_p,
        alpha=alpha,
        beta=beta,
    )

    pred_df = pred_df.rename(
        columns={"p_hierarchical": "predicted_prop_S"}
    )

    keep_cols = [
        "Species",
        "Family",
        "Country",
        "Year",
        "status",
        "predicted_prop_S",
    ]

    return pred_df[keep_cols].copy()


def evaluate_predictions_beta_binomial(df, pred_col, model_name, log_phi):
    """
    Evaluate a fixed-probability baseline under both ordinary point metrics
    and beta-binomial likelihood.
    """
    row = evaluate_predictions(
        model_name=model_name,
        df=df,
        pred_col=pred_col
    )

    p = df[pred_col].to_numpy(dtype=float)
    n_s = df["n_S"].to_numpy(dtype=float)
    n_total = df["n_total"].to_numpy(dtype=float)

    log_phi_tensor = torch.tensor(float(log_phi), dtype=torch.float32)

    bb_nll_per_test = beta_binomial_nll_from_prob(
        p=p,
        n_s=n_s,
        n_total=n_total,
        log_phi=log_phi_tensor,
        reduction="mean_per_test",
    )

    bb_nll_per_cell = beta_binomial_nll_from_prob(
        p=p,
        n_s=n_s,
        n_total=n_total,
        log_phi=log_phi_tensor,
        reduction="mean_per_cell",
    )

    row["beta_binomial_nll_per_test"] = float(bb_nll_per_test.detach().cpu())
    row["beta_binomial_nll_per_cell"] = float(bb_nll_per_cell.detach().cpu())
    row["log_phi"] = float(log_phi)
    row["phi"] = float(F.softplus(log_phi_tensor).detach().cpu())

    return row

def add_hierarchical_prediction_leave_one_out(
    train_df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> pd.DataFrame:
    """
    Compute leave-one-out hierarchical baseline predictions for training rows.

    Each training row is predicted using group statistics computed after
    removing that row itself from every level of the hierarchy.

    Fallback order:
        Country × Species × Family
        Species × Family
        Family
        Species
        Global

    This prevents a training target from leaking into its own baseline offset.
    """
    required = [
        "Country",
        "Species",
        "Family",
        "n_S",
        "n_total",
        "prop_S",
    ]

    missing = [col for col in required if col not in train_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    pred_df = train_df.copy()

    hierarchy = [
        (["Country", "Species", "Family"], "country_species_family"),
        (["Species", "Family"], "species_family"),
        (["Family"], "family"),
        (["Species"], "species"),
    ]

    # Leave-one-out global estimate.
    total_n_s = pred_df["n_S"].sum()
    total_n_total = pred_df["n_total"].sum()

    loo_global_n_s = total_n_s - pred_df["n_S"]
    loo_global_n_total = total_n_total - pred_df["n_total"]

    pred_df["p_global_loo"] = (
        loo_global_n_s + alpha
    ) / (
        loo_global_n_total + alpha + beta
    )

    # In practice the global denominator should always be positive, but we
    # guard it anyway because datasets enjoy finding new ways to disappoint us.
    pred_df.loc[
        loo_global_n_total <= 0,
        "p_global_loo",
    ] = np.nan

    for group_cols, level_name in hierarchy:
        group_stats = (
            train_df.groupby(group_cols)
            .agg(
                group_n_S=("n_S", "sum"),
                group_n_total=("n_total", "sum"),
            )
            .reset_index()
        )

        pred_df = pred_df.merge(
            group_stats,
            on=group_cols,
            how="left",
        )

        loo_n_s = pred_df["group_n_S"] - pred_df["n_S"]
        loo_n_total = pred_df["group_n_total"] - pred_df["n_total"]

        pred_col = f"p_{level_name}_loo"

        pred_df[pred_col] = (
            loo_n_s + alpha
        ) / (
            loo_n_total + alpha + beta
        )

        # A group containing only this row has no remaining historical support,
        # so this level must be unavailable and trigger the next fallback.
        pred_df.loc[loo_n_total <= 0, pred_col] = np.nan

        pred_df = pred_df.drop(
            columns=["group_n_S", "group_n_total"]
        )

    pred_df["p_hierarchical_baseline"] = (
        pred_df["p_country_species_family_loo"]
    )

    fallback_cols = [
        "p_species_family_loo",
        "p_family_loo",
        "p_species_loo",
        "p_global_loo",
    ]

    for fallback_col in fallback_cols:
        pred_df["p_hierarchical_baseline"] = (
            pred_df["p_hierarchical_baseline"]
            .fillna(pred_df[fallback_col])
        )

    pred_df["baseline_source"] = "country_species_family_loo"

    source_rules = [
        (
            pred_df["p_country_species_family_loo"].isna()
            & pred_df["p_species_family_loo"].notna(),
            "species_family_loo",
        ),
        (
            pred_df["p_country_species_family_loo"].isna()
            & pred_df["p_species_family_loo"].isna()
            & pred_df["p_family_loo"].notna(),
            "family_loo",
        ),
        (
            pred_df["p_country_species_family_loo"].isna()
            & pred_df["p_species_family_loo"].isna()
            & pred_df["p_family_loo"].isna()
            & pred_df["p_species_loo"].notna(),
            "species_loo",
        ),
        (
            pred_df["p_country_species_family_loo"].isna()
            & pred_df["p_species_family_loo"].isna()
            & pred_df["p_family_loo"].isna()
            & pred_df["p_species_loo"].isna(),
            "global_loo",
        ),
    ]

    for mask, label in source_rules:
        pred_df.loc[mask, "baseline_source"] = label

    return pred_df


def add_configured_baseline_prediction(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    mode: str,
    global_p: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    pred_col: str = "p_baseline",
    source_col: str = "baseline_source",
) -> pd.DataFrame:
    """
    Add a configurable baseline prediction to target_df.

    Examples:
        mode="species_family":
            Species × Family -> Family -> Species -> Global

        mode="country_species_family":
            Country × Species × Family -> Species × Family
            -> Family -> Species -> Global
    """
    levels = get_baseline_levels(mode)
    pred_df = target_df.copy()

    level_prediction_cols = [f"p_{label}" for _, label in levels]

    pred_df = pred_df.drop(
        columns=level_prediction_cols + [pred_col, source_col],
        errors="ignore",
    )

    for group_cols, label in levels:
        level_df = smoothed_rate(
            train_df,
            group_cols=group_cols,
            alpha=alpha,
            beta=beta,
        ).rename(columns={"p_hat": f"p_{label}"})

        pred_df = pred_df.merge(
            level_df[group_cols + [f"p_{label}"]],
            on=group_cols,
            how="left",
        )

    prediction = pd.Series(np.nan, index=pred_df.index, dtype=float)
    source = pd.Series("global", index=pred_df.index, dtype="string")

    for _, label in levels:
        level_col = f"p_{label}"

        use_this_level = prediction.isna() & pred_df[level_col].notna()

        prediction.loc[use_this_level] = pred_df.loc[
            use_this_level,
            level_col,
        ]

        source.loc[use_this_level] = label

    pred_df[pred_col] = prediction.fillna(global_p)
    pred_df[source_col] = source

    return pred_df

def add_configured_baseline_prediction_leave_one_out(
    train_df: pd.DataFrame,
    mode: str,
    alpha: float = 1.0,
    beta: float = 1.0,
    pred_col: str = "p_baseline",
    source_col: str = "baseline_source",
) -> pd.DataFrame:
    """
    Build leave-one-out baseline predictions for training rows.

    Every row is removed from the statistics used to predict itself.
    This prevents the target from leaking into its own residual offset.
    """
    levels = get_baseline_levels(mode)
    pred_df = train_df.copy()

    total_n_s = pred_df["n_S"].sum()
    total_n_total = pred_df["n_total"].sum()

    full_global_p = (
        total_n_s + alpha
    ) / (
        total_n_total + alpha + beta
    )

    loo_global_n_s = total_n_s - pred_df["n_S"]
    loo_global_n_total = total_n_total - pred_df["n_total"]

    pred_df["p_global_loo"] = (
        loo_global_n_s + alpha
    ) / (
        loo_global_n_total + alpha + beta
    )

    pred_df.loc[
        loo_global_n_total <= 0,
        "p_global_loo",
    ] = full_global_p

    level_prediction_cols = []

    for group_cols, label in levels:
        group_stats = (
            train_df.groupby(group_cols)
            .agg(
                group_n_s=("n_S", "sum"),
                group_n_total=("n_total", "sum"),
            )
            .reset_index()
        )

        pred_df = pred_df.merge(
            group_stats,
            on=group_cols,
            how="left",
        )

        loo_n_s = pred_df["group_n_s"] - pred_df["n_S"]
        loo_n_total = pred_df["group_n_total"] - pred_df["n_total"]

        level_col = f"p_{label}_loo"
        level_prediction_cols.append(level_col)

        pred_df[level_col] = (
            loo_n_s + alpha
        ) / (
            loo_n_total + alpha + beta
        )

        # If this row was the only one in the group, fallback to the next level.
        pred_df.loc[loo_n_total <= 0, level_col] = np.nan

        pred_df = pred_df.drop(
            columns=["group_n_s", "group_n_total"]
        )

    prediction = pd.Series(np.nan, index=pred_df.index, dtype=float)
    source = pd.Series("global_loo", index=pred_df.index, dtype="string")

    for _, label in levels:
        level_col = f"p_{label}_loo"

        use_this_level = prediction.isna() & pred_df[level_col].notna()

        prediction.loc[use_this_level] = pred_df.loc[
            use_this_level,
            level_col,
        ]

        source.loc[use_this_level] = f"{label}_loo"

    pred_df[pred_col] = prediction.fillna(pred_df["p_global_loo"])
    pred_df[source_col] = source

    return pred_df