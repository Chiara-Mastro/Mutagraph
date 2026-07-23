"""
Leakage-safe temporal walk-forward baselines for AMR susceptibility forecasting.

For every target year T:
    history = observed rows with Year < T
    target  = observed rows with Year == T

The module performs explicit temporal assertions and exports provenance columns
with every prediction. No target-year observation is used to construct its own
prediction.

This file replaces src/temporal_baselines.py.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.baselines import smoothed_rate


CSF_COLS = ["Country", "Species", "Family"]
SF_COLS = ["Species", "Family"]


def _require_columns(df: pd.DataFrame, columns: list[str], name: str):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _validate_history_and_target(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_year: int,
):
    if history_df.empty:
        raise ValueError(
            f"Target year {target_year} has no historical observations."
        )
    if target_df.empty:
        raise ValueError(
            f"Target year {target_year} has no target observations."
        )

    history_years = pd.to_numeric(history_df["Year"], errors="raise")
    target_years = pd.to_numeric(target_df["Year"], errors="raise")

    if not history_years.lt(target_year).all():
        bad = sorted(history_years.loc[~history_years.lt(target_year)].unique())
        raise AssertionError(
            f"Temporal leakage for target year {target_year}: history contains "
            f"years {bad}, but every history year must be < {target_year}."
        )

    if not target_years.eq(target_year).all():
        bad = sorted(target_years.loc[~target_years.eq(target_year)].unique())
        raise AssertionError(
            f"Target frame for {target_year} contains other years: {bad}."
        )

    if set(history_df.index).intersection(set(target_df.index)):
        raise AssertionError(
            f"History and target row indices overlap for target year {target_year}."
        )


def _species_family_fallback_table(
    history_df: pd.DataFrame,
    alpha: float,
    beta: float,
):
    """Species x Family smoothed rate calculated from history only."""
    return smoothed_rate(
        history_df,
        group_cols=SF_COLS,
        alpha=alpha,
        beta=beta,
    ).rename(columns={"p_hat": "p_species_family"})


def _apply_fallback(
    target_df: pd.DataFrame,
    level_df: pd.DataFrame,
    level_cols: list[str],
    level_pred_col: str,
    sf_df: pd.DataFrame,
    global_p: float,
    source_label: str,
):
    out = target_df.merge(level_df, on=level_cols, how="left")
    out = out.merge(
        sf_df[SF_COLS + ["p_species_family"]],
        on=SF_COLS,
        how="left",
    )

    p_pred = pd.to_numeric(out[level_pred_col], errors="coerce")
    source = pd.Series(source_label, index=out.index, dtype="string")

    use_sf = p_pred.isna() & out["p_species_family"].notna()
    p_pred = p_pred.where(~use_sf, out["p_species_family"])
    source = source.mask(use_sf, "species_family")

    use_global = p_pred.isna()
    p_pred = p_pred.where(~use_global, float(global_p))
    source = source.mask(use_global, "global")

    out["p_pred"] = np.clip(p_pred.astype(float), 1e-6, 1.0 - 1e-6)
    out["baseline_source"] = source

    return out.drop(columns=[level_pred_col, "p_species_family"])


def predict_species_family_mean(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    alpha: float,
    beta: float,
    global_p: float,
) :
    sf_df = _species_family_fallback_table(history_df, alpha, beta)

    out = target_df.merge(sf_df, on=SF_COLS, how="left")
    p_pred = pd.to_numeric(out["p_species_family"], errors="coerce")
    source = pd.Series("species_family", index=out.index, dtype="string")

    use_global = p_pred.isna()
    p_pred = p_pred.where(~use_global, float(global_p))
    source = source.mask(use_global, "global")

    out["p_pred"] = np.clip(p_pred.astype(float), 1e-6, 1.0 - 1e-6)
    out["baseline_source"] = source

    return out.drop(columns=["p_species_family"])


def predict_country_species_family_mean(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    alpha: float,
    beta: float,
    global_p: float,
) :
    csf_df = smoothed_rate(
        history_df,
        group_cols=CSF_COLS,
        alpha=alpha,
        beta=beta,
    ).rename(columns={"p_hat": "p_country_species_family"})

    # Provenance for the country-specific estimate.
    csf_history = (
        history_df.groupby(CSF_COLS, as_index=False)
        .agg(
            csf_history_min_year=("Year", "min"),
            csf_history_max_year=("Year", "max"),
            csf_history_n_years=("Year", "nunique"),
        )
    )
    csf_df = csf_df.merge(csf_history, on=CSF_COLS, how="left")

    sf_df = _species_family_fallback_table(history_df, alpha, beta)

    return _apply_fallback(
        target_df=target_df,
        level_df=csf_df,
        level_cols=CSF_COLS,
        level_pred_col="p_country_species_family",
        sf_df=sf_df,
        global_p=global_p,
        source_label="country_species_family",
    )


def predict_locf(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    alpha: float,
    beta: float,
    global_p: float,
) :
    """
    Last observed Year < T for each Country x Species x Family.

    The previous implementation accidentally retained a GroupBy object rather
    than selecting the last row. Here tail(1) performs the intended operation.
    """
    last_obs = (
        history_df.sort_values(CSF_COLS + ["Year"])
        .groupby(CSF_COLS, as_index=False, sort=False)
        .tail(1)
        .copy()
    )

    last_obs["p_locf"] = (
        last_obs["n_S"].astype(float) + alpha
    ) / (
        last_obs["n_total"].astype(float) + alpha + beta
    )
    last_obs = last_obs.rename(columns={"Year": "locf_year"})

    sf_df = _species_family_fallback_table(history_df, alpha, beta)

    out = _apply_fallback(
        target_df=target_df,
        level_df=last_obs[
            CSF_COLS + ["p_locf", "locf_year"]
        ],
        level_cols=CSF_COLS,
        level_pred_col="p_locf",
        sf_df=sf_df,
        global_p=global_p,
        source_label="locf",
    )

    out["locf_lag_years"] = out["Year"] - out["locf_year"]

    used_locf = out["baseline_source"].eq("locf")
    if (
        out.loc[used_locf, "locf_year"]
        .ge(out.loc[used_locf, "Year"])
        .any()
    ):
        raise AssertionError("LOCF used a present or future observation.")

    return out


def predict_rolling_mean_k(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    alpha: float,
    beta: float,
    global_p: float,
    k: int = 3,
) :
    if k < 1:
        raise ValueError("k must be at least 1.")

    sorted_hist = history_df.sort_values(
        CSF_COLS + ["Year"],
        ascending=[True, True, True, False],
    )
    topk = sorted_hist.groupby(
        CSF_COLS,
        as_index=False,
        sort=False,
    ).head(k)

    rolling_df = (
        topk.groupby(CSF_COLS, as_index=False)
        .agg(
            n_S=("n_S", "sum"),
            n_total=("n_total", "sum"),
            n_years_used=("Year", "nunique"),
            rolling_history_min_year=("Year", "min"),
            rolling_history_max_year=("Year", "max"),
        )
    )
    rolling_df["p_rolling"] = (
        rolling_df["n_S"] + alpha
    ) / (
        rolling_df["n_total"] + alpha + beta
    )

    sf_df = _species_family_fallback_table(history_df, alpha, beta)

    return _apply_fallback(
        target_df=target_df,
        level_df=rolling_df[
            CSF_COLS
            + [
                "p_rolling",
                "n_years_used",
                "rolling_history_min_year",
                "rolling_history_max_year",
            ]
        ],
        level_cols=CSF_COLS,
        level_pred_col="p_rolling",
        sf_df=sf_df,
        global_p=global_p,
        source_label=f"rolling_mean_k{k}",
    )


def predict_ewma_residual(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    alpha: float,
    beta: float,
    global_p: float,
    halflife_years: float = 2.0,
    eps: float = 1e-6,
) :
    if halflife_years <= 0:
        raise ValueError("halflife_years must be positive.")

    sf_df = _species_family_fallback_table(history_df, alpha, beta)

    hist = history_df.merge(sf_df, on=SF_COLS, how="left")
    hist["p_species_family"] = hist["p_species_family"].fillna(global_p)
    hist["residual"] = hist["prop_S"] - hist["p_species_family"]

    target_year = int(target_df["Year"].iloc[0])
    hist["weight"] = 0.5 ** (
        (target_year - hist["Year"]) / float(halflife_years)
    )
    hist["weighted_residual"] = hist["weight"] * hist["residual"]

    ewma_residual = (
        hist.groupby(CSF_COLS, as_index=False)
        .agg(
            weighted_residual_sum=("weighted_residual", "sum"),
            weight_sum=("weight", "sum"),
            ewma_history_min_year=("Year", "min"),
            ewma_history_max_year=("Year", "max"),
            ewma_history_n_years=("Year", "nunique"),
        )
    )
    ewma_residual["ewma_residual"] = (
        ewma_residual["weighted_residual_sum"]
        / ewma_residual["weight_sum"]
    )
    ewma_residual = ewma_residual.drop(
        columns=["weighted_residual_sum", "weight_sum"]
    )

    ewma_residual = ewma_residual.merge(sf_df, on=SF_COLS, how="left")
    ewma_residual["p_species_family"] = (
        ewma_residual["p_species_family"].fillna(global_p)
    )
    ewma_residual["p_ewma"] = np.clip(
        ewma_residual["p_species_family"]
        + ewma_residual["ewma_residual"],
        eps,
        1.0 - eps,
    )

    return _apply_fallback(
        target_df=target_df,
        level_df=ewma_residual[
            CSF_COLS
            + [
                "p_ewma",
                "ewma_residual",
                "ewma_history_min_year",
                "ewma_history_max_year",
                "ewma_history_n_years",
            ]
        ],
        level_cols=CSF_COLS,
        level_pred_col="p_ewma",
        sf_df=sf_df,
        global_p=global_p,
        source_label="csf_ewma_residual",
    )


def eligible_target_years(
    df_observed: pd.DataFrame,
    min_train_years: int,
) :
    if min_train_years < 1:
        raise ValueError("min_train_years must be at least 1.")

    years = sorted(
        pd.to_numeric(df_observed["Year"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )

    eligible: list[int] = []
    for target_year in years:
        n_history_years = int(
            df_observed.loc[
                df_observed["Year"] < target_year,
                "Year",
            ].nunique()
        )
        has_target_cells = bool(
            df_observed["Year"].eq(target_year).any()
        )

        if n_history_years >= min_train_years and has_target_cells:
            eligible.append(int(target_year))

    return eligible


def run_walk_forward(
    df_observed: pd.DataFrame,
    predict_fn: Callable,
    min_train_years: int,
    alpha: float,
    beta: float,
    method_name: str,
    **extra_kwargs,
) :
    """
    Generate walk-forward predictions with explicit temporal provenance.

    Every output row contains:
        target_year
        forecast_origin_year
        history_min_year
        history_max_year
        n_history_years
        n_history_cells
        n_history_tests
        training_cutoff_exclusive
        prediction_history_rule
        target_outcome_used_as_input
        temporal_leakage_check_passed
    """
    _require_columns(
        df_observed,
        [
            "Country",
            "Species",
            "Family",
            "Year",
            "n_S",
            "n_total",
            "prop_S",
        ],
        "df_observed",
    )

    target_years = eligible_target_years(
        df_observed,
        min_train_years=min_train_years,
    )
    if not target_years:
        raise ValueError(
            "No eligible target years with at least "
            f"{min_train_years} prior observed years."
        )

    rows: list[pd.DataFrame] = []

    for target_year in target_years:
        history_df = df_observed.loc[
            df_observed["Year"] < target_year
        ].copy()
        target_df = df_observed.loc[
            df_observed["Year"] == target_year
        ].copy()

        _validate_history_and_target(
            history_df=history_df,
            target_df=target_df,
            target_year=target_year,
        )

        global_p = smoothed_rate(
            history_df,
            group_cols=None,
            alpha=alpha,
            beta=beta,
        )

        pred_df = predict_fn(
            history_df=history_df,
            target_df=target_df,
            alpha=alpha,
            beta=beta,
            global_p=global_p,
            **extra_kwargs,
        ).copy()

        if len(pred_df) != len(target_df):
            raise AssertionError(
                f"{method_name} returned {len(pred_df)} rows for "
                f"{len(target_df)} targets in year {target_year}."
            )
        if "p_pred" not in pred_df.columns:
            raise ValueError(
                f"{method_name} did not return a p_pred column."
            )
        if not np.isfinite(
            pd.to_numeric(pred_df["p_pred"], errors="coerce")
        ).all():
            raise ValueError(
                f"{method_name} produced non-finite predictions for "
                f"target year {target_year}."
            )
        if not pd.to_numeric(
            pred_df["Year"],
            errors="raise",
        ).eq(target_year).all():
            raise AssertionError(
                f"{method_name} changed target-year identities."
            )

        history_years = pd.to_numeric(
            history_df["Year"],
            errors="raise",
        )
        history_max_year = int(history_years.max())
        history_min_year = int(history_years.min())

        if history_max_year >= target_year:
            raise AssertionError(
                f"Temporal leakage: history_max_year={history_max_year} "
                f"for target_year={target_year}."
            )

        pred_df["target_year"] = int(target_year)
        pred_df["forecast_origin_year"] = int(target_year - 1)
        pred_df["history_min_year"] = history_min_year
        pred_df["history_max_year"] = history_max_year
        pred_df["n_history_years"] = int(history_years.nunique())
        pred_df["n_history_cells"] = int(len(history_df))
        pred_df["n_history_tests"] = int(history_df["n_total"].sum())
        pred_df["training_cutoff_exclusive"] = int(target_year)
        pred_df["prediction_history_rule"] = "Year < target_year"
        pred_df["target_outcome_used_as_input"] = False
        pred_df["temporal_leakage_check_passed"] = True
        pred_df["method"] = method_name

        if "locf_year" in pred_df.columns:
            used = pred_df["locf_year"].notna()
            if pred_df.loc[used, "locf_year"].ge(target_year).any():
                raise AssertionError(
                    f"LOCF provenance failed for target year {target_year}."
                )

        for candidate in [
            "csf_history_max_year",
            "rolling_history_max_year",
            "ewma_history_max_year",
        ]:
            if candidate in pred_df.columns:
                used = pred_df[candidate].notna()
                if pred_df.loc[used, candidate].ge(target_year).any():
                    raise AssertionError(
                        f"{candidate} contains a present/future year for "
                        f"target {target_year}."
                    )

        rows.append(pred_df)

    out = pd.concat(rows, ignore_index=True)

    if not out["history_max_year"].lt(out["target_year"]).all():
        raise AssertionError(
            "Final walk-forward table failed history_max_year < target_year."
        )
    if out["target_outcome_used_as_input"].any():
        raise AssertionError(
            "At least one prediction claims to use the target outcome."
        )

    return out
