"""
Expanding-window Species-Family baseline and logit-scale residuals for the
shared-dynamics temporal residual model (13_train_shared_dynamics_model.py).

For every (Species, Family, Year) combination present in the dataset, the
baseline p^base_{s,f,t} is fit using ONLY observed cells with Year < t
falling back to the global historical rate when a Species-Family pair has no
prior observations at all. Every downstream residual, for encoder input and
decoder target alike, is defined against this leakage-free baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines import smoothed_rate
from src.temporal_baselines import SF_COLS


def _logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def compute_expanding_species_family_baseline(
    df_observed: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-6,
):
    """
    One row per (Species, Family, Year) present in df_observed, giving the
    Species-Family baseline probability and logit fit on strictly earlier
    years only.

    The very first year in the dataset has no prior history for anyone, so
    its baseline is the Beta(alpha, beta) prior mean (alpha / (alpha+beta))
    for every Species-Family -- smoothed_rate's natural behavior on an empty
    history_df, kept rather than special-cased so the fallback rule is
    exactly the same one used everywhere else in the pipeline.
    """
    years = sorted(df_observed["Year"].unique().tolist())
    rows = []

    for year in years:
        history_df = df_observed[df_observed["Year"] < year]

        global_p = smoothed_rate(history_df, group_cols=None, alpha=alpha, beta=beta)

        sf_df = smoothed_rate(history_df, group_cols=SF_COLS, alpha=alpha, beta=beta)
        sf_df = sf_df.rename(columns={"p_hat": "p_baseline"})
        sf_df["baseline_source"] = "species_family"

        # Species-Family pairs present in this year's target cells but with
        # zero prior observations still need a row, tagged as a global
        # fallback rather than silently missing.
        sf_seen_this_year = (
            df_observed.loc[df_observed["Year"].eq(year), SF_COLS]
            .drop_duplicates()
        )
        year_df = sf_seen_this_year.merge(sf_df, on=SF_COLS, how="left")
        missing = year_df["p_baseline"].isna()
        year_df.loc[missing, "p_baseline"] = global_p
        year_df.loc[missing, "baseline_source"] = "global"
        year_df["Year"] = year

        rows.append(year_df[SF_COLS + ["Year", "p_baseline", "baseline_source"]])

    baseline_df = pd.concat(rows, ignore_index=True)
    baseline_df["baseline_logit"] = _logit(baseline_df["p_baseline"], eps=eps)

    return baseline_df


def add_residual_logit_column(
    df_observed: pd.DataFrame,
    baseline_df: pd.DataFrame,
    eps: float = 1e-6,
) :
    """
    Merge the expanding-window Species-Family baseline onto every observed
    cell and add:

        baseline_logit    = logit(p^base_{s,f,t})
        residual_logit    = logit(prop_S) - baseline_logit

    residual_logit is what the encoder consumes as input (for years used as
    context) and what the decoder is trained to reconstruct (for years used
    as targets, via final_logit = baseline_logit_{t+1} + delta_logit).
    """
    out = df_observed.merge(
        baseline_df[SF_COLS + ["Year", "p_baseline", "baseline_logit", "baseline_source"]],
        on=SF_COLS + ["Year"],
        how="left",
    )

    if out["p_baseline"].isna().any():
        n_missing = int(out["p_baseline"].isna().sum())
        raise ValueError(
            f"{n_missing} observed cells have no matching expanding-window "
            "baseline row. This should not happen if baseline_df was built "
            "from the same df_observed with "
            "compute_expanding_species_family_baseline."
        )

    out["observed_logit"] = _logit(out["prop_S"], eps=eps)
    out["residual_logit"] = out["observed_logit"] - out["baseline_logit"]

    return out
