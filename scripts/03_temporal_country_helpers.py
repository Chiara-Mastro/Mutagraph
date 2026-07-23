#!/usr/bin/env python3
"""
Reusable helpers for grouped country temporal forecasting.

This module contains data preparation, leakage safe temporal baselines,
expanding beta binomial dispersion fitting, support flags, pair tables, and
metric summaries. It does not run a country holdout experiment on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from src.baselines import fit_global_phi_for_baseline, smoothed_rate
from src.data_completion import load_observed_dataset
from src.temporal_baselines import (
    CSF_COLS,
    SF_COLS,
    predict_ewma_residual,
    predict_locf,
    predict_rolling_mean_k,
    predict_species_family_mean,
    run_walk_forward,
)
from src.training import (
    beta_binomial_nll_from_prob,
    compute_metrics_from_arrays,
)

from src.temporal_residual_features import add_residual_logit_column


EPS = 1e-6


@dataclass(frozen=True)
class PhiParameters:
    raw_phi: float
    phi: float
    rho: float
    source: str
    calibration_max_target_year: int
    calibration_n_cells: int
    calibration_n_tests: int


def prepare_observed_data(path: Path) -> pd.DataFrame:
    df = load_observed_dataset(path).copy()
    required = [
        "Country",
        "Year",
        "Species",
        "Family",
        "n_S",
        "n_total",
        "prop_S",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Input data are missing columns: {missing}")

    for column in ["Country", "Species", "Family"]:
        df[column] = df[column].astype("string").str.strip()

    country_order = {
        country: index
        for index, country in enumerate(
            df["Country"].dropna().astype(str).drop_duplicates().tolist()
        )
    }
    df["country_input_order"] = df["Country"].astype(str).map(country_order)

    df["Year"] = pd.to_numeric(df["Year"], errors="raise").astype(int)
    df["n_S"] = pd.to_numeric(df["n_S"], errors="raise").astype(float)
    df["n_total"] = pd.to_numeric(
        df["n_total"], errors="raise"
    ).astype(float)
    df = df.loc[df["n_total"] > 0].copy()
    df["prop_S"] = df["n_S"] / df["n_total"]

    key = ["Country", "Year", "Species", "Family"]
    duplicated = df.duplicated(key, keep=False)
    if duplicated.any():
        raise ValueError(
            "Duplicate aggregated cells were found. Examples:\n"
            + df.loc[duplicated, key].head(20).to_string(index=False)
        )

    species_values = sorted(df["Species"].astype(str).unique().tolist())
    family_values = sorted(df["Family"].astype(str).unique().tolist())
    species_to_idx = {
        value: index for index, value in enumerate(species_values)
    }
    family_to_idx = {
        value: index for index, value in enumerate(family_values)
    }

    df["species_idx"] = (
        df["Species"].astype(str).map(species_to_idx).astype(int)
    )
    df["family_idx"] = (
        df["Family"].astype(str).map(family_to_idx).astype(int)
    )
    df = df.sort_values(key).reset_index(drop=True)
    df["cell_row_id"] = np.arange(len(df), dtype=int)
    return df


def _ordered_countries(df: pd.DataFrame) -> list[str]:
    if "country_input_order" in df.columns:
        table = (
            df[["Country", "country_input_order"]]
            .drop_duplicates("Country")
            .sort_values("country_input_order")
        )
        return table["Country"].astype(str).tolist()
    return df["Country"].astype(str).drop_duplicates().tolist()


def _normalise_fold_numbers(values: pd.Series, n_folds: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(int)
    unique = sorted(numeric.unique().tolist())
    if unique == list(range(n_folds)):
        return numeric + 1
    if unique == list(range(1, n_folds + 1)):
        return numeric
    raise ValueError(
        "Fold ids must be exactly 0..n_folds-1 or 1..n_folds. "
        f"Observed ids: {unique}"
    )


def build_colleague_country_folds(
    full_df: pd.DataFrame,
    n_folds: int,
    random_state: int = 42,
    folds_path: Path | None = None,
) -> pd.DataFrame:
    ordered_countries = _ordered_countries(full_df)
    expected_countries = set(ordered_countries)

    if folds_path is not None:
        folds_path = Path(folds_path)
        if not folds_path.exists():
            raise FileNotFoundError(f"Country fold file not found: {folds_path}")

        if folds_path.suffix.lower() == ".json":
            import json

            payload = json.loads(folds_path.read_text(encoding="utf-8"))
            rows: list[dict[str, object]] = []
            for raw_fold, countries in payload.items():
                for country in countries:
                    rows.append(
                        {
                            "Country": str(country).strip(),
                            "loaded_fold": int(raw_fold),
                        }
                    )
            loaded = pd.DataFrame(rows)
        else:
            loaded = pd.read_csv(folds_path)
            if "Country" not in loaded.columns:
                raise ValueError("Fold CSV must contain a Country column.")
            fold_candidates = [
                column
                for column in [
                    "colleague_fold",
                    "fold_model",
                    "external_fold",
                    "fold",
                ]
                if column in loaded.columns
            ]
            if not fold_candidates:
                raise ValueError(
                    "Fold CSV must contain one of colleague_fold, fold_model, "
                    "external_fold, or fold."
                )
            loaded = loaded[["Country", fold_candidates[0]]].rename(
                columns={fold_candidates[0]: "loaded_fold"}
            )
            loaded["Country"] = loaded["Country"].astype(str).str.strip()

        if loaded.empty:
            raise ValueError("The supplied fold file contains no assignments.")
        if loaded["Country"].duplicated().any():
            inconsistent = (
                loaded.groupby("Country")["loaded_fold"]
                .nunique()
                .loc[lambda values: values > 1]
            )
            if not inconsistent.empty:
                raise ValueError(
                    "Some countries map to more than one fold: "
                    + str(inconsistent.index.tolist()[:20])
                )
            loaded = loaded.drop_duplicates("Country").copy()

        loaded_countries = set(loaded["Country"])
        missing = sorted(expected_countries - loaded_countries)
        extra = sorted(loaded_countries - expected_countries)
        if missing or extra:
            raise ValueError(
                "The supplied folds do not match the current dataset. "
                f"Missing countries: {missing}; extra countries: {extra}"
            )

        assignment = loaded.copy()
        assignment["fold"] = _normalise_fold_numbers(
            assignment["loaded_fold"], n_folds
        )
        assignment["colleague_fold"] = assignment["fold"] - 1
        assignment["role"] = "external_test"
        assignment["fold_source"] = str(folds_path)
        return assignment[
            [
                "fold",
                "colleague_fold",
                "Country",
                "role",
                "fold_source",
            ]
        ].sort_values(["fold", "Country"]).reset_index(drop=True)

    splitter = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )
    countries_array = np.asarray(ordered_countries, dtype=object)
    rows: list[dict[str, object]] = []
    for colleague_fold, (_, test_indices) in enumerate(
        splitter.split(countries_array)
    ):
        for country in countries_array[test_indices]:
            rows.append(
                {
                    "fold": int(colleague_fold + 1),
                    "colleague_fold": int(colleague_fold),
                    "Country": str(country),
                    "role": "external_test",
                    "fold_source": (
                        "reproduced_KFold_on_country_first_occurrence_order"
                    ),
                }
            )
    assignment = pd.DataFrame(rows)
    if set(assignment["Country"]) != expected_countries:
        raise AssertionError("Not all countries were assigned to a fold.")
    return assignment.sort_values(["fold", "Country"]).reset_index(drop=True)


def sqrt_n_error_metrics(df: pd.DataFrame) -> dict[str, float]:
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["prop_S", "p_pred", "n_total"]
    )
    clean = clean.loc[clean["n_total"] > 0].copy()
    if clean.empty:
        return {
            "weighted_mae": float("nan"),
            "weighted_rmse": float("nan"),
            "sqrt_n_weighted_mae": float("nan"),
            "sqrt_n_weighted_rmse": float("nan"),
            "error_weighting": "sqrt_n_total",
        }
    error = (
        clean["prop_S"].to_numpy(dtype=float)
        - clean["p_pred"].to_numpy(dtype=float)
    )
    weights = np.sqrt(clean["n_total"].to_numpy(dtype=float))
    mae = float(np.average(np.abs(error), weights=weights))
    rmse = float(math.sqrt(np.average(error**2, weights=weights)))
    return {
        "weighted_mae": mae,
        "weighted_rmse": rmse,
        "sqrt_n_weighted_mae": mae,
        "sqrt_n_weighted_rmse": rmse,
        "error_weighting": "sqrt_n_total",
    }


def consecutive_pairs(df: pd.DataFrame) -> list[tuple[str, int, int]]:
    pairs: list[tuple[str, int, int]] = []
    years_by_country = df.groupby("Country")["Year"].apply(
        lambda values: sorted(set(map(int, values)))
    )
    for country, years in years_by_country.items():
        year_set = set(years)
        for year in years:
            if year + 1 in year_set:
                pairs.append((str(country), int(year), int(year + 1)))
    return pairs


def compute_train_only_expanding_baseline(
    training_country_df: pd.DataFrame,
    target_universe_df: pd.DataFrame,
    alpha: float,
    beta: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    years = sorted(
        pd.to_numeric(target_universe_df["Year"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )

    for year in years:
        history = training_country_df.loc[
            training_country_df["Year"] < year
        ].copy()

        if history.empty:
            global_p = 0.5
            sf = pd.DataFrame(columns=SF_COLS + ["p_baseline"])
        else:
            global_p = float(
                smoothed_rate(
                    history,
                    group_cols=None,
                    alpha=alpha,
                    beta=beta,
                )
            )
            sf = smoothed_rate(
                history,
                group_cols=SF_COLS,
                alpha=alpha,
                beta=beta,
            ).rename(columns={"p_hat": "p_baseline"})

        if not sf.empty:
            sf["baseline_source"] = "training_country_species_family"

        requested = (
            target_universe_df.loc[
                target_universe_df["Year"].eq(year), SF_COLS
            ]
            .drop_duplicates()
            .copy()
        )
        year_table = requested.merge(sf, on=SF_COLS, how="left")
        missing = year_table["p_baseline"].isna()
        year_table.loc[missing, "p_baseline"] = global_p
        year_table.loc[missing, "baseline_source"] = (
            "training_country_global"
        )
        year_table["Year"] = int(year)
        clipped = np.clip(
            year_table["p_baseline"].to_numpy(dtype=float),
            EPS,
            1.0 - EPS,
        )
        year_table["baseline_logit"] = np.log(clipped / (1.0 - clipped))
        year_table["baseline_history_min_year"] = (
            int(history["Year"].min()) if not history.empty else np.nan
        )
        year_table["baseline_history_max_year"] = (
            int(history["Year"].max()) if not history.empty else np.nan
        )
        year_table["baseline_n_history_years"] = int(
            history["Year"].nunique()
        )
        year_table["baseline_n_history_cells"] = int(len(history))
        year_table["baseline_n_history_tests"] = float(
            history["n_total"].sum()
        )
        year_table["baseline_training_cutoff_exclusive"] = int(year)
        year_table["baseline_excludes_external_fold"] = True
        rows.append(year_table)

    if not rows:
        raise ValueError("No yearly baseline tables were created.")
    return pd.concat(rows, ignore_index=True)


def training_only_fallback_predictions(
    training_history: pd.DataFrame,
    target_df: pd.DataFrame,
    alpha: float,
    beta: float,
) -> pd.DataFrame:
    if training_history.empty:
        out = target_df.copy()
        out["p_pred"] = 0.5
        out["baseline_source"] = "training_global_empty_history"
        return out

    global_p = float(
        smoothed_rate(
            training_history,
            group_cols=None,
            alpha=alpha,
            beta=beta,
        )
    )
    return predict_species_family_mean(
        history_df=training_history,
        target_df=target_df,
        alpha=alpha,
        beta=beta,
        global_p=global_p,
    )


def replace_local_fallback(
    local_prediction: pd.DataFrame,
    training_fallback: pd.DataFrame,
    local_source_prefix: str,
) -> pd.DataFrame:
    keys = ["Country", "Year", "Species", "Family"]
    fallback = training_fallback[
        keys + ["p_pred", "baseline_source"]
    ].rename(
        columns={
            "p_pred": "training_fallback_p",
            "baseline_source": "training_fallback_source",
        }
    )
    out = local_prediction.merge(
        fallback,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    use_local = out["baseline_source"].astype(str).str.startswith(
        local_source_prefix
    )
    out.loc[~use_local, "p_pred"] = out.loc[
        ~use_local, "training_fallback_p"
    ]
    out.loc[~use_local, "baseline_source"] = (
        "training_"
        + out.loc[~use_local, "training_fallback_source"].astype(str)
    )
    return out.drop(
        columns=["training_fallback_p", "training_fallback_source"]
    )


def predict_external_ewma(
    training_history: pd.DataFrame,
    local_history: pd.DataFrame,
    target_df: pd.DataFrame,
    alpha: float,
    beta: float,
    halflife_years: float,
) -> pd.DataFrame:
    if halflife_years <= 0:
        raise ValueError("halflife_years must be positive.")

    fallback = training_only_fallback_predictions(
        training_history=training_history,
        target_df=target_df,
        alpha=alpha,
        beta=beta,
    )
    if local_history.empty:
        out = fallback.copy()
        out["baseline_source"] = (
            "training_" + out["baseline_source"].astype(str)
        )
        return out

    target_year = int(target_df["Year"].iloc[0])
    if training_history.empty:
        global_p = 0.5
        sf = pd.DataFrame(columns=SF_COLS + ["p_species_family"])
    else:
        global_p = float(
            smoothed_rate(
                training_history,
                group_cols=None,
                alpha=alpha,
                beta=beta,
            )
        )
        sf = smoothed_rate(
            training_history,
            group_cols=SF_COLS,
            alpha=alpha,
            beta=beta,
        ).rename(columns={"p_hat": "p_species_family"})

    hist = local_history.merge(sf, on=SF_COLS, how="left")
    hist["p_species_family"] = hist["p_species_family"].fillna(global_p)
    hist["residual"] = hist["prop_S"] - hist["p_species_family"]
    hist["weight"] = 0.5 ** (
        (target_year - hist["Year"]) / float(halflife_years)
    )
    hist["weighted_residual"] = hist["weight"] * hist["residual"]

    residual_table = (
        hist.groupby(CSF_COLS, as_index=False)
        .agg(
            weighted_residual_sum=("weighted_residual", "sum"),
            weight_sum=("weight", "sum"),
            ewma_history_min_year=("Year", "min"),
            ewma_history_max_year=("Year", "max"),
            ewma_history_n_years=("Year", "nunique"),
        )
    )
    residual_table["ewma_residual"] = (
        residual_table["weighted_residual_sum"]
        / residual_table["weight_sum"]
    )
    residual_table = residual_table.drop(
        columns=["weighted_residual_sum", "weight_sum"]
    )

    base = fallback[
        ["Country", "Year", "Species", "Family", "p_pred"]
    ].rename(columns={"p_pred": "training_p_baseline"})
    out = target_df.merge(
        residual_table,
        on=CSF_COLS,
        how="left",
    ).merge(
        base,
        on=["Country", "Year", "Species", "Family"],
        how="left",
        validate="one_to_one",
    )
    has_residual = out["ewma_residual"].notna()
    out["p_pred"] = out["training_p_baseline"]
    out.loc[has_residual, "p_pred"] = np.clip(
        out.loc[has_residual, "training_p_baseline"]
        + out.loc[has_residual, "ewma_residual"],
        EPS,
        1.0 - EPS,
    )
    out["baseline_source"] = "training_species_family"
    out.loc[has_residual, "baseline_source"] = (
        "external_country_ewma_residual"
    )
    return out.drop(columns=["training_p_baseline"])


def external_baseline_prediction_for_year(
    method_name: str,
    training_history: pd.DataFrame,
    local_history: pd.DataFrame,
    target_df: pd.DataFrame,
    alpha: float,
    beta: float,
    rolling_k: int,
    ewma_halflife_years: float,
) -> pd.DataFrame:
    training_fallback = training_only_fallback_predictions(
        training_history=training_history,
        target_df=target_df,
        alpha=alpha,
        beta=beta,
    )

    if method_name == "species_family_mean":
        out = training_fallback.copy()
        out["baseline_source"] = (
            "training_" + out["baseline_source"].astype(str)
        )
        return out

    if method_name == "ewma_residual":
        return predict_external_ewma(
            training_history=training_history,
            local_history=local_history,
            target_df=target_df,
            alpha=alpha,
            beta=beta,
            halflife_years=ewma_halflife_years,
        )

    if local_history.empty:
        out = training_fallback.copy()
        out["baseline_source"] = (
            "training_" + out["baseline_source"].astype(str)
        )
        return out

    global_local = float(
        smoothed_rate(
            local_history,
            group_cols=None,
            alpha=alpha,
            beta=beta,
        )
    )

    if method_name == "locf":
        local = predict_locf(
            history_df=local_history,
            target_df=target_df,
            alpha=alpha,
            beta=beta,
            global_p=global_local,
        )
        return replace_local_fallback(
            local,
            training_fallback,
            local_source_prefix="locf",
        )

    if method_name == "rolling_mean_k":
        local = predict_rolling_mean_k(
            history_df=local_history,
            target_df=target_df,
            alpha=alpha,
            beta=beta,
            global_p=global_local,
            k=rolling_k,
        )
        return replace_local_fallback(
            local,
            training_fallback,
            local_source_prefix=f"rolling_mean_k{rolling_k}",
        )

    raise ValueError(f"Unknown baseline method: {method_name}")


def baseline_calibration_stream(
    training_country_df: pd.DataFrame,
    method_name: str,
    args,
) -> pd.DataFrame:
    specs: dict[str, tuple[Callable, dict[str, object]]] = {
        "species_family_mean": (predict_species_family_mean, {}),
        "locf": (predict_locf, {}),
        "rolling_mean_k": (
            predict_rolling_mean_k,
            {"k": args.rolling_k},
        ),
        "ewma_residual": (
            predict_ewma_residual,
            {"halflife_years": args.ewma_halflife_years},
        ),
    }
    if method_name not in specs:
        raise ValueError(f"Unknown baseline method: {method_name}")
    predict_fn, extra = specs[method_name]
    return run_walk_forward(
        df_observed=training_country_df,
        predict_fn=predict_fn,
        min_train_years=args.phi_calibration_min_train_years,
        alpha=args.alpha,
        beta=args.beta,
        method_name=method_name,
        **extra,
    )


def fit_training_phi_for_target(
    calibration_stream: pd.DataFrame,
    target_year: int,
    method_name: str,
    min_cells: int,
    min_tests: int,
) -> PhiParameters:
    calibration = calibration_stream.loc[
        calibration_stream["target_year"] < target_year
    ].copy()
    n_cells = int(len(calibration))
    n_tests = int(calibration["n_total"].sum()) if n_cells else 0
    if n_cells < min_cells or n_tests < min_tests:
        raise ValueError(
            f"Insufficient training country phi calibration for "
            f"{method_name}, target year {target_year}: "
            f"{n_cells} cells and {n_tests} tests."
        )
    max_year = int(calibration["target_year"].max())
    fit = fit_global_phi_for_baseline(
        df=calibration,
        pred_col="p_pred",
    )
    phi = float(fit["phi"])
    return PhiParameters(
        raw_phi=float(fit["log_phi"]),
        phi=phi,
        rho=float(1.0 / (phi + 1.0)),
        source="training_countries_expanding_prequential_errors",
        calibration_max_target_year=max_year,
        calibration_n_cells=n_cells,
        calibration_n_tests=n_tests,
    )


def input_support_flags(
    external_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    previous = external_df.loc[
        external_df["Year"].eq(target_year - 1), SF_COLS
    ].drop_duplicates()
    previous_pairs = set(zip(previous["Species"], previous["Family"]))
    history = external_df.loc[external_df["Year"] < target_year, SF_COLS]
    history_pairs = set(zip(history["Species"], history["Family"]))

    flags = target_df[["Country", "Year", "Species", "Family"]].copy()
    flags["target_cell_seen_in_input_year"] = [
        (species, family) in previous_pairs
        for species, family in zip(flags["Species"], flags["Family"])
    ]
    flags["target_cell_seen_in_any_local_history"] = [
        (species, family) in history_pairs
        for species, family in zip(flags["Species"], flags["Family"])
    ]
    return flags


def training_support_flags(
    training_country_df: pd.DataFrame,
    predictions: pd.DataFrame,
    train_max_target_year: int,
) -> pd.DataFrame:
    fitting_rows = training_country_df.loc[
        training_country_df["Year"] <= train_max_target_year
    ]
    species = set(fitting_rows["Species"].astype(str))
    families = set(fitting_rows["Family"].astype(str))
    pairs = set(
        zip(
            fitting_rows["Species"].astype(str),
            fitting_rows["Family"].astype(str),
        )
    )
    out = predictions.copy()
    out["species_seen_in_neural_training"] = (
        out["Species"].astype(str).isin(species)
    )
    out["family_seen_in_neural_training"] = (
        out["Family"].astype(str).isin(families)
    )
    out["species_family_seen_in_neural_training"] = [
        (str(species_name), str(family_name)) in pairs
        for species_name, family_name in zip(
            out["Species"], out["Family"]
        )
    ]
    out["both_entities_seen_in_neural_training"] = (
        out["species_seen_in_neural_training"]
        & out["family_seen_in_neural_training"]
    )
    return out


def build_temporal_residual_pair_table(
    train_pairs: list[tuple[str, int, int]],
    val_pairs: list[tuple[str, int, int]],
    test_pairs: list[tuple[str, int, int]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, pairs in [
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ]:
        for country, input_year, target_year in pairs:
            rows.append(
                {
                    "pair_id": f"{country}||{input_year}||{target_year}",
                    "Country": country,
                    "input_year": input_year,
                    "target_year": target_year,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def beta_binomial_nll_for_predictions(df: pd.DataFrame) -> float:
    total_nll = 0.0
    total_tests = 0.0
    group_columns = ["target_year", "raw_phi"]
    for _, group in df.groupby(group_columns, dropna=False):
        raw_values = group["raw_phi"].dropna().unique()
        if len(raw_values) != 1:
            return float("nan")
        raw_phi = float(raw_values[0])
        nll_sum = beta_binomial_nll_from_prob(
            p=group["p_pred"].to_numpy(dtype=float),
            n_s=group["n_S"].to_numpy(dtype=float),
            n_total=group["n_total"].to_numpy(dtype=float),
            log_phi=torch.tensor(raw_phi, dtype=torch.float32),
            reduction="sum",
        )
        total_nll += float(nll_sum.detach().cpu())
        total_tests += float(group["n_total"].sum())
    return total_nll / max(total_tests, 1.0)


def summarize_predictions(
    predictions: pd.DataFrame,
    group_columns: list[str],
    evaluation_set: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        metrics = compute_metrics_from_arrays(
            prop_s=group["prop_S"].to_numpy(dtype=float),
            pred_p=group["p_pred"].to_numpy(dtype=float),
            n_s=group["n_S"].to_numpy(dtype=float),
            n_total=group["n_total"].to_numpy(dtype=float),
        )
        row["n_weighted_mae"] = metrics.get("weighted_mae", float("nan"))
        row["n_weighted_rmse"] = metrics.get("weighted_rmse", float("nan"))
        row["n_weighted_signed_error_obs_minus_pred"] = metrics.get(
            "weighted_signed_error_obs_minus_pred", float("nan")
        )
        row.update(metrics)
        sqrt_metrics = sqrt_n_error_metrics(group)
        row.update(sqrt_metrics)
        row["beta_binomial_nll_per_test"] = (
            beta_binomial_nll_for_predictions(group)
        )
        row["evaluation_set"] = evaluation_set
        row["n_countries"] = int(group["Country"].nunique())
        row["n_target_years"] = int(group["target_year"].nunique())
        row["n_seen_in_input_year"] = int(
            group["target_cell_seen_in_input_year"].fillna(False).sum()
        )
        row["fraction_seen_in_input_year"] = float(
            group["target_cell_seen_in_input_year"].fillna(False).mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)
