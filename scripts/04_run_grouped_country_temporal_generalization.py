#!/usr/bin/env python3
"""
Five fold grouped country temporal generalisation with two external tests.

For each fold supplied by the GNN country assignment file, one temporal model
is trained only on the other four folds and only on observations through 2022.
Gradient training uses transition targets before 2022. Target year 2022 from
training countries is used only for checkpoint selection.

The selected fold model is frozen and evaluated twice on the countries assigned
to that fold:

1. country generalisation on consecutive historical transitions with target
   year through 2022
2. country and year generalisation on target years 2023 and 2024 only

The same fitted model, prior, dispersion and baseline parameters are used for
both tests. The five fold uncertainty reported by this script is the sample
standard deviation of the five fold level metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as beta_distribution

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT_FROM_SCRIPT))

from src.CONFIG import EPS
from src.models import SnapshotEncoderResidualModel
from src.temporal_dynamics_dataset import (
    build_temporal_residual_encoder_dataloaders,
)
from src.training import (
    evaluate_temporal_residual_loader,
    train_one_epoch_temporal_residual,
)
from src.utils import choose_device, set_seed


BASELINE_METHODS = [
    "species_family_mean",
    "locf",
    "rolling_mean_k",
    "ewma_residual",
]
MODEL_GROUPS = ["baselines", "temporal_residual"]
TRAIN_MAX_YEAR = 2022
VALIDATION_TARGET_YEAR = 2022
TEST_YEARS = (2023, 2024)
EXPECTED_N_FOLDS = 5
EVALUATION_SET_COUNTRY = "historical_country_generalization_through_2022"
EVALUATION_SET_COUNTRY_YEAR = "country_and_year_generalization_2023_2024"
EVALUATION_SET_FUTURE = "prospective_forecast"
EVALUATION_PROTOCOL = (
    "gnn_country_five_fold_historical_and_2023_2024_external_tests"
)


def load_temporal_helpers():
    helper_path = Path(__file__).with_name(
        "03_temporal_country_helpers.py"
    )
    if not helper_path.exists():
        raise FileNotFoundError(
            "Expected helper script next to this file: " + str(helper_path)
        )

    spec = importlib.util.spec_from_file_location(
        "temporal_country_helpers",
        helper_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load temporal country helpers.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


HELPER = load_temporal_helpers()


TEMPORAL_CELL_INTEGER_COLUMNS = [
    "Year",
    "species_idx",
    "family_idx",
    "cell_row_id",
]
TEMPORAL_CELL_FLOAT_COLUMNS = [
    "n_S",
    "n_total",
    "prop_S",
    "p_baseline",
    "baseline_logit",
    "residual_prop_S",
    "residual_logit",
]
TEMPORAL_PAIR_INTEGER_COLUMNS = ["input_year", "target_year"]


def _coerce_integer_column(
    frame: pd.DataFrame,
    column: str,
    *,
    table_name: str,
) -> None:
    if column not in frame.columns:
        raise ValueError(
            f"{table_name} is missing required integer column {column!r}."
        )
    original = frame[column]
    numeric = pd.to_numeric(original, errors="coerce")
    invalid = numeric.isna()
    if invalid.any():
        examples = original.loc[invalid].head(10).tolist()
        raise ValueError(
            f"{table_name}.{column} contains missing or non numeric values: "
            f"{examples}"
        )
    values = numeric.to_numpy(dtype=np.float64)
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=0.0):
        bad = values[~np.isclose(values, rounded, rtol=0.0, atol=0.0)][:10]
        raise ValueError(
            f"{table_name}.{column} contains non integral values: "
            f"{bad.tolist()}"
        )
    frame[column] = rounded.astype(np.int64)


def _coerce_float_column(
    frame: pd.DataFrame,
    column: str,
    *,
    table_name: str,
) -> None:
    if column not in frame.columns:
        return
    original = frame[column]
    numeric = pd.to_numeric(original, errors="coerce")
    invalid = numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=np.float64))
    if invalid.any():
        examples = original.loc[invalid].head(10).tolist()
        raise ValueError(
            f"{table_name}.{column} contains missing, non numeric, or infinite "
            f"values: {examples}"
        )
    frame[column] = numeric.to_numpy(dtype=np.float32)


def coerce_temporal_model_inputs(
    cells: pd.DataFrame,
    pair_table: pd.DataFrame,
    *,
    fold: int,
    target_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return native NumPy backed dtypes before the PyTorch dataset sees them.

    Pandas nullable and object columns can still contain valid looking numbers,
    but ``Series.to_numpy()`` then yields an object array that PyTorch refuses.
    This conversion is deliberately performed immediately before construction of
    every fixed vault DataLoader.
    """

    clean_cells = cells.copy()
    clean_pairs = pair_table.copy()
    context = f"fold {fold}, target year {target_year}"

    for column in TEMPORAL_CELL_INTEGER_COLUMNS:
        _coerce_integer_column(
            clean_cells, column, table_name=f"cells ({context})"
        )
    for column in TEMPORAL_CELL_FLOAT_COLUMNS:
        _coerce_float_column(
            clean_cells, column, table_name=f"cells ({context})"
        )

    for column in TEMPORAL_PAIR_INTEGER_COLUMNS:
        _coerce_integer_column(
            clean_pairs, column, table_name=f"pairs ({context})"
        )

    for column in ["Country", "pair_id", "split"]:
        if column not in clean_pairs.columns:
            raise ValueError(
                f"pairs ({context}) is missing required text column {column!r}."
            )
        if clean_pairs[column].isna().any():
            raise ValueError(
                f"pairs ({context}).{column} contains missing values."
            )
        clean_pairs[column] = clean_pairs[column].astype(str)

    for column in ["Country", "Species", "Family"]:
        if column not in clean_cells.columns:
            raise ValueError(
                f"cells ({context}) is missing required text column {column!r}."
            )
        if clean_cells[column].isna().any():
            raise ValueError(
                f"cells ({context}).{column} contains missing values."
            )
        clean_cells[column] = clean_cells[column].astype(str)

    numeric_columns = (
        TEMPORAL_CELL_INTEGER_COLUMNS + TEMPORAL_CELL_FLOAT_COLUMNS
    )
    remaining_object_numeric = [
        column
        for column in numeric_columns
        if column in clean_cells.columns
        and clean_cells[column].to_numpy().dtype == object
    ]
    if remaining_object_numeric:
        raise TypeError(
            f"Native dtype conversion failed for {context}: "
            f"{remaining_object_numeric}. Dtypes are "
            f"{clean_cells[remaining_object_numeric].dtypes.astype(str).to_dict()}"
        )

    return clean_cells, clean_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one temporal model per GNN country fold through 2022, then "
            "evaluate the same frozen model on historical external countries "
            "and on external countries in 2023 and 2024."
        )
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--country_folds_path",
        type=Path,
        required=True,
        help="Exact five fold country assignment used by the GNN.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_GROUPS,
        default=MODEL_GROUPS,
    )

    parser.add_argument("--forecast_input_year", type=int, default=2024)
    parser.add_argument("--forecast_year", type=int, default=2025)
    parser.add_argument(
        "--forecast_target_universe",
        choices=[
            "input_year_cells",
            "country_history_cells",
            "training_species_family_cells",
        ],
        default="input_year_cells",
    )

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--rolling_k", type=int, default=3)
    parser.add_argument("--ewma_halflife_years", type=float, default=2.0)
    parser.add_argument("--phi_calibration_min_train_years", type=int, default=1)
    parser.add_argument("--min_phi_calibration_cells", type=int, default=1)
    parser.add_argument("--min_phi_calibration_tests", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--min_input_cells", type=int, default=1)
    parser.add_argument("--min_target_cells", type=int, default=1)
    parser.add_argument("--entity_emb_dim", type=int, default=16)
    parser.add_argument("--edge_hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=12)
    parser.add_argument("--decoder_hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--gradient_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--save_models",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run_future_neural",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--continue_on_error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()
    if args.forecast_year <= args.forecast_input_year:
        raise ValueError("forecast_year must be after forecast_input_year.")
    if args.forecast_year != args.forecast_input_year + 1:
        raise ValueError("This script supports one year forecasts only.")
    if args.forecast_input_year < max(TEST_YEARS):
        raise ValueError("forecast_input_year must be at least 2024.")
    return args

def build_grouped_country_folds(
    full_df: pd.DataFrame,
    folds_path: Path,
) -> pd.DataFrame:
    folds = HELPER.build_colleague_country_folds(
        full_df=full_df,
        n_folds=EXPECTED_N_FOLDS,
        random_state=42,
        folds_path=folds_path,
    )
    observed = sorted(folds["fold"].astype(int).unique().tolist())
    expected = list(range(1, EXPECTED_N_FOLDS + 1))
    if observed != expected:
        raise ValueError(
            f"The supplied GNN fold file resolves to folds {observed}, expected {expected}."
        )
    return folds

def split_pairs_for_dual_evaluation(
    all_pairs: list[tuple[str, int, int]],
    training_countries: set[str],
    test_countries: set[str],
) -> tuple[
    list[tuple[str, int, int]],
    list[tuple[str, int, int]],
    list[tuple[str, int, int]],
    list[tuple[str, int, int]],
]:
    train_pairs = [
        pair
        for pair in all_pairs
        if pair[0] in training_countries
        and pair[2] < VALIDATION_TARGET_YEAR
    ]
    val_pairs = [
        pair
        for pair in all_pairs
        if pair[0] in training_countries
        and pair[2] == VALIDATION_TARGET_YEAR
    ]
    historical_test_pairs = [
        pair
        for pair in all_pairs
        if pair[0] in test_countries
        and pair[2] <= TRAIN_MAX_YEAR
    ]
    country_year_test_pairs = [
        pair
        for pair in all_pairs
        if pair[0] in test_countries
        and pair[2] in TEST_YEARS
    ]

    if not train_pairs:
        raise ValueError("No gradient training pairs with target year before 2022.")
    if not val_pairs:
        raise ValueError("No checkpoint selection pairs targeting 2022.")
    if not historical_test_pairs:
        raise ValueError("No historical external country test pairs through 2022.")
    if not country_year_test_pairs:
        raise ValueError("No external country and year test pairs for 2023 or 2024.")

    observed_vault_years = {pair[2] for pair in country_year_test_pairs}
    missing_vault_years = sorted(set(TEST_YEARS) - observed_vault_years)
    if missing_vault_years:
        raise ValueError(
            f"The external fold is missing test transitions for {missing_vault_years}."
        )

    all_selected = (
        train_pairs
        + val_pairs
        + historical_test_pairs
        + country_year_test_pairs
    )
    for country, input_year, target_year in all_selected:
        if input_year != target_year - 1:
            raise AssertionError(
                f"Nonconsecutive pair for {country}: {input_year}, {target_year}."
            )
    return train_pairs, val_pairs, historical_test_pairs, country_year_test_pairs


def evaluation_set_from_target_year(target_year: int, target_observed: bool = True) -> str:
    if not target_observed:
        return EVALUATION_SET_FUTURE
    year = int(target_year)
    if year <= TRAIN_MAX_YEAR:
        return EVALUATION_SET_COUNTRY
    if year in TEST_YEARS:
        return EVALUATION_SET_COUNTRY_YEAR
    raise ValueError(f"Observed target year {year} belongs to neither evaluation task.")

def build_fold_cells(
    full_df: pd.DataFrame,
    training_countries: set[str],
    alpha: float,
    beta: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_df = full_df.loc[
        full_df["Country"].astype(str).isin(training_countries)
        & full_df["Year"].le(TRAIN_MAX_YEAR)
    ].copy()
    if training_df.empty:
        raise ValueError("The fixed historical training dataframe is empty.")
    if training_df["Year"].gt(TRAIN_MAX_YEAR).any():
        raise AssertionError("Training rows exceed the fixed 2022 cutoff.")

    baseline = HELPER.compute_train_only_expanding_baseline(
        training_country_df=training_df,
        target_universe_df=full_df,
        alpha=alpha,
        beta=beta,
    )

    original_ids = full_df[
        [
            "Country",
            "Year",
            "Species",
            "Family",
            "species_idx",
            "family_idx",
            "cell_row_id",
        ]
    ].copy()
    base_input = full_df.drop(
        columns=["species_idx", "family_idx", "cell_row_id"]
    )
    cells = HELPER.add_residual_logit_column(
        df_observed=base_input,
        baseline_df=baseline,
    )
    cells = cells.merge(
        original_ids,
        on=["Country", "Year", "Species", "Family"],
        how="left",
        validate="one_to_one",
    )
    provenance = baseline[
        [
            "Species",
            "Family",
            "Year",
            "baseline_history_min_year",
            "baseline_history_max_year",
            "baseline_n_history_years",
            "baseline_n_history_cells",
            "baseline_n_history_tests",
            "baseline_training_cutoff_exclusive",
        ]
    ]
    cells = cells.merge(
        provenance,
        on=["Species", "Family", "Year"],
        how="left",
        validate="many_to_one",
    )
    cells["residual_prop_S"] = cells["prop_S"] - cells["p_baseline"]
    cells["baseline_excludes_external_fold"] = True
    cells["baseline_training_max_year"] = int(TRAIN_MAX_YEAR)

    vault_rows = cells["Year"].isin(TEST_YEARS)
    bad_vault_baseline = (
        vault_rows
        & cells["baseline_history_max_year"].notna()
        & cells["baseline_history_max_year"].gt(TRAIN_MAX_YEAR)
    )
    if bad_vault_baseline.any():
        raise AssertionError(
            "A vault baseline used training history after the 2022 cutoff."
        )
    return cells, training_df


def build_forecast_candidates(
    full_df: pd.DataFrame,
    training_df: pd.DataFrame,
    country: str,
    input_year: int,
    forecast_year: int,
    universe_mode: str,
) -> pd.DataFrame:
    country_df = full_df.loc[full_df["Country"].eq(country)].copy()

    if universe_mode == "input_year_cells":
        source = country_df.loc[country_df["Year"].eq(input_year)].copy()
        source_label = "country_input_year_cells"
    elif universe_mode == "country_history_cells":
        source = country_df.loc[country_df["Year"] <= input_year].copy()
        source_label = "country_history_cells"
    elif universe_mode == "training_species_family_cells":
        source = training_df.loc[training_df["Year"] <= input_year].copy()
        source_label = "training_species_family_cells"
    else:
        raise ValueError(universe_mode)

    if source.empty:
        return pd.DataFrame()

    target = (
        source[["Species", "Family", "species_idx", "family_idx"]]
        .drop_duplicates(["Species", "Family"])
        .copy()
    )
    target.insert(0, "Country", country)
    target["Year"] = int(forecast_year)
    target["n_S"] = np.nan
    target["n_total"] = np.nan
    target["prop_S"] = np.nan
    target["target_observed"] = False
    target["target_candidate_source"] = source_label
    target["forecast_input_year"] = int(input_year)
    return target.reset_index(drop=True)


def add_fold_baseline_to_candidates(
    candidates: pd.DataFrame,
    training_df: pd.DataFrame,
    alpha: float,
    beta: float,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    baseline = HELPER.compute_train_only_expanding_baseline(
        training_country_df=training_df,
        target_universe_df=candidates,
        alpha=alpha,
        beta=beta,
    )
    provenance_columns = [
        "Species",
        "Family",
        "Year",
        "p_baseline",
        "baseline_logit",
        "baseline_source",
        "baseline_history_min_year",
        "baseline_history_max_year",
        "baseline_n_history_years",
        "baseline_n_history_cells",
        "baseline_n_history_tests",
        "baseline_training_cutoff_exclusive",
    ]
    out = candidates.merge(
        baseline[provenance_columns],
        on=["Species", "Family", "Year"],
        how="left",
        validate="many_to_one",
    )
    if out["p_baseline"].isna().any():
        raise ValueError("Some forecast candidates lack a fold baseline.")
    return out


def add_fold_metadata(
    predictions: pd.DataFrame,
    fold: int,
    model_name: str,
    n_training_countries: int,
    target_observed: bool,
) -> pd.DataFrame:
    out = predictions.copy()
    out["fold"] = int(fold)
    out["colleague_fold"] = int(fold - 1)
    out["held_out_country"] = out["Country"].astype(str)
    out["model_name"] = model_name
    out["method"] = model_name
    out["evaluation_protocol"] = EVALUATION_PROTOCOL
    out["country_seen_in_parameter_fitting"] = False
    out["n_training_countries"] = int(n_training_countries)
    out["target_observed"] = bool(target_observed)
    out["evaluation_set"] = [
        evaluation_set_from_target_year(year, bool(target_observed))
        for year in out["target_year"]
    ]
    out["dataset_role"] = out["evaluation_set"]
    out["training_data_max_year"] = int(TRAIN_MAX_YEAR)
    out["checkpoint_selection_target_year"] = int(VALIDATION_TARGET_YEAR)
    out["country_year_test_years"] = ",".join(map(str, TEST_YEARS))
    out["target_outcome_used_as_input"] = False
    out["target_outcome_used_as_model_input"] = False
    out["target_outcome_used_for_training"] = False
    out["target_outcome_used_for_checkpoint_selection"] = False
    out["prediction_id"] = (
        out["Country"].astype(str)
        + "||"
        + out["Species"].astype(str)
        + "||"
        + out["Family"].astype(str)
        + "||"
        + out["target_year"].astype(str)
    )
    return out

def add_beta_latent_intervals(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add latent Beta quantiles and explicit 95 percent brackets.

    The bracket describes the latent susceptibility rate. It does not include
    the additional finite sample Binomial variation associated with n_total.
    """

    out = predictions.copy()
    alpha = pd.to_numeric(out["bb_alpha"], errors="coerce")
    beta = pd.to_numeric(out["bb_beta"], errors="coerce")
    valid = (
        np.isfinite(alpha.to_numpy(dtype=float))
        & np.isfinite(beta.to_numpy(dtype=float))
        & alpha.gt(0).to_numpy()
        & beta.gt(0).to_numpy()
    )

    quantile_columns = [
        "beta_latent_q025",
        "beta_latent_q05",
        "beta_latent_q50",
        "beta_latent_q95",
        "beta_latent_q975",
        "latent_rate_ci_lo_95",
        "latent_rate_ci_hi_95",
        "latent_rate_ci_width_95",
        "latent_rate_sd",
    ]
    for column in quantile_columns:
        out[column] = np.nan

    if np.any(valid):
        alpha_values = alpha.to_numpy(dtype=float)[valid]
        beta_values = beta.to_numpy(dtype=float)[valid]
        total = alpha_values + beta_values

        q025 = beta_distribution.ppf(0.025, alpha_values, beta_values)
        q05 = beta_distribution.ppf(0.05, alpha_values, beta_values)
        q50 = beta_distribution.ppf(0.50, alpha_values, beta_values)
        q95 = beta_distribution.ppf(0.95, alpha_values, beta_values)
        q975 = beta_distribution.ppf(0.975, alpha_values, beta_values)
        sd = np.sqrt(
            (alpha_values * beta_values)
            / (total**2 * (total + 1.0))
        )

        out.loc[valid, "beta_latent_q025"] = q025
        out.loc[valid, "beta_latent_q05"] = q05
        out.loc[valid, "beta_latent_q50"] = q50
        out.loc[valid, "beta_latent_q95"] = q95
        out.loc[valid, "beta_latent_q975"] = q975
        out.loc[valid, "latent_rate_ci_lo_95"] = q025
        out.loc[valid, "latent_rate_ci_hi_95"] = q975
        out.loc[valid, "latent_rate_ci_width_95"] = q975 - q025
        out.loc[valid, "latent_rate_sd"] = sd

    out["latent_rate_interval_level"] = 0.95
    out["latent_rate_interval_distribution"] = "beta"
    out["latent_rate_interval_includes_binomial_sampling"] = False
    return out


def run_baselines_for_fold(
    full_df: pd.DataFrame,
    training_df: pd.DataFrame,
    test_countries: set[str],
    fold: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if training_df["Year"].gt(TRAIN_MAX_YEAR).any():
        raise AssertionError("Baseline fitting rows exceed the 2022 cutoff.")

    calibration_streams = {
        method: HELPER.baseline_calibration_stream(
            training_country_df=training_df,
            method_name=method,
            args=args,
        )
        for method in BASELINE_METHODS
    }
    fixed_phi = {
        method: HELPER.fit_training_phi_for_target(
            calibration_stream=calibration_streams[method],
            target_year=VALIDATION_TARGET_YEAR,
            method_name=method,
            min_cells=args.min_phi_calibration_cells,
            min_tests=args.min_phi_calibration_tests,
        )
        for method in BASELINE_METHODS
    }

    outputs: list[pd.DataFrame] = []
    exclusions: list[dict[str, object]] = []

    for country in sorted(test_countries):
        external_df = full_df.loc[full_df["Country"].eq(country)].copy()
        external_pairs = HELPER.consecutive_pairs(external_df)
        observed_target_years = sorted(
            {
                int(target_year)
                for _, _, target_year in external_pairs
                if target_year <= TRAIN_MAX_YEAR or target_year in TEST_YEARS
            }
        )
        target_specs: list[tuple[int, pd.DataFrame, bool]] = []

        for target_year in observed_target_years:
            observed_target = external_df.loc[
                external_df["Year"].eq(target_year)
            ].copy()
            if observed_target.empty:
                exclusions.append(
                    {
                        "fold": fold,
                        "Country": country,
                        "target_year": target_year,
                        "stage": "baseline_external_prediction",
                        "reason": "no_observed_target_rows",
                    }
                )
                continue
            observed_target["target_observed"] = True
            observed_target["target_candidate_source"] = "observed_target_rows"
            target_specs.append((target_year, observed_target, True))

        candidates = build_forecast_candidates(
            full_df=full_df,
            training_df=training_df,
            country=country,
            input_year=args.forecast_input_year,
            forecast_year=args.forecast_year,
            universe_mode=args.forecast_target_universe,
        )
        if candidates.empty:
            exclusions.append(
                {
                    "fold": fold,
                    "Country": country,
                    "target_year": args.forecast_year,
                    "stage": "baseline_future_prediction",
                    "reason": "no_forecast_candidate_cells",
                }
            )
        else:
            target_specs.append((args.forecast_year, candidates, False))

        for method in BASELINE_METHODS:
            phi = fixed_phi[method]
            for target_year, target, is_observed in target_specs:
                training_history = training_df.copy()
                local_history = external_df.loc[
                    external_df["Year"] < target_year
                ].copy()
                try:
                    prediction = HELPER.external_baseline_prediction_for_year(
                        method_name=method,
                        training_history=training_history,
                        local_history=local_history,
                        target_df=target,
                        alpha=args.alpha,
                        beta=args.beta,
                        rolling_k=args.rolling_k,
                        ewma_halflife_years=args.ewma_halflife_years,
                    )
                except Exception as error:
                    exclusions.append(
                        {
                            "fold": fold,
                            "Country": country,
                            "target_year": target_year,
                            "model_name": method,
                            "stage": "baseline_prediction",
                            "reason": type(error).__name__,
                            "error_message": str(error),
                        }
                    )
                    if not args.continue_on_error:
                        raise
                    continue

                prediction["target_year"] = int(target_year)
                prediction["input_year"] = int(target_year - 1)
                prediction["forecast_origin_year"] = int(target_year - 1)
                prediction["prediction_horizon_years"] = 1
                prediction["fixed_validation_target_year"] = int(VALIDATION_TARGET_YEAR)
                prediction["forecast_protocol"] = (
                    "one_fixed_historical_model_two_external_evaluation_sets"
                )
                prediction["history_min_year"] = (
                    int(local_history["Year"].min()) if not local_history.empty else np.nan
                )
                prediction["history_max_year"] = (
                    int(local_history["Year"].max()) if not local_history.empty else np.nan
                )
                prediction["n_history_years"] = int(local_history["Year"].nunique())
                prediction["training_history_max_year"] = int(training_history["Year"].max())
                prediction["raw_phi"] = phi.raw_phi
                prediction["log_phi"] = phi.raw_phi
                prediction["phi"] = phi.phi
                prediction["rho"] = phi.rho
                prediction["bb_alpha"] = prediction["p_pred"] * phi.phi
                prediction["bb_beta"] = (1.0 - prediction["p_pred"]) * phi.phi
                prediction["phi_source"] = phi.source
                prediction["phi_calibration_max_target_year"] = phi.calibration_max_target_year
                prediction["phi_calibration_n_cells"] = phi.calibration_n_cells
                prediction["phi_calibration_n_tests"] = phi.calibration_n_tests
                prediction["prediction_history_rule"] = (
                    "external country observations before target year with transfer "
                    "components fitted on training countries through 2022"
                )
                local_history_ok = (
                    prediction["history_max_year"].isna()
                    | prediction["history_max_year"].lt(target_year)
                )
                prediction["temporal_leakage_check_passed"] = (
                    local_history_ok
                    & prediction["training_history_max_year"].le(TRAIN_MAX_YEAR)
                    & prediction["phi_calibration_max_target_year"].le(TRAIN_MAX_YEAR)
                )

                flags = HELPER.input_support_flags(
                    external_df=external_df,
                    target_df=target,
                    target_year=target_year,
                )
                prediction = prediction.merge(
                    flags,
                    on=["Country", "Year", "Species", "Family"],
                    how="left",
                    validate="one_to_one",
                )
                if (
                    "target_candidate_source" in target.columns
                    and "target_candidate_source" not in prediction.columns
                ):
                    candidate_meta = target[
                        ["Country", "Year", "Species", "Family", "target_candidate_source"]
                    ].drop_duplicates()
                    prediction = prediction.merge(
                        candidate_meta,
                        on=["Country", "Year", "Species", "Family"],
                        how="left",
                        validate="one_to_one",
                    )

                prediction = add_fold_metadata(
                    prediction,
                    fold=fold,
                    model_name=method,
                    n_training_countries=training_df["Country"].nunique(),
                    target_observed=is_observed,
                )
                outputs.append(add_beta_latent_intervals(prediction))

    if not outputs:
        raise RuntimeError("No baseline predictions were produced in the fold.")
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(exclusions)

def train_temporal_residual_model(
    cells: pd.DataFrame,
    full_df: pd.DataFrame,
    pair_table: pd.DataFrame,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> tuple[torch.nn.Module, pd.DataFrame, pd.DataFrame, int]:
    model_cells, model_pairs = coerce_temporal_model_inputs(
        cells=cells,
        pair_table=pair_table,
        fold=fold,
        target_year=VALIDATION_TARGET_YEAR,
    )
    loaders, _ = build_temporal_residual_encoder_dataloaders(
        cells=model_cells,
        pairs=model_pairs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed + 10_000 * fold,
    )

    for split_name in ["train", "val", "test"]:
        try:
            next(iter(loaders[split_name]))
        except StopIteration as error:
            raise ValueError(
                f"Empty {split_name} DataLoader for fold {fold}."
            ) from error
        except TypeError as error:
            raise TypeError(
                f"PyTorch tensor conversion failed in the {split_name} split "
                f"for fold {fold}. Cell dtypes: "
                f"{model_cells.dtypes.astype(str).to_dict()}. Pair dtypes: "
                f"{model_pairs.dtypes.astype(str).to_dict()}."
            ) from error

    model = SnapshotEncoderResidualModel(
        n_species=int(full_df["species_idx"].max()) + 1,
        n_families=int(full_df["family_idx"].max()) + 1,
        entity_emb_dim=args.entity_emb_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        latent_dim=args.latent_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_epoch = None
    best_val_loss = float("inf")
    stale_epochs = 0
    history_rows: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_temporal_residual(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            gradient_clip=args.gradient_clip,
        )
        val_metrics, val_predictions = evaluate_temporal_residual_loader(
            model=model,
            loader=loaders["val"],
            device=device,
            return_predictions=True,
        )
        val_error_metrics = HELPER.sqrt_n_error_metrics(val_predictions)
        val_loss = float(val_metrics["beta_binomial_nll_per_test"])
        history_rows.append(
            {
                "fold": fold,
                "model_name": "temporal_residual_encoder",
                "protocol": EVALUATION_PROTOCOL,
                "training_target_year_rule": "target_year < 2022",
                "validation_target_year": int(VALIDATION_TARGET_YEAR),
                "epoch": epoch,
                "train_loss_per_test": train_loss,
                "val_loss_per_test": val_loss,
                "val_weighted_mae": val_error_metrics["weighted_mae"],
                "val_weighted_rmse": val_error_metrics["weighted_rmse"],
                "val_error_weighting": "sqrt_n_total",
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    if best_state is None or best_epoch is None:
        raise RuntimeError("No temporal residual checkpoint was selected.")
    model.load_state_dict(best_state)
    _, raw_predictions = evaluate_temporal_residual_loader(
        model=model,
        loader=loaders["test"],
        device=device,
        return_predictions=True,
    )

    raw_years = set(
        pd.to_numeric(raw_predictions["target_year"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    declared_test_years = set(
        pd.to_numeric(
            model_pairs.loc[model_pairs["split"].eq("test"), "target_year"],
            errors="raise",
        ).astype(int)
    )
    if raw_years != declared_test_years:
        raise AssertionError(
            f"Neural test predictions cover {sorted(raw_years)}, expected "
            f"{sorted(declared_test_years)}."
        )

    if args.save_models:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "fold": fold,
                "training_data_max_year": int(TRAIN_MAX_YEAR),
                "gradient_training_max_target_year": int(
                    model_pairs.loc[
                        model_pairs["split"].eq("train"), "target_year"
                    ].max()
                ),
                "checkpoint_selection_target_year": int(
                    VALIDATION_TARGET_YEAR
                ),
                "external_test_target_years": sorted(declared_test_years),
                "historical_country_test_target_years": sorted(
                    year for year in declared_test_years if year <= TRAIN_MAX_YEAR
                ),
                "country_year_test_target_years": list(TEST_YEARS),
                "best_epoch": best_epoch,
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "args": vars(args),
            },
            output_dir / "temporal_residual_encoder_model.pt",
        )
    return model, raw_predictions, pd.DataFrame(history_rows), best_epoch


def temporal_residual_observed_predictions(
    raw_predictions: pd.DataFrame,
    model: SnapshotEncoderResidualModel,
    cells: pd.DataFrame,
    full_df: pd.DataFrame,
    training_df: pd.DataFrame,
    fold: int,
    best_epoch: int,
    train_max_target_year: int,
) -> pd.DataFrame:
    target_meta = cells[
        [
            "cell_row_id",
            "Species",
            "Family",
            "species_idx",
            "family_idx",
            "p_baseline",
            "baseline_source",
            "baseline_history_min_year",
            "baseline_history_max_year",
            "baseline_n_history_years",
            "baseline_n_history_cells",
            "baseline_n_history_tests",
            "baseline_training_cutoff_exclusive",
        ]
    ]
    out = raw_predictions.merge(
        target_meta,
        on="cell_row_id",
        how="left",
        validate="one_to_one",
    )
    phi = float(
        (torch.nn.functional.softplus(model.log_phi.detach()) + EPS).cpu()
    )
    out["raw_phi"] = float(model.log_phi.detach().cpu())
    out["log_phi"] = out["raw_phi"]
    out["phi"] = phi
    out["rho"] = 1.0 / (phi + 1.0)
    out["bb_alpha"] = out["p_pred"] * phi
    out["bb_beta"] = (1.0 - out["p_pred"]) * phi
    out["phi_source"] = (
        "jointly_learned_on_historical_training_country_transitions"
    )
    out["phi_fit_max_target_year"] = int(train_max_target_year)
    out["neural_training_max_target_year"] = int(train_max_target_year)
    out["checkpoint_selection_target_year"] = int(
        VALIDATION_TARGET_YEAR
    )
    out["fixed_validation_target_year"] = int(VALIDATION_TARGET_YEAR)
    out["forecast_protocol"] = (
        "one_fixed_historical_model_per_external_country_fold"
    )
    out["best_epoch"] = int(best_epoch)
    out["prediction_history_rule"] = (
        "external country snapshot at target year minus one with transfer "
        "components fitted on training countries through 2022"
    )
    valid_observed_year = (
        out["target_year"].le(TRAIN_MAX_YEAR)
        | out["target_year"].isin(TEST_YEARS)
    )
    out["temporal_leakage_check_passed"] = (
        valid_observed_year
        & out["input_year"].eq(out["target_year"] - 1)
        & (
            out["baseline_history_max_year"].isna()
            | out["baseline_history_max_year"].le(TRAIN_MAX_YEAR)
        )
        & out["neural_training_max_target_year"].lt(VALIDATION_TARGET_YEAR)
        & out["checkpoint_selection_target_year"].eq(VALIDATION_TARGET_YEAR)
    )

    flag_parts: list[pd.DataFrame] = []
    for country, current_target_year in out[
        ["Country", "target_year"]
    ].drop_duplicates().itertuples(index=False):
        external_df = full_df.loc[full_df["Country"].eq(country)]
        target = external_df.loc[
            external_df["Year"].eq(current_target_year)
        ]
        flag_parts.append(
            HELPER.input_support_flags(
                external_df,
                target,
                int(current_target_year),
            )
        )
    out = out.merge(
        pd.concat(flag_parts, ignore_index=True),
        left_on=["Country", "target_year", "Species", "Family"],
        right_on=["Country", "Year", "Species", "Family"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["Year"])
    out = HELPER.training_support_flags(
        training_country_df=training_df,
        predictions=out,
        train_max_target_year=train_max_target_year,
    )
    out = add_fold_metadata(
        out,
        fold=fold,
        model_name="temporal_residual_encoder",
        n_training_countries=training_df["Country"].nunique(),
        target_observed=True,
    )
    out["target_candidate_source"] = "observed_target_rows"
    return add_beta_latent_intervals(out)


def tensor(
    values: Iterable,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    series = pd.Series(list(values))
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        bad = series.loc[numeric.isna()].head(10).tolist()
        raise ValueError(
            f"Cannot convert values to a PyTorch tensor. Invalid examples: {bad}"
        )
    if dtype == torch.long:
        array = numeric.to_numpy(dtype=np.int64)
    else:
        array = numeric.to_numpy(dtype=np.float32)
    return torch.as_tensor(array, dtype=dtype, device=device)


@torch.no_grad()
def predict_temporal_residual_unobserved(
    model: SnapshotEncoderResidualModel,
    input_cells: pd.DataFrame,
    candidates: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    if input_cells.empty or candidates.empty:
        return pd.DataFrame()
    model.eval()
    n_input = len(input_cells)
    n_target = len(candidates)
    final_logits, delta_logits, _ = model(
        input_species_idx=tensor(
            input_cells["species_idx"], torch.long, device
        ),
        input_family_idx=tensor(
            input_cells["family_idx"], torch.long, device
        ),
        input_p_baseline=tensor(
            input_cells["p_baseline"], torch.float32, device
        ),
        input_residual_prop_S=tensor(
            input_cells["residual_prop_S"], torch.float32, device
        ),
        input_n_total=tensor(
            input_cells["n_total"], torch.float32, device
        ),
        input_snapshot_batch_idx=torch.zeros(
            n_input, dtype=torch.long, device=device
        ),
        n_snapshots_in_batch=1,
        target_species_idx=tensor(
            candidates["species_idx"], torch.long, device
        ),
        target_family_idx=tensor(
            candidates["family_idx"], torch.long, device
        ),
        target_snapshot_batch_idx=torch.zeros(
            n_target, dtype=torch.long, device=device
        ),
        target_baseline_logit=tensor(
            candidates["baseline_logit"], torch.float32, device
        ),
    )
    out = candidates.copy()
    out["p_pred"] = torch.sigmoid(final_logits).detach().cpu().numpy()
    out["delta_logit"] = delta_logits.detach().cpu().numpy()
    out["input_year"] = int(input_cells["Year"].iloc[0])
    out["target_year"] = out["Year"].astype(int)
    return out.drop(columns=["Year"])


def add_unobserved_neural_metadata(
    predictions: pd.DataFrame,
    model: SnapshotEncoderResidualModel,
    full_df: pd.DataFrame,
    training_df: pd.DataFrame,
    fold: int,
    best_epoch: int,
    train_max_target_year: int,
    target_year: int,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    out = predictions.copy()
    phi = float(
        (torch.nn.functional.softplus(model.log_phi.detach()) + EPS).cpu()
    )
    out["raw_phi"] = float(model.log_phi.detach().cpu())
    out["log_phi"] = out["raw_phi"]
    out["phi"] = phi
    out["rho"] = 1.0 / (phi + 1.0)
    out["bb_alpha"] = out["p_pred"] * phi
    out["bb_beta"] = (1.0 - out["p_pred"]) * phi
    out["phi_source"] = (
        "jointly_learned_on_historical_training_country_transitions"
    )
    out["phi_fit_max_target_year"] = int(train_max_target_year)
    out["neural_training_max_target_year"] = int(train_max_target_year)
    out["checkpoint_selection_target_year"] = int(
        VALIDATION_TARGET_YEAR
    )
    out["fixed_validation_target_year"] = int(VALIDATION_TARGET_YEAR)
    out["prospective_target_year"] = int(target_year)
    out["forecast_protocol"] = (
        "fixed_historical_model_prospective_forecast"
    )
    out["best_epoch"] = int(best_epoch)
    out["prediction_history_rule"] = (
        "external country forecast input snapshot with transfer components "
        "fitted on training countries through 2022 and fixed neural parameters"
    )
    out["temporal_leakage_check_passed"] = (
        out["input_year"].lt(out["target_year"])
        & (
            out["baseline_history_max_year"].isna()
            | out["baseline_history_max_year"].le(TRAIN_MAX_YEAR)
        )
        & out["neural_training_max_target_year"].lt(
            VALIDATION_TARGET_YEAR
        )
        & out["checkpoint_selection_target_year"].eq(
            VALIDATION_TARGET_YEAR
        )
    )

    flag_parts: list[pd.DataFrame] = []
    for country, current_target_year in out[
        ["Country", "target_year"]
    ].drop_duplicates().itertuples(index=False):
        external_df = full_df.loc[full_df["Country"].eq(country)]
        target = out.loc[
            out["Country"].eq(country)
            & out["target_year"].eq(current_target_year),
            ["Country", "Species", "Family", "target_year"],
        ].rename(columns={"target_year": "Year"})
        flag_parts.append(
            HELPER.input_support_flags(
                external_df,
                target,
                int(current_target_year),
            )
        )
    out = out.merge(
        pd.concat(flag_parts, ignore_index=True),
        left_on=["Country", "target_year", "Species", "Family"],
        right_on=["Country", "Year", "Species", "Family"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["Year"])
    out = HELPER.training_support_flags(
        training_country_df=training_df,
        predictions=out,
        train_max_target_year=train_max_target_year,
    )
    out = add_fold_metadata(
        out,
        fold=fold,
        model_name="temporal_residual_encoder",
        n_training_countries=training_df["Country"].nunique(),
        target_observed=False,
    )
    return add_beta_latent_intervals(out)


def forecast_neural_for_fold(
    model: SnapshotEncoderResidualModel,
    cells: pd.DataFrame,
    full_df: pd.DataFrame,
    training_df: pd.DataFrame,
    test_countries: set[str],
    fold: int,
    best_epoch: int,
    train_max_target_year: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    exclusions: list[dict[str, object]] = []

    for country in sorted(test_countries):
        input_cells = cells.loc[
            cells["Country"].eq(country)
            & cells["Year"].eq(args.forecast_input_year)
        ].copy()
        if input_cells.empty:
            exclusions.append(
                {
                    "fold": fold,
                    "Country": country,
                    "target_year": args.forecast_year,
                    "stage": "temporal_residual_encoder_future",
                    "reason": "no_forecast_input_snapshot",
                }
            )
            continue

        candidates = build_forecast_candidates(
            full_df=full_df,
            training_df=training_df,
            country=country,
            input_year=args.forecast_input_year,
            forecast_year=args.forecast_year,
            universe_mode=args.forecast_target_universe,
        )
        if candidates.empty:
            exclusions.append(
                {
                    "fold": fold,
                    "Country": country,
                    "target_year": args.forecast_year,
                    "stage": "temporal_residual_encoder_future",
                    "reason": "no_forecast_candidate_cells",
                }
            )
            continue

        candidates = add_fold_baseline_to_candidates(
            candidates=candidates,
            training_df=training_df,
            alpha=args.alpha,
            beta=args.beta,
        )
        prediction = predict_temporal_residual_unobserved(
            model=model,
            input_cells=input_cells,
            candidates=candidates,
            device=device,
        )
        prediction = add_unobserved_neural_metadata(
            predictions=prediction,
            model=model,
            full_df=full_df,
            training_df=training_df,
            fold=fold,
            best_epoch=best_epoch,
            train_max_target_year=train_max_target_year,
            target_year=args.forecast_year,
        )
        outputs.append(prediction)

    output_df = (
        pd.concat(outputs, ignore_index=True)
        if outputs
        else pd.DataFrame()
    )
    return output_df, pd.DataFrame(exclusions)


def build_fold_role_table(
    fold: int,
    training_countries: set[str],
    test_countries: set[str],
) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "fold": fold,
                    "colleague_fold": int(fold - 1),
                    "Country": sorted(training_countries),
                    "role": "training",
                }
            ),
            pd.DataFrame(
                {
                    "fold": fold,
                    "colleague_fold": int(fold - 1),
                    "Country": sorted(test_countries),
                    "role": "external_test",
                }
            ),
        ],
        ignore_index=True,
    )


def run_fold(
    full_df: pd.DataFrame,
    fold: int,
    training_countries: set[str],
    test_countries: set[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_dir = args.output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed + fold)

    all_pairs = HELPER.consecutive_pairs(full_df)
    cells, training_df = build_fold_cells(
        full_df=full_df,
        training_countries=training_countries,
        alpha=args.alpha,
        beta=args.beta,
    )

    predictions: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    pair_tables: list[pd.DataFrame] = []
    exclusions: list[pd.DataFrame] = []

    if "baselines" in args.models:
        baseline_predictions, baseline_exclusions = run_baselines_for_fold(
            full_df=full_df,
            training_df=training_df,
            test_countries=test_countries,
            fold=fold,
            args=args,
        )
        predictions.append(baseline_predictions)
        if not baseline_exclusions.empty:
            exclusions.append(baseline_exclusions)

    if "temporal_residual" in args.models:
        model_dir = fold_dir / "fixed_historical_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            (
                train_pairs,
                val_pairs,
                historical_test_pairs,
                country_year_test_pairs,
            ) = split_pairs_for_dual_evaluation(
                all_pairs=all_pairs,
                training_countries=training_countries,
                test_countries=test_countries,
            )
            all_test_pairs = historical_test_pairs + country_year_test_pairs
            pair_table = HELPER.build_temporal_residual_pair_table(
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                test_pairs=all_test_pairs,
            )
            pair_table["fold"] = int(fold)
            pair_table["training_data_max_year"] = int(TRAIN_MAX_YEAR)
            pair_table["checkpoint_selection_target_year"] = int(VALIDATION_TARGET_YEAR)
            pair_table["evaluation_set"] = np.select(
                [
                    pair_table["split"].eq("test")
                    & pair_table["target_year"].le(TRAIN_MAX_YEAR),
                    pair_table["split"].eq("test")
                    & pair_table["target_year"].isin(TEST_YEARS),
                    pair_table["split"].eq("train"),
                    pair_table["split"].eq("val"),
                ],
                [
                    EVALUATION_SET_COUNTRY,
                    EVALUATION_SET_COUNTRY_YEAR,
                    "gradient_training",
                    "checkpoint_selection",
                ],
                default="unknown",
            )
            pair_table.to_csv(
                model_dir / "temporal_pairs_dual_evaluation.csv",
                index=False,
            )
            pair_tables.append(pair_table)

            model, raw, history, best_epoch = train_temporal_residual_model(
                cells=cells,
                full_df=full_df,
                pair_table=pair_table,
                fold=fold,
                args=args,
                device=device,
                output_dir=model_dir,
            )
            train_max_target_year = max(pair[2] for pair in train_pairs)
            observed = temporal_residual_observed_predictions(
                raw_predictions=raw,
                model=model,
                cells=cells,
                full_df=full_df,
                training_df=training_df,
                fold=fold,
                best_epoch=best_epoch,
                train_max_target_year=train_max_target_year,
            )
            predictions.append(observed)
            histories.append(history)

            if args.run_future_neural:
                future, future_exclusions = forecast_neural_for_fold(
                    model=model,
                    cells=cells,
                    full_df=full_df,
                    training_df=training_df,
                    test_countries=test_countries,
                    fold=fold,
                    best_epoch=best_epoch,
                    train_max_target_year=train_max_target_year,
                    args=args,
                    device=device,
                )
                if not future.empty:
                    predictions.append(future)
                if not future_exclusions.empty:
                    exclusions.append(future_exclusions)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as error:
            exclusions.append(
                pd.DataFrame(
                    [
                        {
                            "fold": fold,
                            "stage": "dual_external_neural_evaluation",
                            "reason": type(error).__name__,
                            "error_message": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    ]
                )
            )
            print(f"Fold {fold}, temporal residual model failed: {error}")
            if not args.continue_on_error:
                raise

    if not predictions:
        raise RuntimeError("No predictions were generated for the fold.")

    fold_predictions = pd.concat(predictions, ignore_index=True, sort=False)
    if not fold_predictions["temporal_leakage_check_passed"].fillna(False).all():
        raise AssertionError("Temporal provenance failed in the fold.")

    observed = observed_only(fold_predictions)
    observed_sets = set(observed["evaluation_set"].astype(str).unique())
    expected_sets = {EVALUATION_SET_COUNTRY, EVALUATION_SET_COUNTRY_YEAR}
    if observed_sets != expected_sets:
        raise AssertionError(
            f"Fold {fold} produced evaluation sets {observed_sets}, expected {expected_sets}."
        )

    fold_predictions.to_csv(
        fold_dir / "temporal_external_predictions_dual_evaluation.csv",
        index=False,
    )
    history_df = (
        pd.concat(histories, ignore_index=True, sort=False)
        if histories
        else pd.DataFrame()
    )
    if not history_df.empty:
        history_df.to_csv(fold_dir / "training_history.csv", index=False)

    if pair_tables:
        pd.concat(pair_tables, ignore_index=True, sort=False).to_csv(
            fold_dir / "temporal_pairs_dual_evaluation.csv",
            index=False,
        )

    exclusion_df = (
        pd.concat(exclusions, ignore_index=True, sort=False)
        if exclusions
        else pd.DataFrame()
    )
    if not exclusion_df.empty:
        exclusion_df.to_csv(fold_dir / "prediction_exclusions.csv", index=False)

    fold_assignment = build_fold_role_table(
        fold=fold,
        training_countries=training_countries,
        test_countries=test_countries,
    )
    fold_assignment["training_data_max_year"] = int(TRAIN_MAX_YEAR)
    fold_assignment["country_year_test_years"] = ",".join(map(str, TEST_YEARS))
    fold_assignment.to_csv(fold_dir / "country_assignment.csv", index=False)
    return fold_predictions, history_df, exclusion_df, fold_assignment

def observed_only(predictions: pd.DataFrame) -> pd.DataFrame:
    target_observed = predictions["target_observed"]
    if target_observed.dtype != bool:
        target_observed = target_observed.astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    return predictions.loc[
        target_observed.fillna(False)
        & predictions["prop_S"].notna()
        & predictions["n_total"].notna()
    ].copy()


def matched_observed_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    key = ["fold", "Country", "Species", "Family", "target_year"]
    models = sorted(predictions["model_name"].unique().tolist())
    key_sets: list[set[tuple[object, ...]]] = []
    for model in models:
        model_keys = predictions.loc[
            predictions["model_name"].eq(model), key
        ].drop_duplicates()
        key_sets.append(set(map(tuple, model_keys.to_numpy())))
    common = set.intersection(*key_sets) if key_sets else set()
    if not common:
        raise ValueError("No common observed prediction rows across models.")
    common_df = pd.DataFrame(list(common), columns=key)
    return predictions.merge(
        common_df,
        on=key,
        how="inner",
        validate="many_to_one",
    )


def summarize(
    predictions: pd.DataFrame,
    group_columns: list[str],
    evaluation_set: str,
) -> pd.DataFrame:
    return HELPER.summarize_predictions(
        predictions=predictions,
        group_columns=group_columns,
        evaluation_set=evaluation_set,
    )


def verify_external_fold_membership(
    observed_predictions: pd.DataFrame,
    fold_assignments: pd.DataFrame,
) -> None:
    predicted_pairs = observed_predictions[
        ["Country", "fold"]
    ].drop_duplicates()
    counts = predicted_pairs.groupby("Country")["fold"].nunique()
    if not counts.le(1).all():
        raise AssertionError(
            "At least one country occurs in more than one external fold."
        )

    expected = fold_assignments[["Country", "fold"]].drop_duplicates()
    checked = predicted_pairs.merge(
        expected,
        on=["Country", "fold"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if not checked["_merge"].eq("both").all():
        raise AssertionError(
            "Some predictions do not match the saved country fold assignment."
        )

    if observed_predictions[
        "country_seen_in_parameter_fitting"
    ].fillna(True).any():
        raise AssertionError(
            "An external country was marked as seen during parameter fitting."
        )



def summarize_mean_std_across_folds(
    fold_metrics: pd.DataFrame,
    evaluation_set: str,
) -> pd.DataFrame:
    metric_columns = [
        "weighted_mae",
        "weighted_rmse",
        "sqrt_n_weighted_mae",
        "sqrt_n_weighted_rmse",
        "n_weighted_mae",
        "n_weighted_rmse",
        "unweighted_mae",
        "unweighted_rmse",
        "beta_binomial_nll_per_test",
        "binomial_ce_per_test",
    ]
    rows: list[dict[str, object]] = []
    for model_name, group in fold_metrics.groupby("model_name", sort=True):
        folds = sorted(group["fold"].astype(int).unique().tolist())
        if len(folds) != EXPECTED_N_FOLDS:
            raise RuntimeError(
                f"{evaluation_set}, {model_name} has folds {folds}, expected five folds."
            )
        row: dict[str, object] = {
            "evaluation_set": evaluation_set,
            "model_name": model_name,
            "n_folds": EXPECTED_N_FOLDS,
            "folds": json.dumps(folds),
            "uncertainty_definition": "sample_standard_deviation_across_five_folds",
        }
        for column in metric_columns:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if len(values) != EXPECTED_N_FOLDS:
                row[f"{column}_mean"] = np.nan
                row[f"{column}_std"] = np.nan
                row[f"{column}_mean_plus_minus_std"] = ""
                continue
            mean_value = float(values.mean())
            std_value = float(values.std(ddof=1))
            row[f"{column}_mean"] = mean_value
            row[f"{column}_std"] = std_value
            row[f"{column}_mean_plus_minus_std"] = (
                f"{mean_value:.6f} ± {std_value:.6f}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_evaluation_outputs(
    predictions: pd.DataFrame,
    evaluation_set: str,
    prefix: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = predictions.loc[
        predictions["evaluation_set"].eq(evaluation_set)
    ].copy()
    if subset.empty:
        raise RuntimeError(f"No predictions for {evaluation_set}.")
    matched = matched_observed_predictions(subset)
    matched.to_csv(
        output_dir / f"{prefix}_matched_predictions.csv",
        index=False,
    )
    fold_year = summarize(
        matched,
        ["fold", "target_year", "model_name"],
        evaluation_set,
    )
    fold_metrics = summarize(
        matched,
        ["fold", "model_name"],
        evaluation_set,
    )
    mean_std = summarize_mean_std_across_folds(fold_metrics, evaluation_set)
    fold_year.to_csv(
        output_dir / f"{prefix}_metrics_by_fold_and_year.csv",
        index=False,
    )
    fold_metrics.to_csv(
        output_dir / f"{prefix}_metrics_by_fold.csv",
        index=False,
    )
    mean_std.to_csv(
        output_dir / f"{prefix}_metrics_mean_std_across_folds.csv",
        index=False,
    )
    return fold_metrics, mean_std

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)

    full_df = HELPER.prepare_observed_data(args.input_path)
    available_years = set(full_df["Year"].astype(int).unique().tolist())
    missing_vault_years = sorted(set(TEST_YEARS) - available_years)
    if missing_vault_years:
        raise ValueError(f"The input dataset is missing {missing_vault_years}.")
    if VALIDATION_TARGET_YEAR not in available_years:
        raise ValueError("The input dataset has no observations in 2022.")

    fold_assignments = build_grouped_country_folds(
        full_df,
        args.country_folds_path,
    )
    fold_assignments.to_csv(
        args.output_dir / "country_fold_assignments.csv",
        index=False,
    )

    all_countries = set(full_df["Country"].astype(str).unique().tolist())
    prediction_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []
    exclusion_parts: list[pd.DataFrame] = []
    fold_role_parts: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []

    fold_ids = sorted(fold_assignments["fold"].astype(int).unique().tolist())
    for fold in fold_ids:
        test_countries = set(
            fold_assignments.loc[
                fold_assignments["fold"].eq(fold), "Country"
            ].astype(str)
        )
        training_countries = all_countries - test_countries
        print(
            f"\nFold {fold}/{EXPECTED_N_FOLDS}: "
            f"{len(training_countries)} training countries, "
            f"{len(test_countries)} external countries"
        )
        try:
            fold_predictions, fold_history, fold_exclusions, fold_roles = run_fold(
                full_df=full_df,
                fold=int(fold),
                training_countries=training_countries,
                test_countries=test_countries,
                args=args,
                device=device,
            )
            prediction_parts.append(fold_predictions)
            if not fold_history.empty:
                history_parts.append(fold_history)
            if not fold_exclusions.empty:
                exclusion_parts.append(fold_exclusions)
            fold_role_parts.append(fold_roles)
        except Exception as error:
            failures.append(
                {
                    "fold": int(fold),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"Fold {fold} failed: {error}")
            if not args.continue_on_error:
                raise

    if failures:
        pd.DataFrame(failures).to_csv(
            args.output_dir / "fold_failures.csv",
            index=False,
        )
    if len(prediction_parts) != EXPECTED_N_FOLDS:
        raise RuntimeError(
            f"Only {len(prediction_parts)} of five folds completed. "
            "Mean and standard deviation across five folds cannot be reported."
        )

    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False)
    observed_predictions = observed_only(predictions)
    verify_external_fold_membership(observed_predictions, fold_assignments)

    historical = observed_predictions.loc[
        observed_predictions["evaluation_set"].eq(EVALUATION_SET_COUNTRY)
    ].copy()
    country_year = observed_predictions.loc[
        observed_predictions["evaluation_set"].eq(EVALUATION_SET_COUNTRY_YEAR)
    ].copy()
    if historical.empty or country_year.empty:
        raise RuntimeError("One of the two required evaluation tasks is empty.")

    expected_models: set[str] = set()
    if "baselines" in args.models:
        expected_models.update(BASELINE_METHODS)
    if "temporal_residual" in args.models:
        expected_models.add("temporal_residual_encoder")
    for evaluation_set, frame in [
        (EVALUATION_SET_COUNTRY, historical),
        (EVALUATION_SET_COUNTRY_YEAR, country_year),
    ]:
        missing = sorted(expected_models - set(frame["model_name"].astype(str)))
        if missing:
            raise RuntimeError(f"{evaluation_set} is missing models {missing}.")

    predictions.to_csv(
        args.output_dir / "temporal_predictions_all_outputs.csv",
        index=False,
    )
    observed_predictions.to_csv(
        args.output_dir / "temporal_observed_predictions_all_external_tests.csv",
        index=False,
    )
    historical.to_csv(
        args.output_dir / "temporal_observed_predictions_country_generalization_through_2022.csv",
        index=False,
    )
    country_year.to_csv(
        args.output_dir / "temporal_observed_predictions_country_year_generalization_2023_2024.csv",
        index=False,
    )
    country_year.to_csv(
        args.output_dir / "temporal_observed_predictions_2023_2024.csv",
        index=False,
    )

    write_evaluation_outputs(
        observed_predictions,
        EVALUATION_SET_COUNTRY,
        "temporal_country_generalization",
        args.output_dir,
    )
    write_evaluation_outputs(
        observed_predictions,
        EVALUATION_SET_COUNTRY_YEAR,
        "temporal_country_year_generalization",
        args.output_dir,
    )

    target_observed = predictions["target_observed"]
    if target_observed.dtype != bool:
        target_observed = target_observed.astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    future_predictions = predictions.loc[
        ~target_observed.fillna(False)
    ].copy()
    future_predictions.to_csv(
        args.output_dir / f"temporal_{args.forecast_year}_prospective_predictions.csv",
        index=False,
    )

    if history_parts:
        pd.concat(history_parts, ignore_index=True, sort=False).to_csv(
            args.output_dir / "temporal_training_history.csv",
            index=False,
        )
    if exclusion_parts:
        pd.concat(exclusion_parts, ignore_index=True, sort=False).to_csv(
            args.output_dir / "temporal_exclusions.csv",
            index=False,
        )
    if fold_role_parts:
        pd.concat(fold_role_parts, ignore_index=True, sort=False).to_csv(
            args.output_dir / "fold_country_roles.csv",
            index=False,
        )

    metadata = {
        "script": "04_run_grouped_country_temporal_generalization.py",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "country_folds_path": str(args.country_folds_path),
        "number_of_folds": EXPECTED_N_FOLDS,
        "training_data_max_year": TRAIN_MAX_YEAR,
        "gradient_training_target_rule": "target_year < 2022",
        "checkpoint_selection_target_year": VALIDATION_TARGET_YEAR,
        "historical_country_test_rule": "external countries and target year <= 2022",
        "country_year_test_years": list(TEST_YEARS),
        "uncertainty_definition": "mean and sample standard deviation across five fold metrics",
        "one_training_run_per_fold": True,
        "models": args.models,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf8",
    )

    print("\nCompleted both external evaluation tasks")
    print("Historical country generalization rows:", len(historical))
    print("Country and year generalization rows:", len(country_year))
    print("Completed folds:", fold_ids)

if __name__ == "__main__":
    main()
