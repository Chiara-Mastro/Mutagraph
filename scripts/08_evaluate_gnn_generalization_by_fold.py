#!/usr/bin/env python3
"""
Evaluate saved GNN reconstruction and next year projection predictions by fold.

The script performs four evaluations:

1. Reconstruction country generalization through 2022
2. Reconstruction country and year generalization in 2023 and 2024
3. Temporal country generalization with target years through 2022
4. Temporal country and year generalization with target years 2023 and 2024

For every evaluation, metrics are first calculated independently within each
external country fold. The final uncertainty is the sample standard deviation
across the five fold metrics, using ddof equal to one.

No model fitting is performed by this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln


RECONSTRUCTION_KEYS = ["Country", "Year", "Species", "Family"]
TEMPORAL_KEYS = ["Country", "year_from", "year_to", "Species", "Family"]

HISTORICAL_TASK = "country_generalization_through_2022"
VAULT_TASK = "country_and_year_generalization_2023_2024"

PRIMARY_METRICS = [
    "weighted_mae",
    "weighted_rmse",
    "beta_binomial_nll_per_test",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate GNN reconstruction and temporal metrics independently "
            "inside each external country fold, then report mean and sample "
            "standard deviation across the five folds."
        )
    )
    parser.add_argument(
        "--reconstruction_path",
        type=Path,
        required=True,
        help="GNN leave one out reconstruction predictions.",
    )
    parser.add_argument(
        "--projection_path",
        type=Path,
        required=True,
        help="GNN next year projection predictions.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--country_folds_path",
        type=Path,
        default=None,
        help=(
            "Optional folds JSON or CSV. When supplied, the script verifies "
            "that every saved prediction has the expected country fold."
        ),
    )
    parser.add_argument(
        "--historical_max_year",
        type=int,
        default=2022,
    )
    parser.add_argument(
        "--vault_years",
        nargs="+",
        type=int,
        default=[2023, 2024],
    )
    parser.add_argument(
        "--expected_folds",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    args.vault_years = sorted(set(args.vault_years))
    if args.expected_folds < 2:
        raise ValueError("expected_folds must be at least two.")
    if not args.vault_years:
        raise ValueError("At least one vault year is required.")
    if min(args.vault_years) <= args.historical_max_year:
        raise ValueError(
            "Every vault year must be after historical_max_year."
        )
    return args


def require_columns(
    frame: pd.DataFrame,
    required: list[str],
    table_name: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )


def clean_text_columns(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = out[column].astype(str).str.strip()
    return out


def numeric_column(
    frame: pd.DataFrame,
    column: str,
    table_name: str,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    bad = values.isna() | ~np.isfinite(values.to_numpy(dtype=float))
    if bad.any():
        examples = frame.loc[bad, column].head(10).tolist()
        raise ValueError(
            f"{table_name}.{column} contains missing or nonnumeric values: "
            f"{examples}"
        )
    return values


def boolean_column(
    values: pd.Series,
    column_name: str,
) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    mapped = values.astype(str).str.strip().str.lower().map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )
    if mapped.isna().any():
        examples = values.loc[mapped.isna()].head(10).tolist()
        raise ValueError(
            f"{column_name} contains invalid boolean values: {examples}"
        )
    return mapped.astype(bool)


def load_reconstruction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    required = RECONSTRUCTION_KEYS + [
        "n_S",
        "n_total",
        "prop_S_observed",
        "prop_S_pred",
        "alpha_cal",
        "beta_cal",
        "fold_model",
    ]
    require_columns(frame, required, "reconstruction")

    frame = clean_text_columns(
        frame,
        ["Country", "Species", "Family"],
    )
    for column in [
        "Year",
        "n_S",
        "n_total",
        "prop_S_observed",
        "prop_S_pred",
        "alpha_cal",
        "beta_cal",
        "fold_model",
    ]:
        frame[column] = numeric_column(frame, column, "reconstruction")

    frame["Year"] = frame["Year"].astype(int)
    frame["fold_model"] = frame["fold_model"].astype(int)

    duplicated = frame.duplicated(RECONSTRUCTION_KEYS, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, RECONSTRUCTION_KEYS].head(20)
        raise ValueError(
            "Reconstruction predictions contain duplicate keys:\n"
            + examples.to_string(index=False)
        )

    if (frame["n_total"] <= 0).any():
        raise ValueError("Reconstruction n_total must be positive.")
    if (frame["n_S"] < 0).any() or (
        frame["n_S"] > frame["n_total"]
    ).any():
        raise ValueError("Reconstruction n_S must lie between zero and n_total.")
    if not frame["prop_S_observed"].between(0, 1).all():
        raise ValueError("Reconstruction observed proportions must lie in zero to one.")
    if not frame["prop_S_pred"].between(0, 1).all():
        raise ValueError("Reconstruction predictions must lie in zero to one.")
    if (frame["alpha_cal"] <= 0).any() or (
        frame["beta_cal"] <= 0
    ).any():
        raise ValueError(
            "Reconstruction calibrated alpha and beta must be positive."
        )

    return frame


def load_projection(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    required = TEMPORAL_KEYS + [
        "prop_S_current",
        "prop_S_pred_next",
        "alpha_cal",
        "beta_cal",
        "fold_model",
        "next_year_in_data",
    ]
    require_columns(frame, required, "projection")

    frame = clean_text_columns(
        frame,
        ["Country", "Species", "Family"],
    )
    for column in [
        "year_from",
        "year_to",
        "prop_S_current",
        "prop_S_pred_next",
        "alpha_cal",
        "beta_cal",
        "fold_model",
    ]:
        frame[column] = numeric_column(frame, column, "projection")

    frame["year_from"] = frame["year_from"].astype(int)
    frame["year_to"] = frame["year_to"].astype(int)
    frame["fold_model"] = frame["fold_model"].astype(int)
    frame["next_year_in_data"] = boolean_column(
        frame["next_year_in_data"],
        "next_year_in_data",
    )

    nonconsecutive = frame["year_to"].ne(frame["year_from"] + 1)
    if nonconsecutive.any():
        examples = frame.loc[nonconsecutive, TEMPORAL_KEYS].head(20)
        raise ValueError(
            "Projection rows must describe consecutive years:\n"
            + examples.to_string(index=False)
        )

    duplicated = frame.duplicated(TEMPORAL_KEYS, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, TEMPORAL_KEYS].head(20)
        raise ValueError(
            "Projection predictions contain duplicate keys:\n"
            + examples.to_string(index=False)
        )

    if not frame["prop_S_current"].between(0, 1).all():
        raise ValueError("Projection current proportions must lie in zero to one.")
    if not frame["prop_S_pred_next"].between(0, 1).all():
        raise ValueError("Projection predictions must lie in zero to one.")
    if (frame["alpha_cal"] <= 0).any() or (
        frame["beta_cal"] <= 0
    ).any():
        raise ValueError(
            "Projection calibrated alpha and beta must be positive."
        )

    return frame


def validate_country_fold_consistency(
    frame: pd.DataFrame,
    table_name: str,
) -> None:
    counts = frame.groupby("Country")["fold_model"].nunique()
    bad = counts.loc[counts.ne(1)]
    if not bad.empty:
        raise ValueError(
            f"{table_name} assigns some countries to multiple folds: "
            + bad.head(20).to_dict().__repr__()
        )


def make_fold_index(
    fold_values: list[int],
    expected_folds: int,
) -> dict[int, int]:
    unique = sorted(set(int(value) for value in fold_values))
    if len(unique) != expected_folds:
        raise ValueError(
            f"Expected {expected_folds} folds, found {len(unique)}: {unique}"
        )
    return {
        original_fold: sequential_fold
        for sequential_fold, original_fold in enumerate(unique, start=1)
    }


def load_external_fold_assignments(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        payload: Any = json.loads(path.read_text(encoding="utf8"))
        rows: list[dict[str, object]] = []

        if not isinstance(payload, dict):
            raise ValueError("Fold JSON must contain a dictionary.")

        values_are_lists = all(
            isinstance(value, list) for value in payload.values()
        )
        values_are_scalars = all(
            not isinstance(value, (dict, list)) for value in payload.values()
        )

        if values_are_lists:
            for fold, countries in payload.items():
                for country in countries:
                    rows.append(
                        {
                            "Country": str(country).strip(),
                            "expected_fold_model": int(fold),
                        }
                    )
        elif values_are_scalars:
            for country, fold in payload.items():
                rows.append(
                    {
                        "Country": str(country).strip(),
                        "expected_fold_model": int(fold),
                    }
                )
        else:
            raise ValueError(
                "Fold JSON must map fold ids to country lists or countries to fold ids."
            )
        assignments = pd.DataFrame(rows)
    else:
        source = pd.read_csv(path)
        require_columns(source, ["Country"], "country_folds")
        fold_column = next(
            (
                candidate
                for candidate in [
                    "fold_model",
                    "colleague_fold",
                    "external_fold",
                    "fold",
                ]
                if candidate in source.columns
            ),
            None,
        )
        if fold_column is None:
            raise ValueError(
                "Fold CSV needs one of fold_model, colleague_fold, "
                "external_fold, or fold."
            )
        assignments = source[["Country", fold_column]].copy()
        assignments = assignments.rename(
            columns={fold_column: "expected_fold_model"}
        )
        assignments["Country"] = (
            assignments["Country"].astype(str).str.strip()
        )
        assignments["expected_fold_model"] = pd.to_numeric(
            assignments["expected_fold_model"],
            errors="raise",
        ).astype(int)

    assignments = assignments.drop_duplicates().copy()
    duplicated = assignments.duplicated("Country", keep=False)
    if duplicated.any():
        examples = assignments.loc[duplicated].head(20)
        raise ValueError(
            "Country fold file assigns countries more than once:\n"
            + examples.to_string(index=False)
        )
    return assignments


def validate_against_fold_file(
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    table_name: str,
) -> None:
    comparison = predictions[
        ["Country", "fold_model"]
    ].drop_duplicates().merge(
        assignments,
        on="Country",
        how="left",
        validate="one_to_one",
    )
    missing = comparison["expected_fold_model"].isna()
    if missing.any():
        countries = comparison.loc[missing, "Country"].head(20).tolist()
        raise ValueError(
            f"{table_name} contains countries absent from the fold file: "
            f"{countries}"
        )

    bad = comparison["fold_model"].ne(
        comparison["expected_fold_model"].astype(int)
    )
    if bad.any():
        examples = comparison.loc[bad].head(20)
        raise ValueError(
            f"{table_name} disagrees with the supplied fold file:\n"
            + examples.to_string(index=False)
        )


def beta_binomial_nll(
    n_s: np.ndarray,
    n_total: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    k = np.asarray(n_s, dtype=float)
    n = np.asarray(n_total, dtype=float)
    a = np.asarray(alpha, dtype=float)
    b = np.asarray(beta, dtype=float)

    return -(
        gammaln(n + 1)
        - gammaln(k + 1)
        - gammaln(n - k + 1)
        + betaln(k + a, n - k + b)
        - betaln(a, b)
    )


def compute_metrics(
    frame: pd.DataFrame,
    *,
    observed_column: str,
    prediction_column: str,
    n_s_column: str,
    n_total_column: str,
    alpha_column: str,
    beta_column: str,
) -> dict[str, object]:
    if frame.empty:
        raise ValueError("Cannot calculate metrics on an empty frame.")

    observed = frame[observed_column].to_numpy(dtype=float)
    predicted = frame[prediction_column].to_numpy(dtype=float)
    n_s = frame[n_s_column].to_numpy(dtype=float)
    n_total = frame[n_total_column].to_numpy(dtype=float)
    alpha = frame[alpha_column].to_numpy(dtype=float)
    beta = frame[beta_column].to_numpy(dtype=float)

    error = observed - predicted
    absolute_error = np.abs(error)
    squared_error = error**2
    sqrt_weights = np.sqrt(n_total)

    nll = beta_binomial_nll(
        n_s=n_s,
        n_total=n_total,
        alpha=alpha,
        beta=beta,
    )

    return {
        "n_cells": int(len(frame)),
        "n_tests": int(round(float(n_total.sum()))),
        "n_countries": int(frame["Country"].nunique()),
        "weighted_mae": float(
            np.sum(sqrt_weights * absolute_error) / np.sum(sqrt_weights)
        ),
        "weighted_rmse": float(
            math.sqrt(
                np.sum(sqrt_weights * squared_error)
                / np.sum(sqrt_weights)
            )
        ),
        "beta_binomial_nll_per_test": float(
            np.sum(nll) / np.sum(n_total)
        ),
        "unweighted_mae": float(np.mean(absolute_error)),
        "unweighted_rmse": float(math.sqrt(np.mean(squared_error))),
        "mean_signed_error_observed_minus_predicted": float(
            np.mean(error)
        ),
    }


def metrics_by_fold(
    frame: pd.DataFrame,
    *,
    task_name: str,
    evaluation_family: str,
    fold_index: dict[int, int],
    metric_columns: dict[str, str],
    year_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for fold_model, group in frame.groupby("fold_model", sort=True):
        row: dict[str, object] = {
            "evaluation_family": evaluation_family,
            "evaluation_task": task_name,
            "model_name": "GNN",
            "fold_model": int(fold_model),
            "fold": int(fold_index[int(fold_model)]),
            "year_min": int(group[year_column].min()),
            "year_max": int(group[year_column].max()),
            "n_years": int(group[year_column].nunique()),
        }
        row.update(
            compute_metrics(
                group,
                observed_column=metric_columns["observed"],
                prediction_column=metric_columns["prediction"],
                n_s_column=metric_columns["n_s"],
                n_total_column=metric_columns["n_total"],
                alpha_column=metric_columns["alpha"],
                beta_column=metric_columns["beta"],
            )
        )
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    if len(result) != len(fold_index):
        raise ValueError(
            f"{evaluation_family} {task_name} produced {len(result)} fold "
            f"metrics instead of {len(fold_index)}."
        )
    return result


def mean_std_across_folds(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if fold_metrics.empty:
        raise ValueError("Fold metric table is empty.")

    row: dict[str, object] = {
        "evaluation_family": fold_metrics["evaluation_family"].iloc[0],
        "evaluation_task": fold_metrics["evaluation_task"].iloc[0],
        "model_name": fold_metrics["model_name"].iloc[0],
        "n_folds": int(fold_metrics["fold"].nunique()),
        "n_cells_total": int(fold_metrics["n_cells"].sum()),
        "n_tests_total": int(fold_metrics["n_tests"].sum()),
        "n_countries_total": int(fold_metrics["n_countries"].sum()),
        "year_min": int(fold_metrics["year_min"].min()),
        "year_max": int(fold_metrics["year_max"].max()),
    }

    metric_names = PRIMARY_METRICS + [
        "unweighted_mae",
        "unweighted_rmse",
        "mean_signed_error_observed_minus_predicted",
    ]
    for metric in metric_names:
        values = pd.to_numeric(
            fold_metrics[metric],
            errors="coerce",
        ).dropna()
        row[f"{metric}_mean"] = (
            float(values.mean()) if not values.empty else np.nan
        )
        row[f"{metric}_std"] = (
            float(values.std(ddof=1)) if len(values) > 1 else np.nan
        )
        row[f"{metric}_n_valid_folds"] = int(len(values))
        row[f"{metric}_mean_plus_minus_std"] = (
            f"{row[f'{metric}_mean']:.6f} ± "
            f"{row[f'{metric}_std']:.6f}"
            if np.isfinite(row[f"{metric}_mean"])
            and np.isfinite(row[f"{metric}_std"])
            else ""
        )

    return pd.DataFrame([row])


def prepare_temporal_targets(
    projection: pd.DataFrame,
    reconstruction: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = reconstruction[
        [
            "Country",
            "Year",
            "Species",
            "Family",
            "n_S",
            "n_total",
            "prop_S_observed",
            "fold_model",
        ]
    ].copy()
    target = target.rename(
        columns={
            "Year": "year_to",
            "n_S": "target_n_S",
            "n_total": "target_n_total",
            "prop_S_observed": "target_prop_S_observed",
            "fold_model": "target_fold_model",
        }
    )

    joined = projection.merge(
        target,
        on=["Country", "year_to", "Species", "Family"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    joined["exact_target_observation_available"] = joined["_merge"].eq("both")
    joined["fold_matches_target"] = (
        joined["target_fold_model"].isna()
        | joined["fold_model"].eq(joined["target_fold_model"])
    )

    disagreement = (
        joined["exact_target_observation_available"]
        & ~joined["fold_matches_target"]
    )
    if disagreement.any():
        examples = joined.loc[
            disagreement,
            [
                "Country",
                "year_to",
                "Species",
                "Family",
                "fold_model",
                "target_fold_model",
            ],
        ].head(20)
        raise ValueError(
            "Projection and reconstruction fold assignments disagree:\n"
            + examples.to_string(index=False)
        )

    report = (
        joined.groupby(
            [
                "fold_model",
                "year_to",
                "next_year_in_data",
                "exact_target_observation_available",
            ],
            dropna=False,
            sort=True,
        )
        .size()
        .reset_index(name="n_rows")
    )

    matched = joined.loc[
        joined["next_year_in_data"]
        & joined["exact_target_observation_available"]
    ].copy()
    matched = matched.drop(columns=["_merge"])
    return matched, report


def save_task_outputs(
    output_dir: Path,
    prefix: str,
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[Path, Path]:
    fold_path = output_dir / f"{prefix}_metrics_by_fold.csv"
    summary_path = (
        output_dir
        / f"{prefix}_metrics_mean_std_across_folds.csv"
    )
    fold_metrics.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    return fold_path, summary_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reconstruction = load_reconstruction(args.reconstruction_path)
    projection = load_projection(args.projection_path)

    validate_country_fold_consistency(
        reconstruction,
        "reconstruction",
    )
    validate_country_fold_consistency(
        projection,
        "projection",
    )

    reconstruction_countries = set(reconstruction["Country"])
    projection_countries = set(projection["Country"])
    if reconstruction_countries != projection_countries:
        missing_projection = sorted(
            reconstruction_countries - projection_countries
        )
        missing_reconstruction = sorted(
            projection_countries - reconstruction_countries
        )
        raise ValueError(
            "Reconstruction and projection country sets differ. "
            f"Missing from projection: {missing_projection[:20]}. "
            f"Missing from reconstruction: {missing_reconstruction[:20]}."
        )

    fold_index = make_fold_index(
        reconstruction["fold_model"].astype(int).tolist()
        + projection["fold_model"].astype(int).tolist(),
        expected_folds=args.expected_folds,
    )

    if args.country_folds_path is not None:
        assignments = load_external_fold_assignments(
            args.country_folds_path
        )
        validate_against_fold_file(
            reconstruction,
            assignments,
            "reconstruction",
        )
        validate_against_fold_file(
            projection,
            assignments,
            "projection",
        )

    reconstruction = reconstruction.copy()
    reconstruction["fold"] = (
        reconstruction["fold_model"].map(fold_index).astype(int)
    )
    projection = projection.copy()
    projection["fold"] = (
        projection["fold_model"].map(fold_index).astype(int)
    )

    temporal_matched, temporal_matching = prepare_temporal_targets(
        projection,
        reconstruction,
    )
    temporal_matched["fold"] = (
        temporal_matched["fold_model"].map(fold_index).astype(int)
    )

    vault_years = set(args.vault_years)

    reconstruction_historical = reconstruction.loc[
        reconstruction["Year"].le(args.historical_max_year)
    ].copy()
    reconstruction_vault = reconstruction.loc[
        reconstruction["Year"].isin(vault_years)
    ].copy()

    temporal_historical = temporal_matched.loc[
        temporal_matched["year_to"].le(args.historical_max_year)
    ].copy()
    temporal_vault = temporal_matched.loc[
        temporal_matched["year_to"].isin(vault_years)
    ].copy()

    task_frames = {
        "gnn_reconstruction_country_generalization": (
            reconstruction_historical,
            HISTORICAL_TASK,
            "reconstruction",
            {
                "observed": "prop_S_observed",
                "prediction": "prop_S_pred",
                "n_s": "n_S",
                "n_total": "n_total",
                "alpha": "alpha_cal",
                "beta": "beta_cal",
            },
            "Year",
        ),
        "gnn_reconstruction_country_year_generalization": (
            reconstruction_vault,
            VAULT_TASK,
            "reconstruction",
            {
                "observed": "prop_S_observed",
                "prediction": "prop_S_pred",
                "n_s": "n_S",
                "n_total": "n_total",
                "alpha": "alpha_cal",
                "beta": "beta_cal",
            },
            "Year",
        ),
        "gnn_temporal_country_generalization": (
            temporal_historical,
            HISTORICAL_TASK,
            "temporal_projection",
            {
                "observed": "target_prop_S_observed",
                "prediction": "prop_S_pred_next",
                "n_s": "target_n_S",
                "n_total": "target_n_total",
                "alpha": "alpha_cal",
                "beta": "beta_cal",
            },
            "year_to",
        ),
        "gnn_temporal_country_year_generalization": (
            temporal_vault,
            VAULT_TASK,
            "temporal_projection",
            {
                "observed": "target_prop_S_observed",
                "prediction": "prop_S_pred_next",
                "n_s": "target_n_S",
                "n_total": "target_n_total",
                "alpha": "alpha_cal",
                "beta": "beta_cal",
            },
            "year_to",
        ),
    }

    combined_fold_parts: list[pd.DataFrame] = []
    combined_summary_parts: list[pd.DataFrame] = []
    output_files: dict[str, str] = {}

    for prefix, (
        task_frame,
        task_name,
        evaluation_family,
        metric_columns,
        year_column,
    ) in task_frames.items():
        if task_frame.empty:
            raise ValueError(f"No rows are available for {prefix}.")

        fold_metrics = metrics_by_fold(
            task_frame,
            task_name=task_name,
            evaluation_family=evaluation_family,
            fold_index=fold_index,
            metric_columns=metric_columns,
            year_column=year_column,
        )
        summary = mean_std_across_folds(fold_metrics)

        fold_path, summary_path = save_task_outputs(
            args.output_dir,
            prefix,
            fold_metrics,
            summary,
        )
        output_files[f"{prefix}_by_fold"] = str(fold_path)
        output_files[f"{prefix}_mean_std"] = str(summary_path)
        combined_fold_parts.append(fold_metrics)
        combined_summary_parts.append(summary)

    all_fold_metrics = pd.concat(
        combined_fold_parts,
        ignore_index=True,
        sort=False,
    )
    all_summaries = pd.concat(
        combined_summary_parts,
        ignore_index=True,
        sort=False,
    )

    all_fold_path = (
        args.output_dir
        / "gnn_all_generalization_metrics_by_fold.csv"
    )
    all_summary_path = (
        args.output_dir
        / "gnn_all_generalization_metrics_mean_std_across_folds.csv"
    )
    all_fold_metrics.to_csv(all_fold_path, index=False)
    all_summaries.to_csv(all_summary_path, index=False)
    output_files["all_metrics_by_fold"] = str(all_fold_path)
    output_files["all_metrics_mean_std"] = str(all_summary_path)

    reconstruction_standardized_path = (
        args.output_dir
        / "gnn_reconstruction_standardized_predictions.csv"
    )
    temporal_standardized_path = (
        args.output_dir
        / "gnn_temporal_standardized_predictions.csv"
    )
    temporal_matching_path = (
        args.output_dir
        / "gnn_temporal_target_matching.csv"
    )

    reconstruction.to_csv(
        reconstruction_standardized_path,
        index=False,
    )
    temporal_matched.to_csv(
        temporal_standardized_path,
        index=False,
    )
    temporal_matching.to_csv(
        temporal_matching_path,
        index=False,
    )
    output_files["reconstruction_standardized"] = str(
        reconstruction_standardized_path
    )
    output_files["temporal_standardized"] = str(
        temporal_standardized_path
    )
    output_files["temporal_matching"] = str(
        temporal_matching_path
    )

    metadata = {
        "model_name": "GNN",
        "no_model_fitting_performed": True,
        "reconstruction_path": str(args.reconstruction_path),
        "projection_path": str(args.projection_path),
        "country_folds_path": (
            str(args.country_folds_path)
            if args.country_folds_path is not None
            else None
        ),
        "historical_max_year": int(args.historical_max_year),
        "vault_years": args.vault_years,
        "expected_folds": int(args.expected_folds),
        "fold_id_mapping_to_one_based_fold": {
            str(key): int(value)
            for key, value in fold_index.items()
        },
        "uncertainty_definition": (
            "sample standard deviation across the five fold metrics"
        ),
        "standard_deviation_ddof": 1,
        "error_weighting": "sqrt_n_total",
        "beta_binomial_nll_reduction": (
            "sum of cell beta binomial negative log likelihood divided "
            "by sum of target n_total"
        ),
        "temporal_target_join_keys": [
            "Country",
            "year_to",
            "Species",
            "Family",
        ],
        "temporal_rows_total": int(len(projection)),
        "temporal_rows_with_exact_target": int(
            temporal_matched.shape[0]
        ),
        "temporal_historical_rows": int(
            temporal_historical.shape[0]
        ),
        "temporal_vault_rows": int(
            temporal_vault.shape[0]
        ),
        "reconstruction_historical_rows": int(
            reconstruction_historical.shape[0]
        ),
        "reconstruction_vault_rows": int(
            reconstruction_vault.shape[0]
        ),
        "important_assumption": (
            "The saved GNN fold models were trained only on nonexternal "
            "countries and did not use observations after 2022 for fitting "
            "or model selection. Prediction files alone cannot verify this."
        ),
        "outputs": output_files,
    }
    metadata_path = (
        args.output_dir / "gnn_generalization_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf8",
    )
    output_files["metadata"] = str(metadata_path)

    display = all_summaries[
        [
            "evaluation_family",
            "evaluation_task",
            "weighted_mae_mean_plus_minus_std",
            "weighted_rmse_mean_plus_minus_std",
            "beta_binomial_nll_per_test_mean_plus_minus_std",
        ]
    ]
    print("\nGNN generalization summary")
    print("==========================")
    print(display.to_string(index=False))
    print("\nSaved files")
    for name, path in output_files.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
