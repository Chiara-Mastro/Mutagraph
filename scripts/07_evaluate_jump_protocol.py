#!/usr/bin/env python3
"""Evaluate the jump task with the original GNN notebook conventions.

This script replaces the previous jump evaluation and ranking scripts. It
computes only three reported metrics for each model and direction:

1. Recall at 20 with the original GNN watchlist protocol
2. AUROC independently inside each country fold, followed by mean and sample
   standard deviation across the five folds
3. MCC on the exact same rows and folds used for AUROC, followed by mean and
   sample standard deviation across the five folds

The observed GNN direction export is the canonical source of transition keys,
fold assignments, current susceptibility, and true direction labels. This
preserves the notebook treatment of transitions with insufficient tests as
stable rather than removing them.

Recall at 20 removes the latest two labelled target years by default, ranks
candidates inside Country and input year, retains units with at least 20
candidates and at least one true mover, and reports the direct mean and
population standard deviation across eligible unit recalls.

AUROC and MCC use all labelled transitions. Both metrics are calculated once
inside each of the five external country folds. The reported uncertainty is the
sample standard deviation across those five fold metrics. Probability scores
use an MCC threshold of 0.5. Continuous directional change scores use the jump
threshold, 0.10 by default.

The default output is one CSV:

    jump_metrics_gnn_protocol.csv

Optional row level and fold level files are written only when explicitly
requested.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TRANSITION_KEYS = ["Country", "input_year", "target_year", "Species", "Family"]
UNIT_COLUMNS = ["Country", "input_year"]
DIRECTIONS = ("down", "up")
TOP_K = 20
EXPECTED_FOLDS = 5

DOWN_PROBABILITY_COLUMNS = [
    "down_prob",
    "prob_down",
    "p_down",
    "resistance_jump_prob",
]
UP_PROBABILITY_COLUMNS = [
    "up_prob",
    "prob_up",
    "p_up",
    "susceptibility_jump_prob",
]
CONTINUOUS_PREDICTION_COLUMNS = [
    "p_pred",
    "p_pred_target",
    "predicted_prop_S",
    "prop_S_pred",
    "prop_S_pred_next",
    "predicted_susceptibility",
]
CURRENT_PROPORTION_COLUMNS = [
    "p_current",
    "prop_S_current",
    "source_p_current",
    "prop_S_prev",
]
MODEL_COLUMNS = ["model_name", "method", "model"]
FOLD_COLUMNS = ["fold_model", "colleague_fold", "external_fold", "fold"]


class SourceError(ValueError):
    """Raised when an input table violates the evaluation protocol."""


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for every source.")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("Both NAME and PATH must be nonempty.")
    return name, Path(raw_path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute GNN protocol Recall at 20, fold AUROC, and fold MCC for "
            "the GNN and additional jump models."
        )
    )
    parser.add_argument(
        "--gnn_predictions",
        type=Path,
        required=True,
        help=(
            "Observed GNN direction export containing true_direction, fold, "
            "down_prob, up_prob, and current susceptibility."
        ),
    )
    parser.add_argument(
        "--probability_predictions",
        nargs="*",
        type=parse_named_path,
        default=[],
        metavar="NAME=PATH",
        help="Additional models with down and up probabilities.",
    )
    parser.add_argument(
        "--continuous_predictions",
        nargs="*",
        type=parse_named_path,
        default=[],
        metavar="NAME=PATH",
        help="Continuous next year susceptibility predictions.",
    )
    parser.add_argument(
        "--continuous_models",
        nargs="*",
        default=None,
        help="Optional model names retained from continuous prediction tables.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--exclude_latest_target_years_for_recall",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--recall_target_years",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit target years for Recall at 20.",
    )
    parser.add_argument("--probability_threshold", type=float, default=0.50)
    parser.add_argument("--jump_threshold", type=float, default=0.10)
    parser.add_argument(
        "--require_full_coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require each model to score every canonical GNN transition.",
    )
    parser.add_argument(
        "--save_fold_metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save_standardized_scores",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if args.exclude_latest_target_years_for_recall < 0:
        raise ValueError("The Recall year exclusion must be nonnegative.")
    if not 0 < args.probability_threshold < 1:
        raise ValueError("probability_threshold must lie between zero and one.")
    if args.jump_threshold <= 0:
        raise ValueError("jump_threshold must be positive.")
    args.recall_target_years = (
        sorted(set(args.recall_target_years))
        if args.recall_target_years
        else None
    )
    return args


def read_csv(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise SourceError(f"{source_name} is empty.")
    return frame


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def boolean_series(values: pd.Series, source_name: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    mapped = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        examples = values.loc[mapped.isna()].head(20).tolist()
        raise SourceError(f"{source_name} contains invalid booleans: {examples}")
    return mapped.astype(bool)


def normalize_transition_columns(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    aliases = {
        "country": "Country",
        "species": "Species",
        "family": "Family",
        "year_from": "input_year",
        "year1": "input_year",
        "year_to": "target_year",
        "year2": "target_year",
        "Year": "target_year",
    }
    rename: dict[str, str] = {}
    for old, new in aliases.items():
        if old in frame.columns and new not in frame.columns:
            rename[old] = new
    out = frame.rename(columns=rename).copy()

    if "target_year" in out.columns and "input_year" not in out.columns:
        out["input_year"] = pd.to_numeric(out["target_year"], errors="coerce") - 1

    missing = [column for column in TRANSITION_KEYS if column not in out.columns]
    if missing:
        raise SourceError(
            f"{source_name} is missing transition columns {missing}. "
            f"Available columns: {list(out.columns)}"
        )

    for column in ["Country", "Species", "Family"]:
        out[column] = out[column].astype(str).str.strip()
    for column in ["input_year", "target_year"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=TRANSITION_KEYS).copy()
    out["input_year"] = out["input_year"].astype(int)
    out["target_year"] = out["target_year"].astype(int)

    nonconsecutive = out["target_year"].ne(out["input_year"] + 1)
    if nonconsecutive.any():
        examples = out.loc[nonconsecutive, TRANSITION_KEYS].head(20)
        raise SourceError(
            f"{source_name} contains nonconsecutive transitions:\n"
            + examples.to_string(index=False)
        )
    return out


def assert_unique(frame: pd.DataFrame, source_name: str) -> None:
    duplicated = frame.duplicated(TRANSITION_KEYS, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, TRANSITION_KEYS].head(20)
        raise SourceError(
            f"{source_name} contains duplicate transition keys:\n"
            + examples.to_string(index=False)
        )


def normalize_true_direction(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    normalized = normalized.replace({"": np.nan, "nan": np.nan, "none": np.nan})
    invalid = normalized.notna() & ~normalized.isin(["down", "stable", "up"])
    if invalid.any():
        examples = values.loc[invalid].head(20).tolist()
        raise SourceError(f"true_direction contains invalid values: {examples}")
    return normalized


def normalize_fold(frame: pd.DataFrame, source_name: str) -> pd.Series:
    column = first_existing(frame.columns, FOLD_COLUMNS)
    if column is None:
        raise SourceError(f"{source_name} needs one recognized fold column.")
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise SourceError(f"{source_name}.{column} contains missing fold values.")
    unique = sorted(values.astype(int).unique().tolist())
    if unique == [0, 1, 2, 3, 4]:
        return values.astype(int) + 1
    if unique == [1, 2, 3, 4, 5]:
        return values.astype(int)
    raise SourceError(
        f"{source_name} contains folds {unique}, expected zero through four or one through five."
    )


def prepare_canonical_gnn(path: Path) -> tuple[pd.DataFrame, list[pd.DataFrame], list[int]]:
    raw = normalize_transition_columns(read_csv(path, "gnn_predictions"), "gnn_predictions")
    raw["source_row_order"] = np.arange(len(raw), dtype=int)

    if "next_year_in_data" in raw.columns:
        raw = raw.loc[
            boolean_series(raw["next_year_in_data"], "next_year_in_data")
        ].copy()

    assert_unique(raw, "gnn_predictions")
    if "true_direction" not in raw.columns:
        raise SourceError("gnn_predictions must contain true_direction.")
    raw["true_direction"] = normalize_true_direction(raw["true_direction"])
    raw = raw.loc[raw["true_direction"].notna()].copy()

    down_column = first_existing(raw.columns, DOWN_PROBABILITY_COLUMNS)
    up_column = first_existing(raw.columns, UP_PROBABILITY_COLUMNS)
    current_column = first_existing(raw.columns, CURRENT_PROPORTION_COLUMNS)
    if down_column is None or up_column is None:
        raise SourceError("gnn_predictions must contain down and up probabilities.")
    if current_column is None:
        raise SourceError("gnn_predictions must contain current susceptibility.")

    raw["fold"] = normalize_fold(raw, "gnn_predictions")
    raw["p_current_canonical"] = pd.to_numeric(raw[current_column], errors="coerce")
    raw["gnn_down_score"] = pd.to_numeric(raw[down_column], errors="coerce")
    raw["gnn_up_score"] = pd.to_numeric(raw[up_column], errors="coerce")
    raw = raw.dropna(
        subset=["p_current_canonical", "gnn_down_score", "gnn_up_score"]
    ).copy()

    raw["label_down"] = raw["true_direction"].eq("down").astype(int)
    raw["label_up"] = raw["true_direction"].eq("up").astype(int)

    canonical = raw[
        TRANSITION_KEYS
        + [
            "source_row_order",
            "fold",
            "p_current_canonical",
            "true_direction",
            "label_down",
            "label_up",
            "gnn_down_score",
            "gnn_up_score",
        ]
    ].copy()

    parts: list[pd.DataFrame] = []
    for direction, score_column in [
        ("down", "gnn_down_score"),
        ("up", "gnn_up_score"),
    ]:
        part = canonical.copy()
        part["model_name"] = "GNN"
        part["direction"] = direction
        part["score"] = part[score_column].astype(float)
        part["label"] = part[f"label_{direction}"].astype(int)
        part["score_type"] = "probability"
        parts.append(part)

    years = sorted(canonical["target_year"].unique().tolist())
    return canonical, parts, years


def model_groups(
    frame: pd.DataFrame,
    external_name: str,
    allowed_models: set[str] | None = None,
) -> list[tuple[str, pd.DataFrame]]:
    model_column = first_existing(frame.columns, MODEL_COLUMNS)
    if model_column is None:
        return [(external_name, frame.copy())]

    values = frame[model_column].astype(str)
    unique_values = [value for value in sorted(values.unique()) if value and value != "nan"]
    if len(unique_values) <= 1:
        if allowed_models is None:
            return [(external_name, frame.copy())]
        if not unique_values or unique_values[0] not in allowed_models:
            return []
        return [(unique_values[0], frame.copy())]

    groups: list[tuple[str, pd.DataFrame]] = []
    for value, group in frame.groupby(model_column, sort=True, dropna=False):
        model_name = str(value)
        if allowed_models is not None and model_name not in allowed_models:
            continue
        groups.append((model_name, group.copy()))
    return groups


def source_with_optional_fold(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = frame.copy()
    fold_column = first_existing(out.columns, FOLD_COLUMNS)
    if fold_column is None:
        out["fold_source"] = np.nan
        return out
    raw_fold = pd.to_numeric(out[fold_column], errors="coerce")
    unique = sorted(raw_fold.dropna().astype(int).unique().tolist())
    if unique == [0, 1, 2, 3, 4]:
        out["fold_source"] = raw_fold + 1
    elif unique == [1, 2, 3, 4, 5]:
        out["fold_source"] = raw_fold
    else:
        raise SourceError(f"{source_name} contains unrecognized folds {unique}.")
    return out


def validate_fold_match(merged: pd.DataFrame, source_name: str) -> None:
    if "fold_source" not in merged.columns:
        return
    mismatch = (
        merged["fold_source"].notna()
        & merged["fold"].ne(merged["fold_source"].astype(int))
    )
    if mismatch.any():
        examples = merged.loc[
            mismatch,
            TRANSITION_KEYS + ["fold", "fold_source"],
        ].head(20)
        raise SourceError(
            f"{source_name} disagrees with the GNN fold assignment:\n"
            + examples.to_string(index=False)
        )


def probability_score_rows(
    external_name: str,
    path: Path,
    canonical: pd.DataFrame,
) -> list[pd.DataFrame]:
    raw = normalize_transition_columns(read_csv(path, external_name), external_name)
    parts: list[pd.DataFrame] = []

    for model_name, model_frame in model_groups(raw, external_name):
        assert_unique(model_frame, f"{external_name}:{model_name}")
        source = source_with_optional_fold(model_frame, f"{external_name}:{model_name}")
        down_column = first_existing(source.columns, DOWN_PROBABILITY_COLUMNS)
        up_column = first_existing(source.columns, UP_PROBABILITY_COLUMNS)
        if down_column is None and up_column is None:
            continue

        keep = TRANSITION_KEYS + ["fold_source"]
        for column in [down_column, up_column]:
            if column is not None and column not in keep:
                keep.append(column)
        merged = canonical.merge(
            source[keep],
            on=TRANSITION_KEYS,
            how="inner",
            validate="one_to_one",
        )
        validate_fold_match(merged, f"{external_name}:{model_name}")

        for direction, score_column in [("down", down_column), ("up", up_column)]:
            if score_column is None:
                continue
            part = merged.copy()
            part["model_name"] = model_name
            part["direction"] = direction
            part["score"] = pd.to_numeric(part[score_column], errors="coerce")
            part["label"] = part[f"label_{direction}"].astype(int)
            part["score_type"] = "probability"
            part = part.dropna(subset=["score"]).copy()
            if not part.empty:
                parts.append(part)
    return parts


def continuous_score_rows(
    external_name: str,
    path: Path,
    canonical: pd.DataFrame,
    allowed_models: set[str] | None,
) -> list[pd.DataFrame]:
    raw = normalize_transition_columns(read_csv(path, external_name), external_name)
    prediction_column = first_existing(raw.columns, CONTINUOUS_PREDICTION_COLUMNS)
    if prediction_column is None:
        raise SourceError(f"{external_name} has no recognized target prediction column.")

    parts: list[pd.DataFrame] = []
    for model_name, model_frame in model_groups(raw, external_name, allowed_models):
        assert_unique(model_frame, f"{external_name}:{model_name}")
        source = source_with_optional_fold(model_frame, f"{external_name}:{model_name}")
        source["predicted_target"] = pd.to_numeric(
            source[prediction_column], errors="coerce"
        )
        source = source.dropna(subset=["predicted_target"]).copy()
        source["predicted_target"] = source["predicted_target"].clip(0.0, 1.0)

        merged = canonical.merge(
            source[TRANSITION_KEYS + ["fold_source", "predicted_target"]],
            on=TRANSITION_KEYS,
            how="inner",
            validate="one_to_one",
        )
        validate_fold_match(merged, f"{external_name}:{model_name}")
        merged["predicted_delta"] = (
            merged["predicted_target"] - merged["p_current_canonical"]
        )

        for direction in DIRECTIONS:
            part = merged.copy()
            part["model_name"] = model_name
            part["direction"] = direction
            part["score"] = (
                -part["predicted_delta"]
                if direction == "down"
                else part["predicted_delta"]
            )
            part["label"] = part[f"label_{direction}"].astype(int)
            part["score_type"] = "forecasted_directional_change"
            parts.append(part)
    return parts


def combine_scores(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        raise RuntimeError("No model scores were produced.")
    scores = pd.concat(parts, ignore_index=True, sort=False)
    duplicated = scores.duplicated(
        ["model_name", "direction"] + TRANSITION_KEYS,
        keep=False,
    )
    if duplicated.any():
        examples = scores.loc[
            duplicated,
            ["model_name", "direction"] + TRANSITION_KEYS,
        ].head(20)
        raise SourceError(
            "Duplicate model direction transitions were produced:\n"
            + examples.to_string(index=False)
        )
    return scores


def choose_recall_years(
    available_years: list[int],
    explicit_years: list[int] | None,
    exclude_latest: int,
) -> list[int]:
    if explicit_years is not None:
        missing = sorted(set(explicit_years) - set(available_years))
        if missing:
            raise SourceError(f"Requested Recall target years are unavailable: {missing}")
        return explicit_years
    if exclude_latest == 0:
        return available_years
    if len(available_years) <= exclude_latest:
        raise SourceError("Not enough years remain after the Recall year exclusion.")
    return available_years[:-exclude_latest]


def verify_coverage(
    scores: pd.DataFrame,
    canonical: pd.DataFrame,
    require_full: bool,
) -> None:
    canonical_keys = set(map(tuple, canonical[TRANSITION_KEYS].to_numpy()))
    for (model_name, direction), group in scores.groupby(
        ["model_name", "direction"], sort=True
    ):
        model_keys = set(map(tuple, group[TRANSITION_KEYS].to_numpy()))
        missing = canonical_keys - model_keys
        extra = model_keys - canonical_keys
        if require_full and (missing or extra):
            examples = list(missing)[:5]
            raise SourceError(
                f"{model_name}, {direction} does not cover the full GNN population. "
                f"Missing {len(missing)} rows and found {len(extra)} extra rows. "
                f"Missing examples: {examples}"
            )


def recall_at_20(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_name, direction), model_frame in scores.groupby(
        ["model_name", "direction"], sort=True
    ):
        unit_values: list[float] = []
        for _, group in model_frame.groupby(UNIT_COLUMNS, sort=True, dropna=False):
            if len(group) < TOP_K:
                continue
            n_positive = int(group["label"].sum())
            if n_positive == 0:
                continue
            ordered = group.sort_values(
                ["score", "source_row_order"],
                ascending=[False, True],
                kind="mergesort",
            )
            hits = int(ordered.head(TOP_K)["label"].sum())
            unit_values.append(float(hits / n_positive))

        values = np.asarray(unit_values, dtype=float)
        rows.append(
            {
                "model_name": model_name,
                "direction": direction,
                "recall_at_20_mean": float(values.mean()) if len(values) else np.nan,
                "recall_at_20_sd": float(values.std(ddof=0)) if len(values) else np.nan,
                "recall_at_20_n_units": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def mcc_from_threshold(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> tuple[float, int, int, int, int]:
    y_pred = score >= threshold
    positive = y_true == 1
    negative = ~positive
    tp = int(np.sum(y_pred & positive))
    fp = int(np.sum(y_pred & negative))
    fn = int(np.sum(~y_pred & positive))
    tn = int(np.sum(~y_pred & negative))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    value = float((tp * tn - fp * fn) / denominator) if denominator > 0 else np.nan
    return value, tn, fp, fn, tp


def fold_auroc_and_mcc(
    scores: pd.DataFrame,
    probability_threshold: float,
    jump_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = ["model_name", "direction", "fold"]
    for keys, group in scores.groupby(groups, sort=True, dropna=False):
        model_name, direction, fold = keys
        clean = group.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["label", "score"]
        )
        y_true = clean["label"].astype(int).to_numpy()
        y_score = clean["score"].astype(float).to_numpy()
        n_positive = int(y_true.sum())
        n_negative = int(len(y_true) - n_positive)
        auroc = (
            float(roc_auc_score(y_true, y_score))
            if n_positive > 0 and n_negative > 0
            else np.nan
        )

        score_types = clean["score_type"].dropna().astype(str).unique().tolist()
        if len(score_types) != 1:
            raise SourceError(
                f"{model_name}, {direction}, fold {fold} has score types {score_types}."
            )
        score_type = score_types[0]
        threshold = (
            probability_threshold
            if score_type == "probability"
            else jump_threshold
        )
        mcc, tn, fp, fn, tp = mcc_from_threshold(y_true, y_score, threshold)
        rows.append(
            {
                "model_name": model_name,
                "direction": direction,
                "fold": int(fold),
                "auroc": auroc,
                "mcc": mcc,
                "mcc_threshold": float(threshold),
                "score_type": score_type,
                "n_rows": int(len(clean)),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
    result = pd.DataFrame(rows)
    observed = sorted(result["fold"].unique().tolist())
    if observed != list(range(1, EXPECTED_FOLDS + 1)):
        raise RuntimeError(f"Observed folds {observed}, expected one through five.")
    return result


def summarize_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_name, direction), group in fold_metrics.groupby(
        ["model_name", "direction"], sort=True
    ):
        if group["fold"].nunique() != EXPECTED_FOLDS:
            raise RuntimeError(
                f"{model_name}, {direction} does not contain five fold metrics."
            )
        auroc_values = pd.to_numeric(group["auroc"], errors="coerce").dropna()
        mcc_values = pd.to_numeric(group["mcc"], errors="coerce").dropna()
        thresholds = pd.to_numeric(group["mcc_threshold"], errors="coerce").dropna().unique()
        score_types = group["score_type"].dropna().astype(str).unique().tolist()

        rows.append(
            {
                "model_name": model_name,
                "direction": direction,
                "auroc_mean": float(auroc_values.mean()) if len(auroc_values) else np.nan,
                "auroc_sd": float(auroc_values.std(ddof=1)) if len(auroc_values) > 1 else np.nan,
                "auroc_n_folds": int(len(auroc_values)),
                "mcc_mean": float(mcc_values.mean()) if len(mcc_values) else np.nan,
                "mcc_sd": float(mcc_values.std(ddof=1)) if len(mcc_values) > 1 else np.nan,
                "mcc_n_folds": int(len(mcc_values)),
                "mcc_threshold": float(thresholds[0]) if len(thresholds) == 1 else np.nan,
                "score_type": score_types[0] if len(score_types) == 1 else "mixed",
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    recall: pd.DataFrame,
    fold_summary: pd.DataFrame,
    recall_years: list[int],
    all_years: list[int],
) -> pd.DataFrame:
    out = recall.merge(
        fold_summary,
        on=["model_name", "direction"],
        how="outer",
        validate="one_to_one",
    )
    out["recall_at_20_mean_plus_minus_sd"] = out.apply(
        lambda row: (
            f"{row['recall_at_20_mean']:.6f} ± {row['recall_at_20_sd']:.6f}"
            if np.isfinite(row["recall_at_20_mean"])
            and np.isfinite(row["recall_at_20_sd"])
            else ""
        ),
        axis=1,
    )
    out["auroc_mean_plus_minus_sd"] = out.apply(
        lambda row: (
            f"{row['auroc_mean']:.6f} ± {row['auroc_sd']:.6f}"
            if np.isfinite(row["auroc_mean"]) and np.isfinite(row["auroc_sd"])
            else ""
        ),
        axis=1,
    )
    out["mcc_mean_plus_minus_sd"] = out.apply(
        lambda row: (
            f"{row['mcc_mean']:.6f} ± {row['mcc_sd']:.6f}"
            if np.isfinite(row["mcc_mean"]) and np.isfinite(row["mcc_sd"])
            else ""
        ),
        axis=1,
    )
    out["recall_target_year_min"] = int(min(recall_years))
    out["recall_target_year_max"] = int(max(recall_years))
    out["auroc_mcc_target_year_min"] = int(min(all_years))
    out["auroc_mcc_target_year_max"] = int(max(all_years))
    out["recall_uncertainty"] = (
        "population standard deviation across eligible Country input year recalls"
    )
    out["auroc_uncertainty"] = (
        "sample standard deviation across five fold AUROCs"
    )
    out["mcc_uncertainty"] = (
        "sample standard deviation across five fold MCC values"
    )

    preferred = [
        "model_name",
        "direction",
        "recall_at_20_mean",
        "recall_at_20_sd",
        "recall_at_20_mean_plus_minus_sd",
        "auroc_mean",
        "auroc_sd",
        "auroc_mean_plus_minus_sd",
        "mcc_mean",
        "mcc_sd",
        "mcc_mean_plus_minus_sd",
        "mcc_threshold",
        "score_type",
        "recall_at_20_n_units",
        "auroc_n_folds",
        "mcc_n_folds",
        "recall_target_year_min",
        "recall_target_year_max",
        "auroc_mcc_target_year_min",
        "auroc_mcc_target_year_max",
        "recall_uncertainty",
        "auroc_uncertainty",
        "mcc_uncertainty",
    ]
    remaining = [column for column in out.columns if column not in preferred]
    return out[preferred + remaining].sort_values(
        ["model_name", "direction"]
    ).reset_index(drop=True)


def print_gnn_checks(summary: pd.DataFrame) -> None:
    expected_recall = {"down": 0.644, "up": 0.763}
    expected_down_auroc = 0.730
    gnn = summary.loc[summary["model_name"].eq("GNN")]
    print("\nGNN reproduction checks")
    print("=======================")
    for direction in DIRECTIONS:
        row = gnn.loc[gnn["direction"].eq(direction)]
        if len(row) != 1:
            print(f"{direction}: missing")
            continue
        recall_value = float(row["recall_at_20_mean"].iloc[0])
        auroc_value = float(row["auroc_mean"].iloc[0])
        recall_ok = round(recall_value, 3) == expected_recall[direction]
        auroc_note = ""
        if direction == "down":
            auroc_note = (
                "matches 0.730"
                if round(auroc_value, 3) == expected_down_auroc
                else "does not match 0.730"
            )
        print(
            f"{direction}: Recall at 20 {recall_value:.6f} "
            f"({'matches' if recall_ok else 'does not match'} reference); "
            f"AUROC {auroc_value:.6f} {auroc_note}".rstrip()
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    canonical, parts, all_years = prepare_canonical_gnn(args.gnn_predictions)

    for name, path in args.probability_predictions:
        parts.extend(probability_score_rows(name, path, canonical))

    allowed_models = set(args.continuous_models) if args.continuous_models else None
    for name, path in args.continuous_predictions:
        parts.extend(
            continuous_score_rows(name, path, canonical, allowed_models)
        )

    scores = combine_scores(parts)
    verify_coverage(scores, canonical, args.require_full_coverage)

    recall_years = choose_recall_years(
        all_years,
        args.recall_target_years,
        args.exclude_latest_target_years_for_recall,
    )
    recall_scores = scores.loc[scores["target_year"].isin(recall_years)].copy()
    recall = recall_at_20(recall_scores)

    fold_metrics = fold_auroc_and_mcc(
        scores,
        probability_threshold=args.probability_threshold,
        jump_threshold=args.jump_threshold,
    )
    fold_summary = summarize_folds(fold_metrics)
    summary = build_summary(recall, fold_summary, recall_years, all_years)

    output_path = args.output_dir / "jump_metrics_gnn_protocol.csv"
    summary.to_csv(output_path, index=False)

    if args.save_fold_metrics:
        fold_metrics.to_csv(
            args.output_dir / "jump_metrics_gnn_protocol_by_fold.csv",
            index=False,
        )
    if args.save_standardized_scores:
        scores.to_csv(
            args.output_dir / "jump_scores_gnn_protocol.csv",
            index=False,
        )

    print("\nJump task metrics")
    print("=================")
    display_columns = [
        "model_name",
        "direction",
        "recall_at_20_mean_plus_minus_sd",
        "auroc_mean_plus_minus_sd",
        "mcc_mean_plus_minus_sd",
        "mcc_threshold",
    ]
    print(summary[display_columns].to_string(index=False))
    print_gnn_checks(summary)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
