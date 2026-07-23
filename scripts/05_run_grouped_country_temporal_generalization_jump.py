#!/usr/bin/env python3
"""Train frozen temporal jump heads with two external evaluation tasks.

For every country fold supplied by the GNN assignment file, this script loads
the single residual checkpoint produced by script 04, freezes the complete
backbone, and trains only the down and up direction heads.

The heads use training country transitions with target year before 2022 for
gradient updates and target year 2022 from training countries for checkpoint
selection. The selected heads are evaluated on the held out fold countries in
two separate tasks:

1. country generalisation on historical target years through 2022
2. country and year generalisation on target years 2023 and 2024

One pair of heads is fitted per fold. The same selected heads are used for both
tests. Final uncertainty is the sample standard deviation of the five fold
level metrics.
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

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT_FROM_SCRIPT))

from src.models import SnapshotEncoderResidualModel
from src.temporal_jump_dataset import build_temporal_jump_dataloaders
from src.temporal_jump_training import (
    classification_metrics_from_frame,
    estimate_jump_pos_weights,
    evaluate_jump_heads,
    selection_score,
    train_one_epoch_jump_heads,
)
from src.temporal_residual_jump_model import TemporalResidualJumpHeadsModel
from src.utils import choose_device, set_seed


TRAIN_MAX_YEAR = 2022
VALIDATION_TARGET_YEAR = 2022
TEST_YEARS = (2023, 2024)
EXPECTED_N_FOLDS = 5
EVALUATION_SET_COUNTRY = "historical_country_generalization_through_2022"
EVALUATION_SET_COUNTRY_YEAR = "country_and_year_generalization_2023_2024"
EVALUATION_SET_FUTURE = "prospective_forecast"
EVALUATION_PROTOCOL = (
    "frozen_direction_heads_gnn_country_five_fold_dual_external_tests"
)
MODEL_NAME = "frozen_temporal_residual_jump_heads"

CELL_INTEGER_COLUMNS = ["Year", "species_idx", "family_idx", "cell_row_id"]
CELL_FLOAT_COLUMNS = [
    "n_S",
    "n_total",
    "prop_S",
    "p_baseline",
    "baseline_logit",
    "residual_prop_S",
]
PAIR_INTEGER_COLUMNS = ["input_year", "target_year"]


def load_temporal_helpers():
    helper_path = Path(__file__).with_name("03_temporal_country_helpers.py")
    if not helper_path.exists():
        raise FileNotFoundError(
            "Expected 03_temporal_country_helpers.py next to this script."
        )

    spec = importlib.util.spec_from_file_location(
        "temporal_country_helpers_jump_fixed_vault",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one frozen pair of direction heads per GNN country fold and "
            "evaluate historical country generalisation and 2023 to 2024 "
            "country and year generalisation."
        )
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument(
        "--temporal_model_dir",
        type=Path,
        required=True,
        help="Output directory produced by script 04 with the same GNN folds.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--country_folds_path",
        type=Path,
        required=True,
        help="Exact five fold country assignment used by the GNN and script 04.",
    )

    parser.add_argument("--forecast_input_year", type=int, default=2024)
    parser.add_argument("--forecast_year", type=int, default=2025)
    parser.add_argument(
        "--run_future",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--jump_threshold", type=float, default=0.10)
    parser.add_argument("--jump_min_tests", type=float, default=80.0)
    parser.add_argument(
        "--jump_min_tests_rule",
        choices=["inclusive", "exclusive"],
        default="inclusive",
    )
    parser.add_argument(
        "--jump_label_policy",
        choices=["gnn_stable", "eligible_only"],
        default="gnn_stable",
    )
    parser.add_argument("--down_loss_weight", type=float, default=1.0)
    parser.add_argument("--up_loss_weight", type=float, default=1.0)

    parser.add_argument("--head_hidden_dim", type=int, default=64)
    parser.add_argument("--head_bottleneck_dim", type=int, default=16)
    parser.add_argument("--head_dropout", type=float, default=0.10)
    parser.add_argument("--species_embedding_name", type=str, default=None)
    parser.add_argument("--family_embedding_name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--gradient_clip", type=float, default=5.0)
    parser.add_argument(
        "--checkpoint_metric",
        choices=[
            "mean_average_precision",
            "down_average_precision",
            "mean_auroc",
            "negative_combined_bce",
        ],
        default="mean_average_precision",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--save_models",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--continue_on_error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()
    if args.jump_threshold <= 0:
        raise ValueError("jump_threshold must be positive.")
    if args.jump_min_tests < 0:
        raise ValueError("jump_min_tests must be nonnegative.")
    if args.down_loss_weight < 0 or args.up_loss_weight < 0:
        raise ValueError("Direction loss weights must be nonnegative.")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive.")
    if args.forecast_year != args.forecast_input_year + 1:
        raise ValueError("Only one year future forecasts are supported.")
    if args.forecast_input_year < max(TEST_YEARS):
        raise ValueError("forecast_input_year must be at least 2024.")
    return args

def build_country_folds(
    full_df: pd.DataFrame,
    *,
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

def _coerce_inputs(
    cells: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_cells = cells.copy()
    clean_pairs = pairs.copy()

    for column in CELL_INTEGER_COLUMNS:
        clean_cells[column] = pd.to_numeric(
            clean_cells[column], errors="raise"
        ).to_numpy(dtype=np.int64)

    for column in CELL_FLOAT_COLUMNS:
        values = pd.to_numeric(
            clean_cells[column], errors="raise"
        ).to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Cell column {column!r} contains nonfinite values."
            )
        clean_cells[column] = values

    for column in PAIR_INTEGER_COLUMNS:
        clean_pairs[column] = pd.to_numeric(
            clean_pairs[column], errors="raise"
        ).to_numpy(dtype=np.int64)

    for column in ["Country", "Species", "Family"]:
        clean_cells[column] = clean_cells[column].astype(str)
    for column in ["pair_id", "Country", "split"]:
        clean_pairs[column] = clean_pairs[column].astype(str)

    return clean_cells, clean_pairs


def split_pairs_for_dual_evaluation(
    all_pairs: list[tuple[str, int, int]],
    *,
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
        raise ValueError("No direction training transitions before 2022.")
    if not val_pairs:
        raise ValueError("No direction checkpoint selection transitions for 2022.")
    if not historical_test_pairs:
        raise ValueError("No historical external direction test transitions.")
    if not country_year_test_pairs:
        raise ValueError("No external direction test transitions for 2023 or 2024.")

    vault_years = {int(pair[2]) for pair in country_year_test_pairs}
    missing = sorted(set(TEST_YEARS) - vault_years)
    if missing:
        raise ValueError(f"The external fold is missing transitions for {missing}.")

    selected = train_pairs + val_pairs + historical_test_pairs + country_year_test_pairs
    for country, input_year, target_year in selected:
        if int(input_year) != int(target_year) - 1:
            raise AssertionError(
                f"Nonconsecutive transition for {country}: {input_year}, {target_year}."
            )
    if max(pair[2] for pair in train_pairs) >= VALIDATION_TARGET_YEAR:
        raise AssertionError("Head gradient training reaches 2022 or later.")
    return train_pairs, val_pairs, historical_test_pairs, country_year_test_pairs

def build_pair_table(
    train_pairs: list[tuple[str, int, int]],
    val_pairs: list[tuple[str, int, int]],
    historical_test_pairs: list[tuple[str, int, int]],
    country_year_test_pairs: list[tuple[str, int, int]],
    *,
    fold: int,
) -> pd.DataFrame:
    test_pairs = historical_test_pairs + country_year_test_pairs
    table = HELPER.build_temporal_residual_pair_table(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
    )
    table["fold"] = int(fold)
    table["colleague_fold"] = int(fold - 1)
    table["training_data_max_year"] = int(TRAIN_MAX_YEAR)
    table["head_validation_target_year"] = int(VALIDATION_TARGET_YEAR)
    table["evaluation_set"] = np.select(
        [
            table["split"].eq("test") & table["target_year"].le(TRAIN_MAX_YEAR),
            table["split"].eq("test") & table["target_year"].isin(TEST_YEARS),
            table["split"].eq("train"),
            table["split"].eq("val"),
        ],
        [
            EVALUATION_SET_COUNTRY,
            EVALUATION_SET_COUNTRY_YEAR,
            "gradient_training",
            "checkpoint_selection",
        ],
        default="unknown",
    )
    return table


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
    *,
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

    provenance_columns = [
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
    available_provenance = [
        column for column in provenance_columns if column in baseline.columns
    ]
    cells = cells.merge(
        baseline[available_provenance],
        on=["Species", "Family", "Year"],
        how="left",
        validate="many_to_one",
    )
    cells["residual_prop_S"] = cells["prop_S"] - cells["p_baseline"]
    cells["baseline_training_max_year"] = int(TRAIN_MAX_YEAR)
    cells["baseline_excludes_external_fold"] = True

    if "baseline_history_max_year" in cells.columns:
        bad_vault_baseline = (
            cells["Year"].isin(TEST_YEARS)
            & cells["baseline_history_max_year"].notna()
            & cells["baseline_history_max_year"].gt(TRAIN_MAX_YEAR)
        )
        if bad_vault_baseline.any():
            raise AssertionError(
                "A vault baseline used training history after 2022."
            )

    return cells, training_df


def checkpoint_path_for_fixed_fold(
    root: Path,
    *,
    fold: int,
) -> Path:
    candidates = [
        root
        / f"fold_{fold}"
        / "fixed_historical_model"
        / "temporal_residual_encoder_model.pt",
        root
        / f"fold_{fold:02d}"
        / "fixed_historical_model"
        / "temporal_residual_encoder_model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Fixed historical residual checkpoint not found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _checkpoint_value(
    checkpoint_args: dict[str, object],
    name: str,
    fallback,
):
    value = checkpoint_args.get(name, fallback)
    return fallback if value is None else value


def validate_residual_checkpoint(
    checkpoint: dict[str, object],
    *,
    fold: int,
) -> None:
    checkpoint_fold = int(checkpoint.get("fold", fold))
    if checkpoint_fold != int(fold):
        raise ValueError(
            f"Residual checkpoint fold is {checkpoint_fold}, expected {fold}."
        )

    training_data_max_year = int(checkpoint.get("training_data_max_year", -1))
    selection_year = int(checkpoint.get("checkpoint_selection_target_year", -1))
    gradient_max_target_year = int(
        checkpoint.get("gradient_training_max_target_year", -1)
    )
    country_year_test_years = {
        int(value)
        for value in checkpoint.get(
            "country_year_test_target_years",
            checkpoint.get("external_test_target_years", []),
        )
        if int(value) in TEST_YEARS
    }

    if training_data_max_year != TRAIN_MAX_YEAR:
        raise ValueError("Residual checkpoint was not fitted through 2022 only.")
    if selection_year != VALIDATION_TARGET_YEAR:
        raise ValueError("Residual checkpoint was not selected on target year 2022.")
    if gradient_max_target_year >= VALIDATION_TARGET_YEAR:
        raise ValueError("Residual gradient training includes 2022 or later.")
    if country_year_test_years != set(TEST_YEARS):
        raise ValueError("Residual checkpoint does not declare tests for 2023 and 2024.")

def load_frozen_jump_model(
    checkpoint_path: Path,
    *,
    full_df: pd.DataFrame,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TemporalResidualJumpHeadsModel, dict[str, object]]:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Residual checkpoint must contain a dictionary payload.")
    validate_residual_checkpoint(checkpoint, fold=fold)

    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = vars(checkpoint_args)

    n_species = int(full_df["species_idx"].max()) + 1
    n_families = int(full_df["family_idx"].max()) + 1
    entity_emb_dim = int(
        _checkpoint_value(checkpoint_args, "entity_emb_dim", 16)
    )
    edge_hidden_dim = int(
        _checkpoint_value(checkpoint_args, "edge_hidden_dim", 64)
    )
    latent_dim = int(_checkpoint_value(checkpoint_args, "latent_dim", 12))
    decoder_hidden_dim = int(
        _checkpoint_value(checkpoint_args, "decoder_hidden_dim", 64)
    )
    dropout = float(_checkpoint_value(checkpoint_args, "dropout", 0.10))

    residual_model = SnapshotEncoderResidualModel(
        n_species=n_species,
        n_families=n_families,
        entity_emb_dim=entity_emb_dim,
        edge_hidden_dim=edge_hidden_dim,
        latent_dim=latent_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        dropout=dropout,
    )
    residual_model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    residual_model.to(device)
    residual_model.eval()

    jump_model = TemporalResidualJumpHeadsModel(
        residual_model=residual_model,
        n_species=n_species,
        n_families=n_families,
        latent_dim=latent_dim,
        head_hidden_dim=args.head_hidden_dim,
        head_bottleneck_dim=args.head_bottleneck_dim,
        dropout=args.head_dropout,
        species_embedding_name=args.species_embedding_name,
        family_embedding_name=args.family_embedding_name,
        freeze_backbone=True,
    ).to(device)
    return jump_model, checkpoint


def fit_heads_for_fixed_fold(
    *,
    model: TemporalResidualJumpHeadsModel,
    loaders,
    datasets,
    fold: int,
    train_max_target_year: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], int, pd.DataFrame, dict[str, object]]:
    down_pos_weight, up_pos_weight, class_counts = estimate_jump_pos_weights(
        datasets["train"]
    )
    optimizer = torch.optim.AdamW(
        list(model.head_parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state: dict[str, dict[str, torch.Tensor]] | None = None
    best_epoch: int | None = None
    best_score = -np.inf
    stale = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch_jump_heads(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            gradient_clip=args.gradient_clip,
            down_pos_weight=down_pos_weight,
            up_pos_weight=up_pos_weight,
            down_loss_weight=args.down_loss_weight,
            up_loss_weight=args.up_loss_weight,
        )
        val_metrics, _ = evaluate_jump_heads(
            model=model,
            loader=loaders["val"],
            device=device,
            down_pos_weight=down_pos_weight,
            up_pos_weight=up_pos_weight,
            down_loss_weight=args.down_loss_weight,
            up_loss_weight=args.up_loss_weight,
            return_predictions=False,
        )
        score = selection_score(val_metrics, args.checkpoint_metric)
        history.append(
            {
                "fold": int(fold),
                "epoch": int(epoch),
                "protocol": "fixed_2023_2024_vault",
                "head_training_max_target_year": int(train_max_target_year),
                "head_validation_target_year": int(VALIDATION_TARGET_YEAR),
                "selection_metric": args.checkpoint_metric,
                "selection_score": float(score),
                "train_down_bce": train_metrics["down_bce"],
                "train_up_bce": train_metrics["up_bce"],
                "train_combined_bce": train_metrics["combined_bce"],
                "val_down_bce": val_metrics["down_bce"],
                "val_up_bce": val_metrics["up_bce"],
                "val_combined_bce": val_metrics["combined_bce"],
                "val_down_auroc": val_metrics["down_auroc"],
                "val_down_average_precision": val_metrics[
                    "down_average_precision"
                ],
                "val_up_auroc": val_metrics["up_auroc"],
                "val_up_average_precision": val_metrics[
                    "up_average_precision"
                ],
            }
        )

        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = {
                "down_head": {
                    key: value.detach().cpu().clone()
                    for key, value in model.down_head.state_dict().items()
                },
                "up_head": {
                    key: value.detach().cpu().clone()
                    for key, value in model.up_head.state_dict().items()
                },
            }
            stale = 0
        else:
            stale += 1

        if stale >= args.patience:
            break

    if best_state is None or best_epoch is None:
        raise RuntimeError("No direction head checkpoint was selected.")

    model.down_head.load_state_dict(best_state["down_head"])
    model.up_head.load_state_dict(best_state["up_head"])

    training_info = {
        "down_pos_weight": float(down_pos_weight),
        "up_pos_weight": float(up_pos_weight),
        **class_counts,
        "best_selection_score": float(best_score),
        "class_weights_source": (
            "training_country_transitions_with_target_year_before_2022"
        ),
        "head_training_max_target_year": int(train_max_target_year),
        "head_validation_target_year": int(VALIDATION_TARGET_YEAR),
    }
    return best_state, best_epoch, pd.DataFrame(history), training_info


def add_observed_prediction_metadata(
    predictions: pd.DataFrame,
    *,
    cells: pd.DataFrame,
    fold: int,
    checkpoint_path: Path,
    best_epoch: int,
    train_max_target_year: int,
    model: TemporalResidualJumpHeadsModel,
    args: argparse.Namespace,
) -> pd.DataFrame:
    target_columns = [
        "cell_row_id",
        "Species",
        "Family",
        "species_idx",
        "family_idx",
        "p_baseline",
        "baseline_source",
    ]
    for optional in [
        "baseline_history_min_year",
        "baseline_history_max_year",
        "baseline_n_history_years",
        "baseline_n_history_cells",
        "baseline_n_history_tests",
        "baseline_training_cutoff_exclusive",
    ]:
        if optional in cells.columns:
            target_columns.append(optional)

    target_meta = cells[target_columns].drop_duplicates("cell_row_id")
    out = predictions.merge(
        target_meta,
        on="cell_row_id",
        how="left",
        validate="one_to_one",
    )

    out["fold"] = int(fold)
    out["colleague_fold"] = int(fold - 1)
    out["held_out_country"] = out["Country"].astype(str)
    out["model_name"] = MODEL_NAME
    out["evaluation_protocol"] = EVALUATION_PROTOCOL
    out["evaluation_set"] = [
        evaluation_set_from_target_year(year, True)
        for year in out["target_year"]
    ]
    out["dataset_role"] = out["evaluation_set"]
    out["target_observed"] = True
    out["training_data_max_year"] = int(TRAIN_MAX_YEAR)
    out["head_training_max_target_year"] = int(train_max_target_year)
    out["head_validation_target_year"] = int(VALIDATION_TARGET_YEAR)
    out["country_year_test_years"] = ",".join(map(str, TEST_YEARS))
    out["residual_checkpoint_path"] = str(checkpoint_path)
    out["jump_head_best_epoch"] = int(best_epoch)
    out["backbone_frozen"] = True
    out["continuous_forecast_modified"] = False
    out["country_seen_in_parameter_fitting"] = False
    out["species_embedding_module"] = model.species_embedding_name
    out["family_embedding_module"] = model.family_embedding_name
    out["jump_threshold"] = float(args.jump_threshold)
    out["jump_min_tests"] = float(args.jump_min_tests)
    out["jump_min_tests_rule"] = args.jump_min_tests_rule
    out["jump_label_policy"] = args.jump_label_policy
    out["target_outcome_used_as_model_input"] = False
    out["target_outcome_used_for_training"] = False
    out["target_outcome_used_for_checkpoint_selection"] = False
    out["class_weights_use_external_outcomes"] = False

    baseline_ok = pd.Series(True, index=out.index)
    if "baseline_history_max_year" in out.columns:
        baseline_ok = (
            out["baseline_history_max_year"].isna()
            | out["baseline_history_max_year"].le(TRAIN_MAX_YEAR)
        )
    valid_observed_year = (
        out["target_year"].le(TRAIN_MAX_YEAR)
        | out["target_year"].isin(TEST_YEARS)
    )
    out["temporal_leakage_check_passed"] = (
        valid_observed_year
        & out["input_year"].eq(out["target_year"] - 1)
        & baseline_ok
        & out["head_training_max_target_year"].lt(VALIDATION_TARGET_YEAR)
        & out["head_validation_target_year"].eq(VALIDATION_TARGET_YEAR)
    )
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

def summarize_predictions(
    predictions: pd.DataFrame,
    group_columns: list[str],
    evaluation_set: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(
        group_columns, dropna=False, sort=True
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        metrics = classification_metrics_from_frame(group)
        row.update(metrics)
        auroc_values = np.asarray(
            [metrics.get("down_auroc", np.nan), metrics.get("up_auroc", np.nan)],
            dtype=float,
        )
        ap_values = np.asarray(
            [
                metrics.get("down_average_precision", np.nan),
                metrics.get("up_average_precision", np.nan),
            ],
            dtype=float,
        )
        row["mean_auroc"] = (
            float(np.nanmean(auroc_values)) if np.isfinite(auroc_values).any() else np.nan
        )
        row["mean_average_precision"] = (
            float(np.nanmean(ap_values)) if np.isfinite(ap_values).any() else np.nan
        )
        row["model_name"] = MODEL_NAME
        row["evaluation_set"] = evaluation_set
        row["n_countries"] = int(group["Country"].nunique())
        row["n_rows"] = int(len(group))
        row["evaluation_protocol"] = EVALUATION_PROTOCOL
        rows.append(row)
    return pd.DataFrame(rows)

def tensor(
    values: Iterable,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    series = pd.Series(list(values))
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        examples = series.loc[numeric.isna()].head(10).tolist()
        raise ValueError(
            f"Cannot convert values to a tensor. Invalid examples: {examples}"
        )
    if dtype == torch.long:
        array = numeric.to_numpy(dtype=np.int64)
    else:
        array = numeric.to_numpy(dtype=np.float32)
    return torch.as_tensor(array, dtype=dtype, device=device)


@torch.no_grad()
def predict_future_country(
    *,
    model: TemporalResidualJumpHeadsModel,
    input_cells: pd.DataFrame,
    target_cells: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    if input_cells.empty or target_cells.empty:
        return pd.DataFrame()

    model.eval()
    n_input = len(input_cells)
    n_target = len(target_cells)
    final_logits, delta_logits, _, down_logit, up_logit = model(
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
            target_cells["species_idx"], torch.long, device
        ),
        target_family_idx=tensor(
            target_cells["family_idx"], torch.long, device
        ),
        target_snapshot_batch_idx=torch.zeros(
            n_target, dtype=torch.long, device=device
        ),
        target_baseline_logit=tensor(
            target_cells["baseline_logit"], torch.float32, device
        ),
        target_current_prop_S=tensor(
            target_cells["p_current"], torch.float32, device
        ),
    )

    out = target_cells.copy()
    out["p_pred_frozen"] = torch.sigmoid(final_logits).cpu().numpy()
    out["delta_logit_frozen"] = delta_logits.cpu().numpy()
    out["down_logit"] = down_logit.cpu().numpy()
    out["down_prob"] = torch.sigmoid(down_logit).cpu().numpy()
    out["up_logit"] = up_logit.cpu().numpy()
    out["up_prob"] = torch.sigmoid(up_logit).cpu().numpy()
    return out


def future_candidates(
    *,
    input_cells: pd.DataFrame,
    training_df: pd.DataFrame,
    forecast_year: int,
    alpha: float,
    beta: float,
) -> pd.DataFrame:
    candidates = input_cells[
        [
            "Country",
            "Species",
            "Family",
            "species_idx",
            "family_idx",
            "prop_S",
            "n_total",
        ]
    ].copy()
    candidates = candidates.rename(
        columns={"prop_S": "p_current", "n_total": "n_current"}
    )
    candidates["Year"] = int(forecast_year)

    baseline = HELPER.compute_train_only_expanding_baseline(
        training_country_df=training_df,
        target_universe_df=candidates,
        alpha=alpha,
        beta=beta,
    )
    baseline_columns = [
        "Species",
        "Family",
        "Year",
        "p_baseline",
        "baseline_logit",
        "baseline_source",
    ]
    for optional in [
        "baseline_history_min_year",
        "baseline_history_max_year",
        "baseline_n_history_years",
        "baseline_n_history_cells",
        "baseline_n_history_tests",
        "baseline_training_cutoff_exclusive",
    ]:
        if optional in baseline.columns:
            baseline_columns.append(optional)

    out = candidates.merge(
        baseline[baseline_columns],
        on=["Species", "Family", "Year"],
        how="left",
        validate="many_to_one",
    )
    if out["p_baseline"].isna().any():
        raise ValueError("Some future candidates lack a historical baseline.")
    if (
        "baseline_history_max_year" in out.columns
        and out["baseline_history_max_year"].notna().any()
        and out["baseline_history_max_year"].dropna().gt(TRAIN_MAX_YEAR).any()
    ):
        raise AssertionError(
            "A prospective baseline used training history after 2022."
        )
    return out


def add_future_metadata(
    predictions: pd.DataFrame,
    *,
    fold: int,
    checkpoint_path: Path,
    best_epoch: int,
    train_max_target_year: int,
    model: TemporalResidualJumpHeadsModel,
    args: argparse.Namespace,
) -> pd.DataFrame:
    out = predictions.copy()
    out["input_year"] = int(args.forecast_input_year)
    out["target_year"] = int(args.forecast_year)
    out["fold"] = int(fold)
    out["colleague_fold"] = int(fold - 1)
    out["held_out_country"] = out["Country"].astype(str)
    out["model_name"] = MODEL_NAME
    out["evaluation_protocol"] = EVALUATION_PROTOCOL
    out["dataset_role"] = "prospective_forecast"
    out["target_observed"] = False
    out["training_data_max_year"] = int(TRAIN_MAX_YEAR)
    out["head_training_max_target_year"] = int(train_max_target_year)
    out["head_validation_target_year"] = int(VALIDATION_TARGET_YEAR)
    out["country_year_test_years"] = ",".join(map(str, TEST_YEARS))
    out["residual_checkpoint_path"] = str(checkpoint_path)
    out["jump_head_best_epoch"] = int(best_epoch)
    out["backbone_frozen"] = True
    out["continuous_forecast_modified"] = False
    out["country_seen_in_parameter_fitting"] = False
    out["species_embedding_module"] = model.species_embedding_name
    out["family_embedding_module"] = model.family_embedding_name
    out["jump_threshold"] = float(args.jump_threshold)
    out["jump_min_tests"] = float(args.jump_min_tests)
    out["jump_min_tests_rule"] = args.jump_min_tests_rule
    out["jump_label_policy"] = args.jump_label_policy
    out["target_outcome_used_as_model_input"] = False
    out["target_outcome_used_for_training"] = False
    out["target_outcome_used_for_checkpoint_selection"] = False
    out["class_weights_use_vault_outcomes"] = False
    out["temporal_leakage_check_passed"] = (
        out["input_year"].eq(args.forecast_input_year)
        & out["target_year"].eq(args.forecast_year)
        & out["head_training_max_target_year"].lt(
            VALIDATION_TARGET_YEAR
        )
        & out["head_validation_target_year"].eq(
            VALIDATION_TARGET_YEAR
        )
    )
    return out


def run_fold(
    *,
    full_df: pd.DataFrame,
    folds: pd.DataFrame,
    fold: int,
    all_countries: set[str],
    all_pairs: list[tuple[str, int, int]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
]:
    test_countries = set(
        folds.loc[folds["fold"].eq(fold), "Country"].astype(str)
    )
    training_countries = all_countries - test_countries
    if training_countries & test_countries:
        raise AssertionError("Training and external countries overlap.")

    fold_dir = args.output_dir / f"fold_{fold}"
    model_dir = fold_dir / "fixed_historical_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed + int(fold))

    cells, training_df = build_fold_cells(
        full_df=full_df,
        training_countries=training_countries,
        alpha=args.alpha,
        beta=args.beta,
    )
    (
        train_pairs,
        val_pairs,
        historical_test_pairs,
        country_year_test_pairs,
    ) = split_pairs_for_dual_evaluation(
        all_pairs,
        training_countries=training_countries,
        test_countries=test_countries,
    )
    train_max_target_year = max(pair[2] for pair in train_pairs)

    pair_table = build_pair_table(
        train_pairs,
        val_pairs,
        historical_test_pairs,
        country_year_test_pairs,
        fold=fold,
    )
    pair_table.to_csv(
        model_dir / "jump_pairs_dual_evaluation.csv",
        index=False,
    )

    model_cells, model_pairs = _coerce_inputs(cells, pair_table)
    loaders, datasets = build_temporal_jump_dataloaders(
        cells=model_cells,
        pairs=model_pairs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed + 10_000 * int(fold),
        jump_threshold=args.jump_threshold,
        jump_min_tests=args.jump_min_tests,
        jump_min_tests_rule=args.jump_min_tests_rule,
        jump_label_policy=args.jump_label_policy,
    )

    residual_checkpoint = checkpoint_path_for_fixed_fold(
        args.temporal_model_dir,
        fold=int(fold),
    )
    model, residual_payload = load_frozen_jump_model(
        residual_checkpoint,
        full_df=full_df,
        fold=int(fold),
        args=args,
        device=device,
    )

    best_state, best_epoch, history, training_info = fit_heads_for_fixed_fold(
        model=model,
        loaders=loaders,
        datasets=datasets,
        fold=int(fold),
        train_max_target_year=int(train_max_target_year),
        args=args,
        device=device,
    )

    _, test_predictions = evaluate_jump_heads(
        model=model,
        loader=loaders["test"],
        device=device,
        down_pos_weight=float(training_info["down_pos_weight"]),
        up_pos_weight=float(training_info["up_pos_weight"]),
        down_loss_weight=args.down_loss_weight,
        up_loss_weight=args.up_loss_weight,
        return_predictions=True,
    )
    if test_predictions is None or test_predictions.empty:
        raise RuntimeError("No external direction predictions were produced.")

    observed = add_observed_prediction_metadata(
        test_predictions,
        cells=model_cells,
        fold=int(fold),
        checkpoint_path=residual_checkpoint,
        best_epoch=int(best_epoch),
        train_max_target_year=int(train_max_target_year),
        model=model,
        args=args,
    )
    if not set(observed["Country"].astype(str)).issubset(test_countries):
        raise AssertionError("Jump predictions include nonexternal countries.")
    observed_sets = set(observed["evaluation_set"].astype(str).unique())
    expected_sets = {EVALUATION_SET_COUNTRY, EVALUATION_SET_COUNTRY_YEAR}
    if observed_sets != expected_sets:
        raise AssertionError(
            f"Fold {fold} produced evaluation sets {observed_sets}, expected {expected_sets}."
        )
    if not observed["temporal_leakage_check_passed"].fillna(False).all():
        raise AssertionError("Temporal provenance failed for jump predictions.")

    observed.to_csv(
        model_dir / "jump_predictions_dual_evaluation.csv",
        index=False,
    )
    observed.loc[
        observed["evaluation_set"].eq(EVALUATION_SET_COUNTRY)
    ].to_csv(
        model_dir / "jump_predictions_country_generalization_through_2022.csv",
        index=False,
    )
    observed.loc[
        observed["evaluation_set"].eq(EVALUATION_SET_COUNTRY_YEAR)
    ].to_csv(
        model_dir / "jump_predictions_2023_2024.csv",
        index=False,
    )
    history.to_csv(model_dir / "jump_training_history.csv", index=False)

    if args.save_models:
        torch.save(
            {
                "down_head_state_dict": best_state["down_head"],
                "up_head_state_dict": best_state["up_head"],
                "fold": int(fold),
                "training_data_max_year": int(TRAIN_MAX_YEAR),
                "head_gradient_training_max_target_year": int(train_max_target_year),
                "head_checkpoint_selection_target_year": int(VALIDATION_TARGET_YEAR),
                "historical_country_test_target_years": sorted(
                    {pair[2] for pair in historical_test_pairs}
                ),
                "country_year_test_target_years": list(TEST_YEARS),
                "best_epoch": int(best_epoch),
                "residual_checkpoint_path": str(residual_checkpoint),
                "residual_checkpoint_best_epoch": residual_payload.get("best_epoch"),
                "species_embedding_name": model.species_embedding_name,
                "family_embedding_name": model.family_embedding_name,
                "backbone_frozen": True,
                "continuous_forecast_modified": False,
                "class_weights_use_external_outcomes": False,
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "arguments": vars(args),
            },
            model_dir / "temporal_residual_jump_heads.pt",
        )

    future = pd.DataFrame()
    if args.run_future:
        country_parts: list[pd.DataFrame] = []
        for country in sorted(test_countries):
            input_cells = model_cells.loc[
                model_cells["Country"].eq(country)
                & model_cells["Year"].eq(args.forecast_input_year)
            ].copy()
            if input_cells.empty:
                continue
            candidates = future_candidates(
                input_cells=input_cells,
                training_df=training_df,
                forecast_year=int(args.forecast_year),
                alpha=args.alpha,
                beta=args.beta,
            )
            prediction = predict_future_country(
                model=model,
                input_cells=input_cells,
                target_cells=candidates,
                device=device,
            )
            if not prediction.empty:
                country_parts.append(prediction)
        if country_parts:
            future = add_future_metadata(
                pd.concat(country_parts, ignore_index=True, sort=False),
                fold=int(fold),
                checkpoint_path=residual_checkpoint,
                best_epoch=int(best_epoch),
                train_max_target_year=int(train_max_target_year),
                model=model,
                args=args,
            )
            future["evaluation_set"] = EVALUATION_SET_FUTURE
            future["dataset_role"] = EVALUATION_SET_FUTURE
            if not future["temporal_leakage_check_passed"].fillna(False).all():
                raise AssertionError("Temporal provenance failed for future scores.")
            future.to_csv(model_dir / "future_jump_predictions.csv", index=False)

    training_info = {
        "fold": int(fold),
        "colleague_fold": int(fold - 1),
        "n_training_countries": int(len(training_countries)),
        "n_external_countries": int(len(test_countries)),
        "residual_checkpoint_path": str(residual_checkpoint),
        "jump_head_best_epoch": int(best_epoch),
        "species_embedding_name": model.species_embedding_name,
        "family_embedding_name": model.family_embedding_name,
        **training_info,
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return observed, history, training_info, future


def summarize_mean_std_across_folds(
    fold_metrics: pd.DataFrame,
    evaluation_set: str,
) -> pd.DataFrame:
    metric_columns = [
        "down_auroc",
        "down_average_precision",
        "up_auroc",
        "up_average_precision",
        "mean_auroc",
        "mean_average_precision",
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
            values = pd.to_numeric(group[column], errors="coerce")
            finite = values[np.isfinite(values)]
            row[f"{column}_n_valid_folds"] = int(len(finite))
            if len(finite) != EXPECTED_N_FOLDS:
                row[f"{column}_mean"] = np.nan
                row[f"{column}_std"] = np.nan
                row[f"{column}_mean_plus_minus_std"] = ""
                continue
            mean_value = float(finite.mean())
            std_value = float(finite.std(ddof=1))
            row[f"{column}_mean"] = mean_value
            row[f"{column}_std"] = std_value
            row[f"{column}_mean_plus_minus_std"] = (
                f"{mean_value:.6f} ± {std_value:.6f}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_jump_evaluation_outputs(
    predictions: pd.DataFrame,
    evaluation_set: str,
    prefix: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = predictions.loc[
        predictions["evaluation_set"].eq(evaluation_set)
    ].copy()
    if subset.empty:
        raise RuntimeError(f"No jump predictions for {evaluation_set}.")
    fold_year = summarize_predictions(
        subset,
        ["fold", "target_year"],
        evaluation_set,
    )
    fold_metrics = summarize_predictions(
        subset,
        ["fold"],
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
    available_years = set(
        pd.to_numeric(full_df["Year"], errors="raise").astype(int).unique().tolist()
    )
    missing_vault_years = sorted(set(TEST_YEARS) - available_years)
    if missing_vault_years:
        raise ValueError(f"Input data are missing {missing_vault_years}.")
    if VALIDATION_TARGET_YEAR not in available_years:
        raise ValueError("Input data are missing target year 2022.")

    folds = build_country_folds(
        full_df,
        folds_path=args.country_folds_path,
    )
    folds["training_data_max_year"] = int(TRAIN_MAX_YEAR)
    folds["head_validation_target_year"] = int(VALIDATION_TARGET_YEAR)
    folds["country_year_test_years"] = ",".join(map(str, TEST_YEARS))
    folds.to_csv(args.output_dir / "country_fold_assignments.csv", index=False)

    all_countries = set(full_df["Country"].astype(str))
    all_pairs = HELPER.consecutive_pairs(full_df)
    prediction_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []
    training_info_rows: list[dict[str, object]] = []
    future_parts: list[pd.DataFrame] = []
    failure_rows: list[dict[str, object]] = []

    fold_ids = sorted(folds["fold"].astype(int).unique().tolist())
    for fold in fold_ids:
        try:
            observed, history, training_info, future = run_fold(
                full_df=full_df,
                folds=folds,
                fold=int(fold),
                all_countries=all_countries,
                all_pairs=all_pairs,
                args=args,
                device=device,
            )
            prediction_parts.append(observed)
            history_parts.append(history)
            training_info_rows.append(training_info)
            if not future.empty:
                future_parts.append(future)
        except Exception as error:
            failure_rows.append(
                {
                    "fold": int(fold),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"Fold {fold}, jump model failed: {error}")
            if not args.continue_on_error:
                raise

    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(
            args.output_dir / "jump_failures.csv",
            index=False,
        )
    if len(prediction_parts) != EXPECTED_N_FOLDS:
        raise RuntimeError(
            f"Only {len(prediction_parts)} of five folds completed. "
            "Five fold mean and standard deviation cannot be reported."
        )

    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False)
    if not predictions["temporal_leakage_check_passed"].fillna(False).all():
        raise AssertionError("Combined jump predictions failed provenance checks.")

    historical = predictions.loc[
        predictions["evaluation_set"].eq(EVALUATION_SET_COUNTRY)
    ].copy()
    country_year = predictions.loc[
        predictions["evaluation_set"].eq(EVALUATION_SET_COUNTRY_YEAR)
    ].copy()
    if historical.empty or country_year.empty:
        raise RuntimeError("One of the two jump evaluation tasks is empty.")

    predictions.to_csv(
        args.output_dir / "temporal_residual_jump_predictions_all_external_tests.csv",
        index=False,
    )
    historical.to_csv(
        args.output_dir / "temporal_residual_jump_predictions_country_generalization_through_2022.csv",
        index=False,
    )
    country_year.to_csv(
        args.output_dir / "temporal_residual_jump_predictions_country_year_generalization_2023_2024.csv",
        index=False,
    )
    country_year.to_csv(
        args.output_dir / "temporal_residual_jump_predictions_2023_2024.csv",
        index=False,
    )

    write_jump_evaluation_outputs(
        predictions,
        EVALUATION_SET_COUNTRY,
        "jump_country_generalization",
        args.output_dir,
    )
    write_jump_evaluation_outputs(
        predictions,
        EVALUATION_SET_COUNTRY_YEAR,
        "jump_country_year_generalization",
        args.output_dir,
    )

    if history_parts:
        pd.concat(history_parts, ignore_index=True, sort=False).to_csv(
            args.output_dir / "jump_training_history.csv",
            index=False,
        )
    if training_info_rows:
        pd.DataFrame(training_info_rows).to_csv(
            args.output_dir / "jump_training_info.csv",
            index=False,
        )
    if future_parts:
        pd.concat(future_parts, ignore_index=True, sort=False).to_csv(
            args.output_dir / "temporal_residual_future_jump_predictions.csv",
            index=False,
        )

    metadata = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "country_folds_path": str(args.country_folds_path),
        "number_of_folds": EXPECTED_N_FOLDS,
        "training_data_max_year": TRAIN_MAX_YEAR,
        "head_gradient_training_target_year_rule": "target_year < 2022",
        "head_checkpoint_selection_target_year": VALIDATION_TARGET_YEAR,
        "historical_country_test_rule": "external countries and target year <= 2022",
        "country_year_test_years": list(TEST_YEARS),
        "one_head_model_per_country_fold": True,
        "same_heads_used_for_both_tests": True,
        "backbone_frozen": True,
        "class_weights_use_external_outcomes": False,
        "uncertainty_definition": "mean and sample standard deviation across five fold metrics",
        "direction_outputs": ["down_prob", "up_prob"],
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "jump_run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf8",
    )

    print("Completed both jump evaluation tasks")
    print("Historical country generalization rows:", len(historical))
    print("Country and year generalization rows:", len(country_year))
    print("Completed folds:", fold_ids)
    print("Backbone weights changed: False")

if __name__ == "__main__":
    main()
