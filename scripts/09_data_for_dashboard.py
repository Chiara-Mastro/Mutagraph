#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
08_data_for_dashboard.py

Build the three data products consumed by the AMR dashboard from the current
modelling pipeline.

Outputs
=======

1. Susceptibility landscape

   <output_dir>/landscape_prediction/landscape_predictions.csv

   Observed rows use external country cellwise leave one out predictions from
   script 02. Genuine to_impute rows are predicted with the matching frozen
   external fold checkpoint and the complete observed Country Year snapshot as
   context. Intrinsic resistance rows are retained without a model prediction.

2. Sampling return and biological heterogeneity panel

   <output_dir>/sampling_returns/sampling_returns_cells.csv

   This table is derived directly from the residual encoder leave one out
   predictions exported by script 02. The fold training beta binomial
   concentration is used as the effective heterogeneity scale. The table
   reports the irreducible latent rate floor, the finite sample predictive
   standard deviation, and the ratio n_total / phi used to describe whether
   additional isolates are still expected to help materially.

3. Abrupt change watchlist

   <output_dir>/temporal_prediction/temporal_jump_candidate_rankings.csv

   The watchlist is built directly from the down and up probabilities exported
   by script 05. Observed external tests and the prospective forecast are
   combined, converted to one row per direction, and ranked inside each
   Country, input year, model, and direction.

Required inputs
===============

input_path
    Standard table produced by script 01.

completion_output_dir
    Output directory produced by script 02.

jump_output_dir
    Output directory produced by script 05. A previously prepared long ranking
    file can be supplied instead through temporal_rankings_path.

When the GNN notebook reconstruction_export directory is supplied, its
observed reconstruction drives the heterogeneity page, its imputed
reconstruction is added to the completion page, and its directional
probabilities are merged with the temporal residual jump rankings.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as beta_distribution


PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT_FROM_SCRIPT))

from src.CONFIG import EPS
from src.utils import choose_device


KEY_COLUMNS = ["Country", "Year", "Species", "Family"]
TRAIN_MAX_YEAR = 2022
MODEL_PRIOR = "species_family_prior"
MODEL_RESIDUAL = "snapshot_encoder_residual_model"

STATUS_OBSERVED = "observed"
STATUS_TO_IMPUTE = "to_impute"
STATUS_INTRINSIC = "intrinsic_resistance"

OBSERVED_CONTEXT = "observed_cell_leave_one_out"
IMPUTED_CONTEXT = "full_observed_snapshot_to_impute"
INTRINSIC_CONTEXT = "excluded_intrinsic_resistance"

CHECKPOINT_CANDIDATE_NAMES = [
    "residual_encoder_model.pt",
    "snapshot_encoder_residual_model.pt",
    "snapshot_encoder_residual_model_encoder_model.pt",
]


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create dashboard data from the standardized table, completion "
            "outputs, and temporal jump outputs."
        )
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Standardized table produced by 01_create_standard_dataset.py.",
    )
    parser.add_argument(
        "--completion_output_dir",
        type=Path,
        required=True,
        help="Output directory produced by script 02.",
    )
    parser.add_argument(
        "--jump_output_dir",
        type=Path,
        default=None,
        help=(
            "Output directory produced by script 05. Required unless "
            "--temporal_rankings_path is supplied."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Dashboard root in which the data folders are created.",
    )
    parser.add_argument(
        "--completion_script_path",
        type=Path,
        default=Path(__file__).with_name(
            "02_country_generalization_completion_encoders.py"
        ),
        help="Path to script 02, imported to reuse the exact model and features.",
    )
    parser.add_argument(
        "--temporal_rankings_path",
        type=Path,
        default=None,
        help=(
            "Optional already prepared long temporal ranking. This is retained "
            "for compatibility with older runs."
        ),
    )
    parser.add_argument(
        "--gnn_export_dir",
        "--graph_export_dir",
        dest="gnn_export_dir",
        type=Path,
        default=None,
        help=(
            "Optional reconstruction_export directory produced by the GNN "
            "notebook. Recognised files are copied into "
            "<output_dir>/graph_model and the GNN jump probabilities are "
            "merged into the dashboard temporal ranking."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device accepted by src.utils.choose_device.",
    )
    parser.add_argument(
        "--target_batch_size",
        type=int,
        default=512,
        help="Maximum number of to_impute targets predicted in one model call.",
    )
    parser.add_argument(
        "--require_2025_temporal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require at least one prospective target year 2025 row.",
    )
    parser.add_argument(
        "--required_temporal_years",
        nargs="+",
        type=int,
        default=[2023, 2024, 2025],
        help="Target years required in the temporal watchlist.",
    )
    parser.add_argument(
        "--jump_probability_threshold",
        type=float,
        default=0.50,
        help=(
            "Probability threshold used only to mark dashboard alerts. "
            "Ranking always uses the full probability score."
        ),
    )
    args = parser.parse_args()
    if args.jump_output_dir is None and args.temporal_rankings_path is None:
        parser.error(
            "Supply --jump_output_dir from script 05 or "
            "--temporal_rankings_path from an older prepared run."
        )
    if not 0.0 < args.jump_probability_threshold < 1.0:
        parser.error("--jump_probability_threshold must lie between zero and one.")
    return args

# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def normalize_status(value: object):
    text = str(value or "").strip().lower().replace(" ", "_")
    aliases = {
        "observed": STATUS_OBSERVED,
        "to_impute": STATUS_TO_IMPUTE,
        "impute": STATUS_TO_IMPUTE,
        "missing": STATUS_TO_IMPUTE,
        "intrinsic_resistance": STATUS_INTRINSIC,
        "do_not_impute_intrinsic": STATUS_INTRINSIC,
        "intrinsic": STATUS_INTRINSIC,
    }
    return aliases.get(text, text)


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
):
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def safe_numeric(series: pd.Series):
    return pd.to_numeric(series, errors="coerce")


def first_existing_path(
    directory: Path,
    candidate_names: Iterable[str],
    label: str,
    *,
    required: bool = True,
):
    candidates = [directory / name for name in candidate_names]
    for path in candidates:
        if path.exists():
            return path
    if required:
        rendered = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Could not find {label}. Tried: {rendered}"
        )
    return None


def tensor(
    values: Iterable[object],
    dtype: torch.dtype,
    device: torch.device,
):
    return torch.as_tensor(list(values), dtype=dtype, device=device)


def load_script_02(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            "Could not find script 02 at: " + str(path)
        )

    spec = importlib.util.spec_from_file_location(
        "country_generalization_completion_helpers",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not build an import specification for script 02.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def torch_load_checkpoint(path: Path, device: torch.device):
    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} is not a dictionary.")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {path} has no model_state_dict.")
    return checkpoint


def find_fold_checkpoint(fold_dir: Path):
    for name in CHECKPOINT_CANDIDATE_NAMES:
        path = fold_dir / name
        if path.exists():
            return path

    matches = sorted(
        path
        for path in fold_dir.glob("*.pt")
        if "residual" in path.name.lower()
        and "encoder" in path.name.lower()
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple residual encoder checkpoints found in {fold_dir}: "
            + ", ".join(path.name for path in matches)
        )

    raise FileNotFoundError(
        "No residual encoder checkpoint was found in "
        + str(fold_dir)
        + ". Expected the checkpoint written by the current script 02, "
        + "normally snapshot_encoder_residual_model_encoder_model.pt."
    )


def namespace_from_checkpoint(checkpoint: dict[str, object]):
    raw_args = checkpoint.get("args", {})
    if isinstance(raw_args, argparse.Namespace):
        raw_args = vars(raw_args)
    if not isinstance(raw_args, dict):
        raise TypeError("Checkpoint args must be a dictionary or Namespace.")

    defaults = {
        "entity_emb_dim": 16,
        "edge_hidden_dim": 64,
        "latent_dim": 12,
        "decoder_hidden_dim": 64,
        "dropout": 0.10,
        "alpha": 1.0,
        "beta": 1.0,
        "baseline_mode": "species_family",
        "seed": 42,
    }
    defaults.update(raw_args)
    return SimpleNamespace(**defaults)


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------


def expand_global_species_family_domain(frame: pd.DataFrame):
    """Show the same complete Species by Family universe in every snapshot.

    Existing rows keep their original status and measurements. A combination
    absent from one Country Year is generated as to_impute, unless that
    Species Family pair is marked as intrinsic resistance anywhere in the
    standardized domain.
    """
    country_years = frame[["Country", "Year"]].drop_duplicates().copy()
    species = sorted(frame["Species"].astype(str).unique().tolist())
    families = sorted(frame["Family"].astype(str).unique().tolist())

    pair_grid = pd.MultiIndex.from_product(
        [species, families],
        names=["Species", "Family"],
    ).to_frame(index=False)
    complete_grid = country_years.merge(pair_grid, how="cross")

    intrinsic_pairs = set(
        zip(
            frame.loc[frame["status"].eq(STATUS_INTRINSIC), "Species"].astype(str),
            frame.loc[frame["status"].eq(STATUS_INTRINSIC), "Family"].astype(str),
        )
    )

    expanded = complete_grid.merge(
        frame,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    generated = expanded["_merge"].eq("left_only")
    generated_pairs = list(zip(expanded["Species"], expanded["Family"]))
    generated_status = [
        STATUS_INTRINSIC if pair in intrinsic_pairs else STATUS_TO_IMPUTE
        for pair in generated_pairs
    ]
    expanded.loc[generated, "status"] = np.asarray(generated_status, dtype=object)[generated.to_numpy()]
    expanded["domain_cell_generated"] = generated
    expanded = expanded.drop(columns=["_merge"])
    return expanded


def load_full_domain(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    require_columns(
        frame,
        ["Country", "Year", "Species", "Family", "status"],
        path.name,
    )

    out = frame.copy()
    for column in ["Country", "Species", "Family"]:
        out[column] = out[column].astype("string").str.strip()
    out["Year"] = safe_numeric(out["Year"])
    out["status"] = out["status"].map(normalize_status)

    for column in ["n_S", "n_total", "prop_S"]:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = safe_numeric(out[column])

    observed_missing_prop = (
        out["status"].eq(STATUS_OBSERVED)
        & out["prop_S"].isna()
        & out["n_S"].notna()
        & out["n_total"].gt(0)
    )
    out.loc[observed_missing_prop, "prop_S"] = (
        out.loc[observed_missing_prop, "n_S"]
        / out.loc[observed_missing_prop, "n_total"]
    )

    out = out.dropna(subset=KEY_COLUMNS).copy()
    out["Year"] = out["Year"].astype(int)

    duplicated = out.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = out.loc[duplicated, KEY_COLUMNS + ["status"]].head(20)
        raise ValueError(
            "The standardized input has duplicate Country Year Species Family cells. "
            "Examples:\n" + examples.to_string(index=False)
        )

    accepted = {STATUS_OBSERVED, STATUS_TO_IMPUTE, STATUS_INTRINSIC}
    unknown = sorted(set(out["status"].dropna()) - accepted)
    if unknown:
        print("Warning: unrecognised status values will be retained:", unknown)

    out = expand_global_species_family_domain(out)
    out["dashboard_row_id"] = np.arange(len(out), dtype=np.int64)
    return out.sort_values(KEY_COLUMNS).reset_index(drop=True)


def load_fold_assignment(path: Path):
    assignment = pd.read_csv(path)
    require_columns(assignment, ["Country", "external_fold"], path.name)
    assignment = assignment[["Country", "external_fold"]].copy()
    assignment["Country"] = assignment["Country"].astype("string").str.strip()
    assignment["external_fold"] = safe_numeric(
        assignment["external_fold"]
    ).astype("Int64")
    assignment = assignment.dropna().copy()
    assignment["external_fold"] = assignment["external_fold"].astype(int)

    if assignment["Country"].duplicated().any():
        raise ValueError("Every country must have one external fold assignment.")
    return assignment


def load_phi_table(path: Path):
    frame = pd.read_csv(path)
    require_columns(
        frame,
        ["fold", "model_name", "phi_train"],
        path.name,
    )
    frame = frame.copy()
    frame["fold"] = safe_numeric(frame["fold"]).astype("Int64")
    frame["phi_train"] = safe_numeric(frame["phi_train"])
    frame["raw_phi_train"] = (
        safe_numeric(frame["raw_phi_train"])
        if "raw_phi_train" in frame.columns
        else np.nan
    )
    frame["rho_train"] = (
        safe_numeric(frame["rho_train"])
        if "rho_train" in frame.columns
        else 1.0 / (frame["phi_train"] + 1.0)
    )
    if "phi_source" not in frame.columns:
        frame["phi_source"] = "not_exported"
    return frame


def get_phi_info(
    phi_table: pd.DataFrame,
    fold: int,
    model_name: str,
):
    match = phi_table.loc[
        phi_table["fold"].eq(fold)
        & phi_table["model_name"].astype(str).eq(model_name)
    ]
    if len(match) != 1:
        raise ValueError(
            f"Expected one phi row for fold {fold}, model {model_name}; "
            f"found {len(match)}."
        )
    row = match.iloc[0]
    phi = float(row["phi_train"])
    if not np.isfinite(phi) or phi <= 0:
        raise ValueError(
            f"Invalid phi for fold {fold}, model {model_name}: {phi}"
        )
    return {
        "raw_phi_train": float(row["raw_phi_train"])
        if np.isfinite(row["raw_phi_train"])
        else np.nan,
        "phi_train": phi,
        "rho_train": float(row["rho_train"]),
        "phi_source": str(row["phi_source"]),
    }


# -----------------------------------------------------------------------------
# Leave one out observed export
# -----------------------------------------------------------------------------


def prepare_observed_leave_one_out(
    path: Path,
    full_domain: pd.DataFrame,
):
    predictions = pd.read_csv(path)
    require_columns(
        predictions,
        KEY_COLUMNS + ["model_name", "p_pred"],
        path.name,
    )

    predictions = predictions.copy()
    predictions["Year"] = safe_numeric(predictions["Year"]).astype("Int64")
    predictions["p_pred"] = safe_numeric(predictions["p_pred"])

    keep_optional = [
        "fold",
        "p_baseline",
        "baseline_source",
        "predictive_prop_q05",
        "predictive_prop_q95",
        "epistemic_p_q05",
        "epistemic_p_q95",
        "posterior_p_sd",
        "context_n_cells",
        "context_n_tests",
        "snapshot_n_observed_cells",
        "phi_train",
        "rho_train",
        "phi_source",
        "species_seen_in_train",
        "family_seen_in_train",
        "species_family_seen_in_train",
        "both_entities_seen_in_train",
    ]
    available = [column for column in keep_optional if column in predictions.columns]

    residual = predictions.loc[
        predictions["model_name"].astype(str).eq(MODEL_RESIDUAL),
        KEY_COLUMNS + ["p_pred"] + available,
    ].copy()
    prior = predictions.loc[
        predictions["model_name"].astype(str).eq(MODEL_PRIOR),
        KEY_COLUMNS
        + ["p_pred"]
        + [
            column
            for column in [
                "predictive_prop_q05",
                "predictive_prop_q95",
                "phi_train",
                "rho_train",
                "phi_source",
            ]
            if column in predictions.columns
        ],
    ].copy()

    if residual.empty:
        raise ValueError(
            "The leave one out prediction file has no residual encoder rows."
        )

    residual_rename = {
        "p_pred": "p_residual_encoder",
        "predictive_prop_q05": "p_residual_encoder_q05",
        "predictive_prop_q95": "p_residual_encoder_q95",
        "phi_train": "residual_phi_train",
        "rho_train": "residual_rho_train",
        "phi_source": "residual_phi_source",
    }
    prior_rename = {
        "p_pred": "p_species_family_prior",
        "predictive_prop_q05": "p_species_family_prior_q05",
        "predictive_prop_q95": "p_species_family_prior_q95",
        "phi_train": "prior_phi_train",
        "rho_train": "prior_rho_train",
        "phi_source": "prior_phi_source",
    }
    residual = residual.rename(columns=residual_rename)
    prior = prior.rename(columns=prior_rename)

    if residual.duplicated(KEY_COLUMNS).any():
        raise ValueError("Residual leave one out predictions have duplicate keys.")
    if not prior.empty and prior.duplicated(KEY_COLUMNS).any():
        raise ValueError("Prior leave one out predictions have duplicate keys.")

    observed = full_domain.loc[
        full_domain["status"].eq(STATUS_OBSERVED)
    ].copy()
    observed = observed.merge(
        residual,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if not prior.empty:
        observed = observed.merge(
            prior,
            on=KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )

    observed["prediction_context"] = OBSERVED_CONTEXT
    observed["uncertainty_interval_type"] = (
        "beta_binomial_count_predictive_interval_using_observed_n_total"
    )
    observed["completion_prediction_available"] = observed[
        "p_residual_encoder"
    ].notna()
    observed["completion_exclusion_reason"] = np.where(
        observed["completion_prediction_available"],
        "",
        "no_leave_one_out_prediction",
    )
    return observed


# -----------------------------------------------------------------------------
# Genuine to_impute prediction
# -----------------------------------------------------------------------------


def add_categorical_indices(
    full_domain: pd.DataFrame,
    observed_standardized: pd.DataFrame,
):
    species_map = (
        observed_standardized[["Species", "species_idx"]]
        .drop_duplicates()
        .set_index("Species")["species_idx"]
        .to_dict()
    )
    family_map = (
        observed_standardized[["Family", "family_idx"]]
        .drop_duplicates()
        .set_index("Family")["family_idx"]
        .to_dict()
    )

    out = full_domain.copy()
    out["species_idx"] = out["Species"].map(species_map)
    out["family_idx"] = out["Family"].map(family_map)
    return out


def add_latent_beta_interval(
    frame: pd.DataFrame,
    p_column: str,
    phi: float,
    prefix: str,
):
    out = frame.copy()
    p = np.clip(safe_numeric(out[p_column]).to_numpy(float), EPS, 1.0 - EPS)
    valid = np.isfinite(p) & np.isfinite(phi) & (phi > 0)

    low = np.full(len(out), np.nan, dtype=float)
    high = np.full(len(out), np.nan, dtype=float)
    if valid.any():
        alpha = np.clip(p[valid] * phi, EPS, None)
        beta = np.clip((1.0 - p[valid]) * phi, EPS, None)
        low[valid] = beta_distribution.ppf(0.05, alpha, beta)
        high[valid] = beta_distribution.ppf(0.95, alpha, beta)

    out[f"{prefix}_q05"] = low
    out[f"{prefix}_q95"] = high
    return out


@torch.no_grad()
def predict_snapshot_targets(
    model: torch.nn.Module,
    context: pd.DataFrame,
    targets: pd.DataFrame,
    device: torch.device,
    batch_size: int,
):
    if context.empty:
        raise ValueError("Cannot encode an empty observed snapshot.")
    if targets.empty:
        return pd.DataFrame()

    model.eval()
    n_context = len(context)
    output_parts: list[pd.DataFrame] = []

    input_species = tensor(context["species_idx"], torch.long, device)
    input_family = tensor(context["family_idx"], torch.long, device)
    input_baseline = tensor(context["p_baseline"], torch.float32, device)
    input_residual = tensor(
        context["residual_prop_S"], torch.float32, device
    )
    input_tests = tensor(context["n_total"], torch.float32, device)
    input_batch_index = torch.zeros(
        n_context,
        dtype=torch.long,
        device=device,
    )

    for start in range(0, len(targets), batch_size):
        stop = min(start + batch_size, len(targets))
        chunk = targets.iloc[start:stop].copy()
        n_target = len(chunk)

        final_logits, delta_logits, _ = model(
            input_species_idx=input_species,
            input_family_idx=input_family,
            input_p_baseline=input_baseline,
            input_residual_prop_S=input_residual,
            input_n_total=input_tests,
            input_snapshot_batch_idx=input_batch_index,
            n_snapshots_in_batch=1,
            target_species_idx=tensor(
                chunk["species_idx"], torch.long, device
            ),
            target_family_idx=tensor(
                chunk["family_idx"], torch.long, device
            ),
            target_snapshot_batch_idx=torch.zeros(
                n_target,
                dtype=torch.long,
                device=device,
            ),
            target_baseline_logit=tensor(
                chunk["baseline_logit"], torch.float32, device
            ),
        )

        chunk["p_residual_encoder"] = (
            torch.sigmoid(final_logits).detach().cpu().numpy()
        )
        chunk["delta_logit_residual_encoder"] = (
            delta_logits.detach().cpu().numpy()
        )
        chunk["context_n_cells"] = int(len(context))
        chunk["context_n_tests"] = float(context["n_total"].sum())
        output_parts.append(chunk)

    return pd.concat(output_parts, ignore_index=True, sort=False)


def predict_to_impute_cells(
    full_domain: pd.DataFrame,
    observed_standardized: pd.DataFrame,
    assignment: pd.DataFrame,
    phi_table: pd.DataFrame,
    completion_output_dir: Path,
    helper,
    device: torch.device,
    target_batch_size: int,
):
    domain = add_categorical_indices(full_domain, observed_standardized)
    domain = domain.merge(
        assignment.rename(columns={"external_fold": "fold"}),
        on="Country",
        how="left",
        validate="many_to_one",
    )

    missing_assignment = domain["fold"].isna()
    if missing_assignment.any():
        countries = sorted(domain.loc[missing_assignment, "Country"].unique())
        raise ValueError(
            "Some dashboard countries have no external fold assignment: "
            + ", ".join(map(str, countries))
        )
    domain["fold"] = domain["fold"].astype(int)

    all_outputs: list[pd.DataFrame] = []
    exclusions: list[dict[str, object]] = []

    for fold in sorted(domain["fold"].unique().tolist()):
        fold_dir = completion_output_dir / f"fold_{fold:02d}"
        checkpoint_path = find_fold_checkpoint(fold_dir)
        checkpoint = torch_load_checkpoint(checkpoint_path, device)
        model_args = namespace_from_checkpoint(checkpoint)

        checkpoint_fold = int(checkpoint.get("fold", fold))
        if checkpoint_fold != fold:
            raise ValueError(
                f"Checkpoint {checkpoint_path} reports fold {checkpoint_fold}, expected {fold}."
            )

        external_countries = set(
            assignment.loc[
                assignment["external_fold"].eq(fold), "Country"
            ].astype(str)
        )
        train_observed = observed_standardized.loc[
            ~observed_standardized["Country"].astype(str).isin(external_countries)
            & observed_standardized["Year"].le(TRAIN_MAX_YEAR)
        ].copy()
        external_observed = observed_standardized.loc[
            observed_standardized["Country"].astype(str).isin(external_countries)
        ].copy()

        if train_observed.empty or external_observed.empty:
            raise ValueError(f"Fold {fold} has no training or external observed rows.")

        train_features, external_features, global_p = helper.build_fold_features(
            train_df=train_observed,
            external_df=external_observed,
            alpha=float(model_args.alpha),
            beta=float(model_args.beta),
            baseline_mode=str(model_args.baseline_mode),
        )

        n_species = int(checkpoint.get("n_species", observed_standardized["species_idx"].max() + 1))
        n_families = int(checkpoint.get("n_families", observed_standardized["family_idx"].max() + 1))
        model = helper.build_model(
            MODEL_RESIDUAL,
            n_species=n_species,
            n_families=n_families,
            args=model_args,
            device=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        residual_phi = get_phi_info(phi_table, fold, MODEL_RESIDUAL)
        prior_phi = get_phi_info(phi_table, fold, MODEL_PRIOR)

        fold_targets = domain.loc[
            domain["fold"].eq(fold)
            & domain["status"].eq(STATUS_TO_IMPUTE)
        ].copy()

        if fold_targets.empty:
            del model
            continue

        fold_targets["p_species_family_prior"] = np.nan
        fold_targets["p_residual_encoder"] = np.nan
        fold_targets["delta_logit_residual_encoder"] = np.nan
        fold_targets["baseline_source"] = ""
        fold_targets["prediction_context"] = IMPUTED_CONTEXT
        fold_targets["completion_prediction_available"] = False
        fold_targets["completion_exclusion_reason"] = ""

        unsupported_entity = (
            fold_targets["species_idx"].isna()
            | fold_targets["family_idx"].isna()
        )
        for row in fold_targets.loc[unsupported_entity].itertuples(index=False):
            exclusions.append(
                {
                    "fold": fold,
                    "Country": row.Country,
                    "Year": int(row.Year),
                    "Species": row.Species,
                    "Family": row.Family,
                    "reason": "species_or_family_absent_from_global_observed_mapping",
                }
            )
        fold_targets.loc[
            unsupported_entity, "completion_exclusion_reason"
        ] = "species_or_family_absent_from_global_observed_mapping"

        supported = fold_targets.loc[~unsupported_entity].copy()
        supported["species_idx"] = supported["species_idx"].astype(int)
        supported["family_idx"] = supported["family_idx"].astype(int)

        if not supported.empty:
            supported = helper.add_configured_baseline_prediction(
                train_df=train_observed,
                target_df=supported,
                mode=str(model_args.baseline_mode),
                global_p=float(global_p),
                alpha=float(model_args.alpha),
                beta=float(model_args.beta),
                pred_col="p_species_family_prior",
                source_col="baseline_source",
            )
            prior_p = np.clip(
                supported["p_species_family_prior"].to_numpy(float),
                EPS,
                1.0 - EPS,
            )
            supported["baseline_logit"] = np.log(prior_p / (1.0 - prior_p))

            predicted_parts: list[pd.DataFrame] = []
            grouped_targets = supported.groupby(
                ["Country", "Year"],
                sort=True,
            )
            context_groups = {
                (str(country), int(year)): group.copy()
                for (country, year), group in external_features.groupby(
                    ["Country", "Year"],
                    sort=False,
                )
            }

            for (country, year), targets in grouped_targets:
                key = (str(country), int(year))
                context = context_groups.get(key)
                if context is None or context.empty:
                    targets = targets.copy()
                    targets["completion_prediction_available"] = False
                    targets["completion_exclusion_reason"] = (
                        "no_observed_context_in_country_year"
                    )
                    predicted_parts.append(targets)
                    for row in targets.itertuples(index=False):
                        exclusions.append(
                            {
                                "fold": fold,
                                "Country": row.Country,
                                "Year": int(row.Year),
                                "Species": row.Species,
                                "Family": row.Family,
                                "reason": "no_observed_context_in_country_year",
                            }
                        )
                    continue

                predicted = predict_snapshot_targets(
                    model=model,
                    context=context,
                    targets=targets,
                    device=device,
                    batch_size=target_batch_size,
                )
                predicted["completion_prediction_available"] = True
                predicted["completion_exclusion_reason"] = ""
                predicted_parts.append(predicted)

            supported = pd.concat(
                predicted_parts,
                ignore_index=True,
                sort=False,
            )
            supported = add_latent_beta_interval(
                supported,
                p_column="p_species_family_prior",
                phi=float(prior_phi["phi_train"]),
                prefix="p_species_family_prior",
            )
            supported = add_latent_beta_interval(
                supported,
                p_column="p_residual_encoder",
                phi=float(residual_phi["phi_train"]),
                prefix="p_residual_encoder",
            )

            supported["prior_phi_train"] = prior_phi["phi_train"]
            supported["prior_rho_train"] = prior_phi["rho_train"]
            supported["prior_phi_source"] = prior_phi["phi_source"]
            supported["residual_phi_train"] = residual_phi["phi_train"]
            supported["residual_rho_train"] = residual_phi["rho_train"]
            supported["residual_phi_source"] = residual_phi["phi_source"]
            supported["uncertainty_interval_type"] = (
                "latent_beta_population_interval_without_assumed_sample_size"
            )
            supported["model_name"] = MODEL_RESIDUAL
            supported["evaluation_protocol"] = (
                "grouped_country_external_to_impute_full_snapshot_context"
            )
            supported["country_seen_in_parameter_fitting"] = False
            supported["target_outcome_used_as_input"] = False
            supported["checkpoint_path"] = str(checkpoint_path)
            supported["global_train_prior"] = float(global_p)
            supported = helper.add_training_support_flags(
                supported,
                train_observed,
            )

        unsupported_rows = fold_targets.loc[unsupported_entity].copy()
        if not unsupported_rows.empty:
            unsupported_rows["uncertainty_interval_type"] = "not_available"
            unsupported_rows["model_name"] = MODEL_RESIDUAL
            unsupported_rows["evaluation_protocol"] = (
                "grouped_country_external_to_impute_full_snapshot_context"
            )
            unsupported_rows["country_seen_in_parameter_fitting"] = False
            unsupported_rows["target_outcome_used_as_input"] = False
            unsupported_rows["checkpoint_path"] = str(checkpoint_path)
            unsupported_rows["global_train_prior"] = float(global_p)

        fold_parts = [
            part for part in [supported, unsupported_rows] if not part.empty
        ]
        fold_output = pd.concat(
            fold_parts,
            ignore_index=True,
            sort=False,
        )
        all_outputs.append(fold_output)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if all_outputs:
        predictions = pd.concat(all_outputs, ignore_index=True, sort=False)
    else:
        predictions = domain.loc[
            domain["status"].eq(STATUS_TO_IMPUTE)
        ].copy()

    exclusions_frame = pd.DataFrame(exclusions)
    return predictions, exclusions_frame


# -----------------------------------------------------------------------------
# Final dashboard table
# -----------------------------------------------------------------------------


def prepare_intrinsic_rows(full_domain: pd.DataFrame):
    intrinsic = full_domain.loc[
        full_domain["status"].eq(STATUS_INTRINSIC)
    ].copy()
    intrinsic["prediction_context"] = INTRINSIC_CONTEXT
    intrinsic["completion_prediction_available"] = False
    intrinsic["completion_exclusion_reason"] = "intrinsic_resistance"
    intrinsic["uncertainty_interval_type"] = "not_applicable"
    return intrinsic


def prepare_other_status_rows(full_domain: pd.DataFrame):
    known = {STATUS_OBSERVED, STATUS_TO_IMPUTE, STATUS_INTRINSIC}
    other = full_domain.loc[~full_domain["status"].isin(known)].copy()
    other["prediction_context"] = "unsupported_status"
    other["completion_prediction_available"] = False
    other["completion_exclusion_reason"] = "unsupported_status"
    other["uncertainty_interval_type"] = "not_available"
    return other


def select_dashboard_columns(frame: pd.DataFrame):
    preferred = [
        "Country",
        "Year",
        "Species",
        "Family",
        "status",
        "n_S",
        "n_total",
        "prop_S",
        "fold",
        "p_species_family_prior",
        "p_species_family_prior_q05",
        "p_species_family_prior_q95",
        "p_residual_encoder",
        "p_residual_encoder_q05",
        "p_residual_encoder_q95",
        "delta_logit_residual_encoder",
        "baseline_source",
        "prediction_context",
        "completion_prediction_available",
        "completion_exclusion_reason",
        "uncertainty_interval_type",
        "context_n_cells",
        "context_n_tests",
        "snapshot_n_observed_cells",
        "prior_phi_train",
        "prior_rho_train",
        "prior_phi_source",
        "residual_phi_train",
        "residual_rho_train",
        "residual_phi_source",
        "posterior_p_sd",
        "species_seen_in_train",
        "family_seen_in_train",
        "species_family_seen_in_train",
        "both_entities_seen_in_train",
        "country_seen_in_parameter_fitting",
        "target_outcome_used_as_input",
        "evaluation_protocol",
        "checkpoint_path",
        "global_train_prior",
        "domain_cell_generated",
        "dashboard_row_id",
    ]

    out = frame.copy()
    for column in preferred:
        if column not in out.columns:
            out[column] = np.nan
    return out[preferred].sort_values(KEY_COLUMNS).reset_index(drop=True)


def validate_final_landscape(frame: pd.DataFrame):
    if frame.duplicated(KEY_COLUMNS).any():
        examples = frame.loc[
            frame.duplicated(KEY_COLUMNS, keep=False),
            KEY_COLUMNS + ["status"],
        ].head(20)
        raise ValueError(
            "Final landscape has duplicate cells. Examples:\n"
            + examples.to_string(index=False)
        )

    observed = frame.loc[frame["status"].eq(STATUS_OBSERVED)]
    to_impute = frame.loc[frame["status"].eq(STATUS_TO_IMPUTE)]

    if observed.empty:
        raise ValueError("Final landscape contains no observed cells.")
    if to_impute.empty:
        print("Warning: final landscape contains no to_impute cells.")

    n_predicted_to_impute = int(
        to_impute["p_residual_encoder"].notna().sum()
    )
    print("Observed dashboard cells:", len(observed))
    print("To impute dashboard cells:", len(to_impute))
    print("To impute cells with residual predictions:", n_predicted_to_impute)
    print(
        "Intrinsic resistance dashboard cells:",
        int(frame["status"].eq(STATUS_INTRINSIC).sum()),
    )


# -----------------------------------------------------------------------------
# Sampling return from the current completion model
# -----------------------------------------------------------------------------


def prepare_completion_sampling_returns(
    leave_one_out_path: Path,
    destination_dir: Path,
):
    frame = pd.read_csv(leave_one_out_path)
    require_columns(
        frame,
        KEY_COLUMNS
        + [
            "model_name",
            "n_total",
            "prop_S",
            "p_pred",
            "phi_train",
        ],
        leave_one_out_path.name,
    )

    out = frame.loc[
        frame["model_name"].astype(str).eq(MODEL_RESIDUAL)
    ].copy()
    if out.empty:
        raise ValueError(
            "The completion prediction table contains no residual encoder rows."
        )

    for column in ["n_total", "prop_S", "p_pred", "phi_train"]:
        out[column] = safe_numeric(out[column])
    if "n_S" in out.columns:
        out["n_S"] = safe_numeric(out["n_S"])
    if "rho_train" in out.columns:
        out["rho_train"] = safe_numeric(out["rho_train"])
    else:
        out["rho_train"] = 1.0 / (out["phi_train"] + 1.0)

    reason_lists: list[list[str]] = []
    for row in out.itertuples(index=False):
        reasons: list[str] = []
        if not np.isfinite(row.n_total) or row.n_total <= 0:
            reasons.append("n_total_missing_or_nonpositive")
        if not np.isfinite(row.prop_S) or not 0.0 <= row.prop_S <= 1.0:
            reasons.append("observed_susceptibility_invalid")
        if not np.isfinite(row.p_pred) or not 0.0 <= row.p_pred <= 1.0:
            reasons.append("predicted_susceptibility_invalid")
        if not np.isfinite(row.phi_train) or row.phi_train <= 0:
            reasons.append("phi_train_missing_or_nonpositive")
        reason_lists.append(reasons)

    excluded_mask = np.asarray([bool(value) for value in reason_lists])
    excluded = out.loc[excluded_mask].copy()
    if not excluded.empty:
        excluded["sampling_exclusion_reason"] = [
            ";".join(value)
            for value, keep in zip(reason_lists, excluded_mask)
            if keep
        ]

    eligible = out.loc[~excluded_mask].copy()
    if eligible.empty:
        raise ValueError(
            "No residual encoder rows are eligible for the sampling return panel."
        )

    p = np.clip(
        eligible["p_pred"].to_numpy(dtype=float),
        EPS,
        1.0 - EPS,
    )
    phi = eligible["phi_train"].to_numpy(dtype=float)
    n_total = eligible["n_total"].to_numpy(dtype=float)

    floor_variance = p * (1.0 - p) / (phi + 1.0)
    predictive_variance = (
        p
        * (1.0 - p)
        * (phi + n_total)
        / (n_total * (phi + 1.0))
    )

    eligible["prop_S_observed"] = eligible["prop_S"]
    eligible["prop_S_pred"] = eligible["p_pred"]
    eligible["nu_cal"] = eligible["phi_train"]
    eligible["floor_var_cal"] = floor_variance
    eligible["floor_sd_cal"] = np.sqrt(
        np.maximum(floor_variance, 0.0)
    )
    eligible["pred_var_cal"] = predictive_variance
    eligible["pred_sd_cal"] = np.sqrt(
        np.maximum(predictive_variance, 0.0)
    )
    eligible["n_total_over_nu_cal"] = n_total / phi
    eligible["sd_ratio_to_floor"] = (
        eligible["pred_sd_cal"]
        / eligible["floor_sd_cal"].replace(0.0, np.nan)
    )
    eligible["finite_sample_surcharge_fraction"] = (
        eligible["sd_ratio_to_floor"] - 1.0
    )

    if {
        "predictive_prop_q05",
        "predictive_prop_q95",
    }.issubset(eligible.columns):
        eligible["ci_lo_cal"] = safe_numeric(
            eligible["predictive_prop_q05"]
        )
        eligible["ci_hi_cal"] = safe_numeric(
            eligible["predictive_prop_q95"]
        )
        eligible["ci_width_cal"] = (
            eligible["ci_hi_cal"] - eligible["ci_lo_cal"]
        )
    else:
        eligible["ci_lo_cal"] = np.nan
        eligible["ci_hi_cal"] = np.nan
        eligible["ci_width_cal"] = np.nan

    ratio = eligible["n_total_over_nu_cal"]
    eligible["sampling_category"] = np.select(
        [ratio.lt(2.0), ratio.le(10.0)],
        ["sampling_sensitive", "diminishing_returns"],
        default="heterogeneity_dominated",
    )
    eligible["sampling_regime_label"] = eligible[
        "sampling_category"
    ].map(
        {
            "sampling_sensitive": "Sampling sensitive",
            "diminishing_returns": "Diminishing returns",
            "heterogeneity_dominated": "Heterogeneity dominated",
        }
    )
    eligible["abs_error"] = (
        eligible["prop_S_observed"] - eligible["prop_S_pred"]
    ).abs()
    eligible["sampling_scale_source"] = (
        "fold_training_beta_binomial_phi_from_script_02"
    )

    duplicated = eligible.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = eligible.loc[duplicated, KEY_COLUMNS].head(20)
        raise ValueError(
            "Sampling return rows are not unique by dashboard cell.\n"
            + examples.to_string(index=False)
        )

    eligible = eligible.sort_values(KEY_COLUMNS).reset_index(drop=True)
    excluded = excluded.sort_values(KEY_COLUMNS).reset_index(drop=True)

    destination_dir.mkdir(parents=True, exist_ok=True)
    cells_path = destination_dir / "sampling_returns_cells.csv"
    excluded_path = destination_dir / "sampling_returns_excluded.csv"
    metadata_path = destination_dir / "sampling_returns_metadata.json"

    eligible.to_csv(cells_path, index=False)
    excluded.to_csv(excluded_path, index=False)

    category_counts = {
        str(key): int(value)
        for key, value in eligible[
            "sampling_category"
        ].value_counts().items()
    }
    metadata = {
        "source": str(leave_one_out_path),
        "model_name": MODEL_RESIDUAL,
        "cells_output": str(cells_path),
        "excluded_output": str(excluded_path),
        "n_source_rows": int(len(out)),
        "n_eligible_rows": int(len(eligible)),
        "n_excluded_rows": int(len(excluded)),
        "category_counts": category_counts,
        "category_definition": {
            "sampling_sensitive": "n_total < 2 * phi_train",
            "diminishing_returns": (
                "2 * phi_train <= n_total <= 10 * phi_train"
            ),
            "heterogeneity_dominated": "n_total > 10 * phi_train",
        },
        "variance_definition": (
            "Beta binomial rate variance p(1-p)(phi+n)/(n(phi+1)); "
            "the latent floor is p(1-p)/(phi+1)."
        ),
        "interpretation": (
            "The categories describe expected marginal return from additional "
            "isolates under the fitted completion model. They are descriptive "
            "regimes, not formal adequacy thresholds."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


# -----------------------------------------------------------------------------
# Temporal copy
# -----------------------------------------------------------------------------


def build_temporal_rankings_from_jump_output(
    source_dir: Path,
    destination: Path,
    require_2025: bool,
    required_years: list[int],
    alert_threshold: float,
):
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    observed_path = first_existing_path(
        source_dir,
        [
            "temporal_residual_jump_predictions_all_external_tests.csv",
            "temporal_residual_jump_predictions_country_year_generalization_2023_2024.csv",
            "temporal_residual_jump_predictions_2023_2024.csv",
        ],
        "observed temporal jump predictions",
    )
    future_path = first_existing_path(
        source_dir,
        ["temporal_residual_future_jump_predictions.csv"],
        "prospective temporal jump predictions",
        required=require_2025,
    )

    source_frames: list[pd.DataFrame] = []
    observed = pd.read_csv(observed_path)
    observed["dashboard_temporal_source"] = "observed_external_test"
    source_frames.append(observed)

    if future_path is not None:
        future = pd.read_csv(future_path)
        future["dashboard_temporal_source"] = "prospective_forecast"
        source_frames.append(future)

    wide = pd.concat(source_frames, ignore_index=True, sort=False)
    require_columns(
        wide,
        [
            "Country",
            "Species",
            "Family",
            "input_year",
            "target_year",
            "model_name",
            "down_prob",
            "up_prob",
        ],
        "script 05 temporal outputs",
    )

    for column in [
        "input_year",
        "target_year",
        "down_prob",
        "up_prob",
    ]:
        wide[column] = safe_numeric(wide[column])

    wide = wide.dropna(
        subset=[
            "Country",
            "Species",
            "Family",
            "input_year",
            "target_year",
            "model_name",
            "down_prob",
            "up_prob",
        ]
    ).copy()
    wide["input_year"] = wide["input_year"].astype(int)
    wide["target_year"] = wide["target_year"].astype(int)

    nonconsecutive = wide["target_year"].ne(wide["input_year"] + 1)
    if nonconsecutive.any():
        examples = wide.loc[
            nonconsecutive,
            [
                "Country",
                "Species",
                "Family",
                "input_year",
                "target_year",
            ],
        ].head(20)
        raise ValueError(
            "Temporal dashboard rows must be one year forecasts.\n"
            + examples.to_string(index=False)
        )

    requested_years = sorted(set(map(int, required_years)))
    wide = wide.loc[
        wide["target_year"].isin(requested_years)
    ].copy()
    if wide.empty:
        raise ValueError(
            "No script 05 predictions remain for the requested dashboard years."
        )

    if "current_cell_observed" in wide.columns:
        raw_current_observed = wide["current_cell_observed"]
        if raw_current_observed.dtype == bool:
            current_observed = raw_current_observed.astype("boolean")
        else:
            normalized = raw_current_observed.astype(str).str.strip().str.lower()
            current_observed = normalized.map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                    "nan": pd.NA,
                    "none": pd.NA,
                    "": pd.NA,
                }
            ).astype("boolean")
        future_source = wide["dashboard_temporal_source"].eq(
            "prospective_forecast"
        )
        invalid = current_observed.isna() & ~future_source
        if invalid.any():
            raise ValueError(
                "current_cell_observed contains invalid boolean values."
            )
        current_observed = current_observed.fillna(future_source)
        wide = wide.loc[current_observed.astype(bool)].copy()

    for probability_column in ["down_prob", "up_prob"]:
        invalid = (
            wide[probability_column].isna()
            | wide[probability_column].lt(0.0)
            | wide[probability_column].gt(1.0)
        )
        if invalid.any():
            examples = wide.loc[
                invalid,
                [
                    "Country",
                    "Species",
                    "Family",
                    "target_year",
                    probability_column,
                ],
            ].head(20)
            raise ValueError(
                f"{probability_column} must lie between zero and one.\n"
                + examples.to_string(index=False)
            )

    if "target_observed" not in wide.columns:
        wide["target_observed"] = wide[
            "dashboard_temporal_source"
        ].eq("observed_external_test")

    if "p_current" in wide.columns:
        wide["prop_S_prev"] = safe_numeric(wide["p_current"])
    elif "prop_S_current" in wide.columns:
        wide["prop_S_prev"] = safe_numeric(wide["prop_S_current"])
    else:
        wide["prop_S_prev"] = np.nan

    if "prop_S" in wide.columns:
        wide["prop_S_target"] = safe_numeric(wide["prop_S"])
    else:
        wide["prop_S_target"] = np.nan

    if "p_pred_frozen" in wide.columns:
        wide["predicted_prop_S_target"] = safe_numeric(
            wide["p_pred_frozen"]
        )
    else:
        wide["predicted_prop_S_target"] = np.nan

    if "jump_observed_delta" in wide.columns:
        wide["observed_delta"] = safe_numeric(
            wide["jump_observed_delta"]
        )
    else:
        wide["observed_delta"] = (
            wide["prop_S_target"] - wide["prop_S_prev"]
        )

    true_direction = pd.Series(
        pd.NA,
        index=wide.index,
        dtype="string",
    )
    observed_target = wide["target_observed"].fillna(False).astype(bool)
    if {
        "jump_down_label",
        "jump_up_label",
    }.issubset(wide.columns):
        down_label = safe_numeric(wide["jump_down_label"]).fillna(0.0)
        up_label = safe_numeric(wide["jump_up_label"]).fillna(0.0)
        true_direction.loc[observed_target] = "stable"
        true_direction.loc[observed_target & down_label.ge(0.5)] = "down"
        true_direction.loc[observed_target & up_label.ge(0.5)] = "up"
    wide["true_direction"] = true_direction

    long_parts: list[pd.DataFrame] = []
    for direction, probability_column, label_column in [
        ("down", "down_prob", "jump_down_label"),
        ("up", "up_prob", "jump_up_label"),
    ]:
        part = wide.copy()
        part["direction"] = direction
        part["score"] = part[probability_column].astype(float)
        part["jump_probability"] = part["score"]
        part["score_type"] = "learned_probability_score"
        part["score_is_probability"] = True
        part["alert_threshold"] = float(alert_threshold)
        part["predicted_alert"] = part["score"].ge(float(alert_threshold))
        if label_column in part.columns:
            part["is_true_jump"] = (
                safe_numeric(part[label_column]).fillna(0.0).ge(0.5)
            )
        else:
            part["is_true_jump"] = pd.NA
        long_parts.append(part)

    frame = pd.concat(long_parts, ignore_index=True, sort=False)
    frame = frame.sort_values(
        [
            "model_name",
            "direction",
            "Country",
            "input_year",
            "target_year",
            "score",
            "Species",
            "Family",
        ],
        ascending=[True, True, True, True, True, False, True, True],
    ).reset_index(drop=True)
    frame["rank_within_country_year_direction"] = (
        frame.groupby(
            [
                "model_name",
                "direction",
                "Country",
                "input_year",
                "target_year",
            ],
            sort=False,
        )["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    frame["rank_country"] = frame["rank_within_country_year_direction"]
    frame["n_candidates_ranked"] = frame.groupby(
        [
            "model_name",
            "direction",
            "Country",
            "input_year",
            "target_year",
        ],
        sort=False,
    )["score"].transform("size").astype(int)

    unique_key = [
        "model_name",
        "direction",
        "Country",
        "Species",
        "Family",
        "input_year",
        "target_year",
    ]
    duplicated = frame.duplicated(unique_key, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, unique_key].head(20)
        raise ValueError(
            "The script 05 outputs create duplicate dashboard rankings.\n"
            + examples.to_string(index=False)
        )

    years = sorted(frame["target_year"].unique().tolist())
    missing_years = sorted(set(requested_years) - set(years))
    if missing_years:
        raise ValueError(
            "The temporal watchlist is missing required years: "
            + ", ".join(map(str, missing_years))
        )
    if require_2025 and 2025 not in years:
        raise ValueError(
            "The temporal watchlist contains no target year 2025 rows."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)

    metadata_path = destination.with_name(
        "temporal_jump_candidate_rankings_metadata.json"
    )
    metadata = {
        "source_dir": str(source_dir),
        "observed_source": str(observed_path),
        "future_source": str(future_path) if future_path is not None else None,
        "destination": str(destination),
        "n_rows": int(len(frame)),
        "target_years": years,
        "n_2025_rows": int(frame["target_year"].eq(2025).sum()),
        "directions": sorted(frame["direction"].unique().tolist()),
        "ranking_unit": (
            "model, direction, Country, input_year, target_year"
        ),
        "candidate_rule": (
            "Current cell must be observed when the source exports that flag."
        ),
        "temporal_value_column": "jump_probability",
        "temporal_value_type": "learned direction probability",
        "alert_threshold": float(alert_threshold),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def copy_temporal_rankings(
    source: Path,
    destination: Path,
    require_2025: bool,
    required_years: list[int],
    alert_threshold: float,
):
    if not source.exists():
        raise FileNotFoundError(source)

    frame = pd.read_csv(source)

    year_aliases = {
        "year_to": "target_year",
        "Year": "target_year",
        "year_from": "input_year",
    }

    for source_column, canonical_column in year_aliases.items():
        if (
            canonical_column not in frame.columns
            and source_column in frame.columns
        ):
            frame[canonical_column] = frame[source_column]

    required_columns = [
        "Country",
        "Species",
        "Family",
        "target_year",
        "model_name",
        "direction",
        "score",
        "score_is_probability",
    ]

    require_columns(
        frame,
        required_columns,
        source.name,
    )

    frame["target_year"] = safe_numeric(
        frame["target_year"]
    )
    frame["score"] = safe_numeric(
        frame["score"]
    )

    probability_flag = frame["score_is_probability"]

    if probability_flag.dtype != bool:
        probability_flag = (
            probability_flag
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
        )

    if probability_flag.isna().any():
        raise ValueError(
            "score_is_probability contains invalid values."
        )

    frame["score_is_probability"] = (
        probability_flag.astype(bool)
    )

    if not frame["score_is_probability"].all():
        invalid = frame.loc[
            ~frame["score_is_probability"],
            [
                "model_name",
                "score_type",
            ],
        ].drop_duplicates()

        raise ValueError(
            "The dashboard accepts only probabilities "
            "produced by temporal direction heads.\n"
            + invalid.to_string(index=False)
        )

    invalid_score = (
        frame["score"].isna()
        | frame["score"].lt(0.0)
        | frame["score"].gt(1.0)
    )

    if invalid_score.any():
        examples = frame.loc[
            invalid_score,
            [
                "model_name",
                "direction",
                "score",
            ],
        ].head(20)

        raise ValueError(
            "Temporal jump probabilities must be "
            "between zero and one.\n"
            + examples.to_string(index=False)
        )

    frame = frame.dropna(
        subset=[
            "Country",
            "Species",
            "Family",
            "target_year",
            "model_name",
            "direction",
            "score",
        ]
    ).copy()

    frame["target_year"] = (
        frame["target_year"].astype(int)
    )

    if "input_year" not in frame.columns:
        frame["input_year"] = (
            frame["target_year"] - 1
        )
    else:
        frame["input_year"] = safe_numeric(
            frame["input_year"]
        )

    frame["jump_probability"] = frame["score"]
    frame["score_type"] = "learned_probability_score"
    frame["alert_threshold"] = float(alert_threshold)
    if "predicted_alert" not in frame.columns:
        frame["predicted_alert"] = frame["score"].ge(float(alert_threshold))
    if "rank_country" not in frame.columns:
        frame["rank_country"] = (
            frame.groupby(
                [
                    "model_name",
                    "direction",
                    "Country",
                    "input_year",
                    "target_year",
                ],
                sort=False,
            )["score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )

    if "p_current" in frame.columns:
        frame["prop_S_prev"] = safe_numeric(
            frame["p_current"]
        )

    if (
        "jump_observed_delta" in frame.columns
        and "prop_S_prev" in frame.columns
    ):
        frame["prop_S_target"] = (
            frame["prop_S_prev"]
            + safe_numeric(
                frame["jump_observed_delta"]
            )
        )

    duplicated = frame.duplicated(
        [
            "model_name",
            "direction",
            "Country",
            "Species",
            "Family",
            "target_year",
        ],
        keep=False,
    )

    if duplicated.any():
        examples = frame.loc[
            duplicated,
            [
                "model_name",
                "direction",
                "Country",
                "Species",
                "Family",
                "target_year",
            ],
        ].head(20)

        raise ValueError(
            "The temporal ranking contains duplicate rows.\n"
            + examples.to_string(index=False)
        )

    years = sorted(
        frame["target_year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if require_2025 and 2025 not in years:
        raise ValueError(
            "The temporal ranking contains no "
            "target year 2025 rows."
        )

    missing_years = sorted(
        set(required_years) - set(years)
    )

    if missing_years:
        raise ValueError(
            "The temporal ranking is missing required years: "
            + ", ".join(map(str, missing_years))
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        destination,
        index=False,
    )

    return {
        "source": str(source),
        "destination": str(destination),
        "n_rows": int(len(frame)),
        "target_years": years,
        "n_2025_rows": int(
            frame["target_year"].eq(2025).sum()
        ),
        "temporal_value_column": "jump_probability",
        "temporal_value_type": (
            "learned direction probability"
        ),
        "alert_threshold": float(alert_threshold),
    }

GNN_SAMPLING_REQUIRED_COLUMNS = [
    "Country",
    "Year",
    "Species",
    "Family",
    "n_total",
    "prop_S_observed",
    "prop_S_pred",
    "nu_cal",
    "floor_sd_cal",
    "ci_lo_cal",
    "ci_hi_cal",
    "ci_width_cal",
    "fold_model",
]


def prepare_gnn_sampling_returns(
    observed_path: Path,
    destination_dir: Path,
):
    """Validate GNN observed reconstructions and derive sampling return regimes.

    The regimes retain the original numerical cut points but use neutral labels
    describing expected marginal return from additional sampling. They do not
    state that a surveillance system collected too few or too many isolates.
    """
    frame = pd.read_csv(observed_path)
    require_columns(
        frame,
        GNN_SAMPLING_REQUIRED_COLUMNS,
        observed_path.name,
    )

    out = frame.copy()
    for column in ["Country", "Species", "Family"]:
        out[column] = out[column].astype("string").str.strip()
    for column in [
        "Year",
        "n_total",
        "prop_S_observed",
        "prop_S_pred",
        "nu_cal",
        "floor_sd_cal",
        "ci_lo_cal",
        "ci_hi_cal",
        "ci_width_cal",
        "fold_model",
    ]:
        out[column] = safe_numeric(out[column])

    missing_key = out[KEY_COLUMNS].isna().any(axis=1)
    if missing_key.any():
        examples = out.loc[missing_key, KEY_COLUMNS].head(20)
        raise ValueError(
            "The GNN observed reconstruction contains incomplete cell keys. "
            "Examples:\n" + examples.to_string(index=False)
        )

    out["Year"] = out["Year"].astype(int)
    duplicated = out.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = out.loc[duplicated, KEY_COLUMNS].head(20)
        raise ValueError(
            "The GNN observed reconstruction contains duplicate Country Year "
            "Species Family cells. Examples:\n"
            + examples.to_string(index=False)
        )

    negative_floor = out["floor_sd_cal"].notna() & out["floor_sd_cal"].lt(0)
    if negative_floor.any():
        examples = out.loc[
            negative_floor,
            KEY_COLUMNS + ["floor_sd_cal"],
        ].head(20)
        raise ValueError(
            "The GNN observed reconstruction contains negative floor_sd_cal. "
            "Examples:\n" + examples.to_string(index=False)
        )

    bounds_missing = out[["ci_lo_cal", "ci_hi_cal"]].isna().any(axis=1)
    if bounds_missing.any():
        examples = out.loc[
            bounds_missing,
            KEY_COLUMNS + ["ci_lo_cal", "ci_hi_cal", "ci_width_cal"],
        ].head(20)
        raise ValueError(
            "The GNN observed reconstruction contains missing calibrated "
            "interval bounds, so ci_width_cal cannot be verified. Examples:\n"
            + examples.to_string(index=False)
        )

    expected_width = out["ci_hi_cal"] - out["ci_lo_cal"]
    width_available = out["ci_width_cal"].notna()
    tolerance = np.maximum(
        1e-8,
        1e-6 * np.maximum(1.0, expected_width.abs().to_numpy(float)),
    )
    mismatch = width_available & (
        (out["ci_width_cal"] - expected_width).abs().to_numpy(float)
        > tolerance
    )
    if mismatch.any():
        examples = out.loc[
            mismatch,
            KEY_COLUMNS + ["ci_lo_cal", "ci_hi_cal", "ci_width_cal"],
        ].head(20)
        raise ValueError(
            "ci_width_cal does not equal ci_hi_cal minus ci_lo_cal. Examples:\n"
            + examples.to_string(index=False)
        )

    exclusion_reasons: list[list[str]] = []
    for row in out.itertuples(index=False):
        reasons: list[str] = []
        if not np.isfinite(row.n_total) or row.n_total <= 0:
            reasons.append("n_total_missing_or_nonpositive")
        if not np.isfinite(row.nu_cal) or row.nu_cal <= 0:
            reasons.append("nu_cal_missing_or_nonpositive")
        if not np.isfinite(row.ci_width_cal):
            reasons.append("ci_width_cal_missing")
        exclusion_reasons.append(reasons)

    excluded_mask = np.asarray([bool(value) for value in exclusion_reasons])
    excluded = out.loc[excluded_mask].copy()
    if not excluded.empty:
        excluded["sampling_exclusion_reason"] = [
            ";".join(value)
            for value, is_excluded in zip(exclusion_reasons, excluded_mask)
            if is_excluded
        ]

    eligible = out.loc[~excluded_mask].copy()
    ratio = eligible["n_total"] / eligible["nu_cal"]
    eligible["n_total_over_nu_cal"] = ratio
    eligible["sampling_category"] = np.select(
        [ratio.lt(2.0), ratio.le(10.0)],
        ["sampling_sensitive", "diminishing_returns"],
        default="heterogeneity_dominated",
    )
    eligible["sampling_regime_label"] = eligible["sampling_category"].map(
        {
            "sampling_sensitive": "Sampling sensitive",
            "diminishing_returns": "Diminishing returns",
            "heterogeneity_dominated": "Heterogeneity dominated",
        }
    )
    eligible["abs_error"] = (
        eligible["prop_S_observed"] - eligible["prop_S_pred"]
    ).abs()

    eligible = eligible.sort_values(KEY_COLUMNS).reset_index(drop=True)
    excluded = excluded.sort_values(KEY_COLUMNS).reset_index(drop=True)

    destination_dir.mkdir(parents=True, exist_ok=True)
    cells_path = destination_dir / "sampling_returns_cells.csv"
    excluded_path = destination_dir / "sampling_returns_excluded.csv"
    metadata_path = destination_dir / "sampling_returns_metadata.json"

    eligible.to_csv(cells_path, index=False)
    excluded.to_csv(excluded_path, index=False)

    category_counts = {
        str(key): int(value)
        for key, value in eligible["sampling_category"].value_counts().items()
    }
    metadata = {
        "source": str(observed_path),
        "cells_output": str(cells_path),
        "excluded_output": str(excluded_path),
        "n_source_rows": int(len(out)),
        "n_eligible_rows": int(len(eligible)),
        "n_excluded_rows": int(len(excluded)),
        "category_counts": category_counts,
        "category_definition": {
            "sampling_sensitive": "n_total < 2 * nu_cal",
            "diminishing_returns": "2 * nu_cal <= n_total <= 10 * nu_cal",
            "heterogeneity_dominated": "n_total > 10 * nu_cal",
        },
        "interpretation": (
            "Regimes describe expected marginal return from additional "
            "sampling under the calibrated GNN model. They are not formal "
            "surveillance adequacy thresholds and do not indicate too few or "
            "too many collected isolates."
        ),
        "validation": {
            "unique_country_year_species_family": True,
            "ci_width_matches_bounds": True,
            "floor_sd_nonnegative": True,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


GRAPH_EXPORT_FILES = [
    "reconstruction_observed_loo.csv",
    "reconstruction_imputed.csv",
    "uncertainty_decile_table.csv",
    "uncertainty_ordering.csv",
    "uncertainty_ordering_by_n_stratum.csv",
    "sampling_effort_summary.csv",
    "threshold_diagnostic.csv",
    "precision_at_k_movers_only.csv",
    "gated_alerts_summary.csv",
    "alert_coverage_summary.csv",
    "observed_direction_forecast.csv",
    "projection_next_year.csv",
    "README.json",
]


def copy_graph_exports(source_dir: Path, output_dir: Path):
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    required = [
        source_dir / "reconstruction_observed_loo.csv",
        source_dir / "reconstruction_imputed.csv",
        source_dir / "observed_direction_forecast.csv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    destination_dir = output_dir / "graph_model"
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    missing_optional = []

    for name in GRAPH_EXPORT_FILES:
        source = source_dir / name
        if not source.exists():
            missing_optional.append(name)
            continue
        destination = destination_dir / name
        shutil.copyfile(source, destination)
        copied.append(str(destination))

    sampling_returns = prepare_gnn_sampling_returns(
        observed_path=(
            destination_dir / "reconstruction_observed_loo.csv"
        ),
        destination_dir=destination_dir,
    )

    return {
        "source_dir": str(source_dir),
        "destination_dir": str(destination_dir),
        "copied_files": copied,
        "missing_optional_files": missing_optional,
        "sampling_returns": sampling_returns,
    }



def _dashboard_boolean(values: pd.Series, column_name: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    )
    if mapped.isna().any():
        examples = values.loc[mapped.isna()].head(20).tolist()
        raise ValueError(
            f"{column_name} contains invalid boolean values: {examples}"
        )
    return mapped.astype(bool)


def _probability_from_logit(values: pd.Series) -> pd.Series:
    numeric = safe_numeric(values)
    clipped = numeric.clip(lower=-60.0, upper=60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def append_gnn_direction_rankings(
    direction_path: Path,
    destination: Path,
    required_years: list[int],
    require_2025: bool,
):
    """Append notebook GNN direction probabilities to the long jump ranking.

    The notebook file has one wide row per current observed cell with separate
    down and up probabilities. Rows are converted to the same long schema used
    by the temporal residual jump heads. Notebook alert flags are preserved,
    because their thresholds are fitted on logits inside each fold.
    """
    if not direction_path.exists():
        raise FileNotFoundError(direction_path)
    if not destination.exists():
        raise FileNotFoundError(
            "Residual temporal ranking must be created before GNN rows are "
            f"appended: {destination}"
        )

    raw = pd.read_csv(direction_path)
    required = [
        "Country",
        "year_from",
        "year_to",
        "Species",
        "Family",
        "prop_S_current",
        "down_prob",
        "up_prob",
        "tau_down",
        "tau_up",
        "next_year_in_data",
        "fold_model",
        "down_alert_rank",
        "is_down_alert",
        "up_alert_rank",
        "is_up_alert",
    ]
    require_columns(raw, required, direction_path.name)

    for column in ["Country", "Species", "Family"]:
        raw[column] = raw[column].astype("string").str.strip()
    for column in [
        "year_from",
        "year_to",
        "prop_S_current",
        "down_prob",
        "up_prob",
        "tau_down",
        "tau_up",
        "fold_model",
        "down_alert_rank",
        "up_alert_rank",
    ]:
        raw[column] = safe_numeric(raw[column])

    raw["next_year_in_data"] = _dashboard_boolean(
        raw["next_year_in_data"],
        "next_year_in_data",
    )
    raw["is_down_alert"] = _dashboard_boolean(
        raw["is_down_alert"],
        "is_down_alert",
    )
    raw["is_up_alert"] = _dashboard_boolean(
        raw["is_up_alert"],
        "is_up_alert",
    )

    raw = raw.dropna(
        subset=[
            "Country",
            "year_from",
            "year_to",
            "Species",
            "Family",
            "down_prob",
            "up_prob",
            "fold_model",
        ]
    ).copy()
    raw["year_from"] = raw["year_from"].astype(int)
    raw["year_to"] = raw["year_to"].astype(int)
    raw["fold_model"] = raw["fold_model"].astype(int)

    nonconsecutive = raw["year_to"].ne(raw["year_from"] + 1)
    if nonconsecutive.any():
        examples = raw.loc[
            nonconsecutive,
            [
                "Country",
                "year_from",
                "year_to",
                "Species",
                "Family",
            ],
        ].head(20)
        raise ValueError(
            "GNN direction rows must describe consecutive years.\n"
            + examples.to_string(index=False)
        )

    requested_years = sorted(set(map(int, required_years)))
    raw = raw.loc[raw["year_to"].isin(requested_years)].copy()
    if raw.empty:
        raise ValueError(
            "No GNN direction rows remain for the requested dashboard years."
        )

    for column in ["down_prob", "up_prob"]:
        invalid = raw[column].isna() | raw[column].lt(0.0) | raw[column].gt(1.0)
        if invalid.any():
            examples = raw.loc[
                invalid,
                [
                    "Country",
                    "year_to",
                    "Species",
                    "Family",
                    column,
                ],
            ].head(20)
            raise ValueError(
                f"GNN {column} must lie between zero and one.\n"
                + examples.to_string(index=False)
            )

    if "true_direction" not in raw.columns:
        raw["true_direction"] = pd.NA
    else:
        normalized_direction = (
            raw["true_direction"]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        normalized_direction = normalized_direction.where(
            normalized_direction.isin(["down", "stable", "up"]),
            pd.NA,
        )
        raw["true_direction"] = normalized_direction

    long_parts: list[pd.DataFrame] = []
    for direction, probability_column, tau_column, rank_column, alert_column in [
        (
            "down",
            "down_prob",
            "tau_down",
            "down_alert_rank",
            "is_down_alert",
        ),
        (
            "up",
            "up_prob",
            "tau_up",
            "up_alert_rank",
            "is_up_alert",
        ),
    ]:
        part = raw.copy()
        part["input_year"] = part["year_from"]
        part["target_year"] = part["year_to"]
        part["model_name"] = np.where(
            part["target_year"].eq(2025),
            "gnn_future",
            "gnn",
        )
        part["direction"] = direction
        part["score"] = part[probability_column].astype(float)
        part["jump_probability"] = part["score"]
        part["score_type"] = "learned_probability_score"
        part["score_is_probability"] = True
        part["alert_threshold"] = _probability_from_logit(part[tau_column])
        part["predicted_alert"] = part[alert_column].astype(bool)
        part["notebook_alert_rank"] = safe_numeric(part[rank_column])
        part["rank_country"] = (
            part.groupby(
                [
                    "model_name",
                    "direction",
                    "Country",
                    "input_year",
                    "target_year",
                ],
                sort=False,
            )["score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        part["rank_within_country_year_direction"] = part["rank_country"]
        part["n_candidates_ranked"] = part.groupby(
            [
                "model_name",
                "direction",
                "Country",
                "input_year",
                "target_year",
            ],
            sort=False,
        )["score"].transform("size").astype(int)
        part["prop_S_prev"] = safe_numeric(part["prop_S_current"])
        part["target_observed"] = part["next_year_in_data"].astype(bool)
        part["prospective"] = part["target_year"].eq(2025)
        part["is_true_jump"] = np.where(
            part["true_direction"].notna(),
            part["true_direction"].eq(direction),
            pd.NA,
        )
        part["dashboard_temporal_source"] = (
            "gnn_notebook_observed_direction_forecast"
        )
        long_parts.append(part)

    gnn = pd.concat(long_parts, ignore_index=True, sort=False)
    residual = pd.read_csv(destination)
    combined = pd.concat([residual, gnn], ignore_index=True, sort=False)

    unique_key = [
        "model_name",
        "direction",
        "Country",
        "Species",
        "Family",
        "input_year",
        "target_year",
    ]
    duplicated = combined.duplicated(unique_key, keep=False)
    if duplicated.any():
        examples = combined.loc[duplicated, unique_key].head(20)
        raise ValueError(
            "Residual and GNN rows create duplicate dashboard rankings.\n"
            + examples.to_string(index=False)
        )

    combined = combined.sort_values(
        [
            "model_name",
            "direction",
            "target_year",
            "Country",
            "score",
            "Species",
            "Family",
        ],
        ascending=[True, True, True, True, False, True, True],
    ).reset_index(drop=True)

    years = sorted(
        pd.to_numeric(combined["target_year"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    missing_years = sorted(set(requested_years) - set(years))
    if missing_years:
        raise ValueError(
            "The combined temporal ranking is missing required years: "
            + ", ".join(map(str, missing_years))
        )
    if require_2025 and 2025 not in years:
        raise ValueError(
            "The combined temporal ranking contains no target year 2025 rows."
        )

    combined.to_csv(destination, index=False)

    metadata_path = destination.with_name(
        "temporal_jump_candidate_rankings_metadata.json"
    )
    metadata: dict[str, object] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "destination": str(destination),
            "gnn_direction_source": str(direction_path),
            "n_rows": int(len(combined)),
            "n_gnn_rows": int(len(gnn)),
            "n_residual_rows": int(len(residual)),
            "target_years": years,
            "n_2025_rows": int(
                pd.to_numeric(
                    combined["target_year"],
                    errors="coerce",
                ).eq(2025).sum()
            ),
            "models": sorted(
                combined["model_name"].dropna().astype(str).unique().tolist()
            ),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return {
        "source": str(direction_path),
        "destination": str(destination),
        "n_rows_added": int(len(gnn)),
        "n_rows_total": int(len(combined)),
        "target_years": years,
        "models": metadata["models"],
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    args = parse_args()
    if args.target_batch_size < 1:
        raise ValueError("target_batch_size must be at least one.")

    completion_dir = args.completion_output_dir
    assignment_path = first_existing_path(
        completion_dir,
        ["country_generalization_fold_assignment.csv"],
        "completion fold assignment",
    )
    loo_path = first_existing_path(
        completion_dir,
        ["country_generalization_leave_one_out_predictions.csv"],
        "completion leave one out predictions",
    )
    phi_path = first_existing_path(
        completion_dir,
        [
            "country_generalization_historical_phi_train.csv",
            "country_generalization_phi_train.csv",
        ],
        "completion training phi table",
    )

    helper = load_script_02(args.completion_script_path)
    device = choose_device(args.device)

    print("Dashboard data preparation")
    print("Device:", device)
    print("Standard data:", args.input_path)
    print("Completion outputs:", completion_dir)
    if args.jump_output_dir is not None:
        print("Jump outputs:", args.jump_output_dir)

    full_domain = load_full_domain(args.input_path)

    observed_raw = helper.standardize_observed_data(args.input_path)
    if all(
        hasattr(helper, name)
        for name in [
            "split_fixed_temporal_vault",
            "rebuild_modeling_identifiers",
        ]
    ):
        historical_pool, vault_pool, _ = helper.split_fixed_temporal_vault(
            observed_raw
        )
        observed_standardized = helper.rebuild_modeling_identifiers(
            pd.concat(
                [historical_pool, vault_pool],
                ignore_index=True,
                sort=False,
            )
        )
    else:
        observed_standardized = observed_raw

    assignment = load_fold_assignment(assignment_path)
    phi_table = load_phi_table(phi_path)

    observed_rows = prepare_observed_leave_one_out(
        loo_path,
        full_domain,
    )
    to_impute_rows, exclusions = predict_to_impute_cells(
        full_domain=full_domain,
        observed_standardized=observed_standardized,
        assignment=assignment,
        phi_table=phi_table,
        completion_output_dir=completion_dir,
        helper=helper,
        device=device,
        target_batch_size=args.target_batch_size,
    )
    intrinsic_rows = prepare_intrinsic_rows(full_domain)
    other_rows = prepare_other_status_rows(full_domain)

    final = pd.concat(
        [observed_rows, to_impute_rows, intrinsic_rows, other_rows],
        ignore_index=True,
        sort=False,
    )
    final = select_dashboard_columns(final)
    validate_final_landscape(final)

    landscape_dir = args.output_dir / "landscape_prediction"
    landscape_dir.mkdir(parents=True, exist_ok=True)
    landscape_path = landscape_dir / "landscape_predictions.csv"
    exclusion_path = landscape_dir / "dashboard_completion_exclusions.csv"
    metadata_path = landscape_dir / "dashboard_data_metadata.json"

    final.to_csv(landscape_path, index=False)
    exclusions.to_csv(exclusion_path, index=False)

    sampling_metadata = prepare_completion_sampling_returns(
        leave_one_out_path=loo_path,
        destination_dir=args.output_dir / "sampling_returns",
    )

    temporal_path = (
        args.output_dir
        / "temporal_prediction"
        / "temporal_jump_candidate_rankings.csv"
    )
    if args.temporal_rankings_path is not None:
        temporal_metadata = copy_temporal_rankings(
            source=args.temporal_rankings_path,
            destination=temporal_path,
            require_2025=args.require_2025_temporal,
            required_years=args.required_temporal_years,
            alert_threshold=args.jump_probability_threshold,
        )
    else:
        temporal_metadata = build_temporal_rankings_from_jump_output(
            source_dir=args.jump_output_dir,
            destination=temporal_path,
            require_2025=args.require_2025_temporal,
            required_years=args.required_temporal_years,
            alert_threshold=args.jump_probability_threshold,
        )

    graph_metadata = None
    gnn_temporal_metadata = None
    if args.gnn_export_dir is not None:
        graph_metadata = copy_graph_exports(
            source_dir=args.gnn_export_dir,
            output_dir=args.output_dir,
        )
        gnn_direction_path = (
            args.output_dir
            / "graph_model"
            / "observed_direction_forecast.csv"
        )
        if gnn_direction_path.exists():
            gnn_temporal_metadata = append_gnn_direction_rankings(
                direction_path=gnn_direction_path,
                destination=temporal_path,
                required_years=args.required_temporal_years,
                require_2025=args.require_2025_temporal,
            )

    metadata = {
        "input_path": str(args.input_path),
        "completion_output_dir": str(completion_dir),
        "completion_script_path": str(args.completion_script_path),
        "completion_phi_path": str(phi_path),
        "landscape_output": str(landscape_path),
        "completion_exclusions_output": str(exclusion_path),
        "n_landscape_rows": int(len(final)),
        "status_counts": {
            str(key): int(value)
            for key, value in final["status"].value_counts(dropna=False).items()
        },
        "n_to_impute_predictions": int(
            final.loc[
                final["status"].eq(STATUS_TO_IMPUTE),
                "p_residual_encoder",
            ].notna().sum()
        ),
        "observed_prediction_context": OBSERVED_CONTEXT,
        "to_impute_prediction_context": IMPUTED_CONTEXT,
        "to_impute_uncertainty": (
            "Beta population interval using fold training phi. This is not a "
            "count predictive interval because unsampled cells have no n_total."
        ),
        "sampling_returns": sampling_metadata,
        "temporal_watchlist": temporal_metadata,
        "graph_copy": graph_metadata,
        "gnn_temporal_merge": gnn_temporal_metadata,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSaved dashboard landscape data:")
    print(" ", landscape_path)
    print(" ", exclusion_path)
    print(" ", metadata_path)
    print("Saved sampling return data:")
    print(" ", sampling_metadata["cells_output"])
    print("Saved temporal watchlist:")
    print(" ", temporal_metadata["destination"])
    print("2025 temporal rows:", temporal_metadata["n_2025_rows"])
    if graph_metadata is not None:
        print("Saved GNN notebook exports:")
        print(" ", graph_metadata["destination_dir"])
    if gnn_temporal_metadata is not None:
        print("Merged GNN direction probabilities:")
        print(" ", gnn_temporal_metadata["n_rows_added"], "rows")


if __name__ == "__main__":
    main()
