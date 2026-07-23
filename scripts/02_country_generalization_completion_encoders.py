"""
Five fold country generalization benchmark for AMR completion encoders.

Each fold uses the exact country assignment supplied by the GNN fold file. One
model is trained per fold using only the fold training countries and only
observations with Year less than or equal to 2022. The trained fold model, the
Species Family prior, and the beta binomial dispersion are then frozen.

The same frozen fold model is evaluated on two distinct external country test
sets.

1. Historical country holdout
   External fold countries with Year less than or equal to 2022. This measures
   generalization to countries absent from fitting while remaining inside the
   historical time period.

2. Temporal vault country holdout
   The same external fold countries, restricted to 2023 and 2024. This measures
   simultaneous generalization to unseen countries and years excluded from all
   fitting.

Every eligible external Country Year snapshot is evaluated exhaustively by
cellwise leave one out reconstruction. The primary reported result for each
test set is the mean and sample standard deviation of the five fold metrics.
There is no validation set and no early stopping. Every fold model is trained
for the configured fixed number of epochs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import beta as beta_distribution
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT_FROM_SCRIPT))

from src.CONFIG import EPS
from src.baselines import (
    add_configured_baseline_prediction,
    add_configured_baseline_prediction_leave_one_out,
    fit_global_phi_for_baseline,
    smoothed_rate,
)
from src.models import (
    SnapshotEncoderResidualModel,
)
from src.training import (
    beta_binomial_nll_from_prob,
    train_one_epoch_residual_snapshot_encoder,
)
from src.utils import choose_device


MODEL_PRIOR = "species_family_prior"
MODEL_RESIDUAL = "snapshot_encoder_residual_model"

MODEL_ALIASES = {
    "prior": MODEL_PRIOR,
    "residual": MODEL_RESIDUAL,
}

TRAIN_MAX_YEAR = 2022
TEST_YEARS = (2023, 2024)
EVALUATION_SET_HISTORICAL = "historical_country_holdout_through_2022"
EVALUATION_SET_VAULT = "external_country_temporal_vault_2023_2024"
EVALUATION_PROTOCOL = (
    "gnn_country_folds_train_through_2022_dual_external_test_"
    "cellwise_leave_one_out"
)


# -----------------------------------------------------------------------------
# Arguments and reproducibility
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train one AMR completion encoder per GNN country fold using "
            "training countries through 2022, then evaluate the same frozen "
            "model on historical and 2023 to 2024 external country snapshots."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Standardized AMR dataset produced by 01_create_standard_dataset.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["residual"],
        default=["residual"],
        help="Neural encoders to train. The Species-Family prior is always evaluated.",
    )    
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fold-random-state",
        type=int,
        default=42,
        help=(
            "Random state used by the colleague KFold country partition. "
            "Ignored when --country-folds-path is supplied."
        ),
    )
    parser.add_argument(
        "--country-folds-path",
        type=Path,
        required=True,
        help=(
            "Required exact GNN fold file. Accepts folds.json mapping "
            "zero-based fold ids to country lists, or a CSV containing Country "
            "and one of fold, external_fold, fold_model, or colleague_fold. "
            "The reconstruction export itself is accepted because repeated "
            "country rows are collapsed when their fold is consistent."
        ),
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-context-frac", type=float, default=0.75)
    parser.add_argument("--min-input-cells", type=int, default=1)
    parser.add_argument("--min-target-cells", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--entity-emb-dim", type=int, default=16)
    parser.add_argument("--edge-hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--decoder-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--baseline-mode",
        choices=["species_family"],
        default="species_family",
        help="Country-aware priors are forbidden in an external-country benchmark.",
    )

    parser.add_argument("--min-logvar", type=float, default=-8.0)
    parser.add_argument("--max-logvar", type=float, default=4.0)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--kl-anneal-epochs", type=int, default=50)
    parser.add_argument("--n-train-samples", type=int, default=1)
    parser.add_argument(
        "--n-eval-samples",
        type=int,
        default=128,
        help="Posterior latent samples per leave-one-out target for the variational model.",
    )
    parser.add_argument(
        "--n-predictive-samples",
        type=int,
        default=512,
        help="Samples used for beta-binomial posterior-predictive intervals.",
    )
    parser.add_argument(
        "--loo-target-batch-size",
        type=int,
        default=64,
        help=(
            "Number of held-out targets vectorized together. Each target has its own "
            "copy of the remaining snapshot context. Reduce this if GPU memory complains."
        ),
    )
    parser.add_argument(
        "--phi-calibration-context-frac",
        type=float,
        default=0.75,
        help=(
            "One fixed random-mask reconstruction pass on training snapshots is used "
            "to fit fold-level phi when the model does not expose learned log_phi."
        ),
    )
    parser.add_argument("--min-country-cells-for-posthoc-phi", type=int, default=20)
    parser.add_argument("--min-country-tests-for-posthoc-phi", type=int, default=100)
    parser.add_argument(
        "--save-fold-models",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    if not 0.0 < args.train_context_frac < 1.0:
        raise ValueError("--train-context-frac must be strictly between 0 and 1.")
    if not 0.0 < args.phi_calibration_context_frac < 1.0:
        raise ValueError("--phi-calibration-context-frac must be strictly between 0 and 1.")
    if args.n_folds != 5:
        raise ValueError("--n-folds must be exactly 5 for the GNN fold protocol.")
    if args.loo_target_batch_size < 1:
        raise ValueError("--loo-target-batch-size must be at least 1.")
    if args.n_eval_samples < 2:
        raise ValueError("--n-eval-samples must be at least 2.")
    if args.n_predictive_samples < 20:
        raise ValueError("--n-predictive-samples should be at least 20.")

    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_kl_weight(epoch: int, final_weight: float, anneal_epochs: int):
    if anneal_epochs <= 0:
        return float(final_weight)
    return float(final_weight) * min(float(epoch) / float(anneal_epochs), 1.0)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------


def standardize_observed_data(path: Path) :
    df = pd.read_csv(path)

    required = [
        "Country",
        "Year",
        "Species",
        "Family",
        "status",
        "n_S",
        "n_total",
        "prop_S",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Input dataset misses required columns: {missing}")

    out = df.copy()
    for column in ["Country", "Species", "Family", "status"]:
        out[column] = out[column].astype("string").str.strip()

    country_order = {
        country: index
        for index, country in enumerate(
            out["Country"].dropna().astype(str).drop_duplicates().tolist()
        )
    }
    out["country_input_order"] = out["Country"].astype(str).map(country_order)

    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out["n_S"] = pd.to_numeric(out["n_S"], errors="coerce")
    out["n_total"] = pd.to_numeric(out["n_total"], errors="coerce")
    out["prop_S"] = pd.to_numeric(out["prop_S"], errors="coerce")

    observed_labels = {"observed", "Observed"}
    out = out[out["status"].isin(observed_labels)].copy()
    out = out.dropna(
        subset=["Country", "Year", "Species", "Family", "n_S", "n_total", "prop_S"]
    )
    out = out[out["n_total"] > 0].copy()

    if out.empty:
        raise ValueError("No usable observed rows remain after input validation.")

    out["Year"] = out["Year"].astype(int)
    out["n_S"] = out["n_S"].astype(float)
    out["n_total"] = out["n_total"].astype(float)
    out["prop_S"] = (out["n_S"] / out["n_total"]).clip(0.0, 1.0)
    out["snapshot_id"] = out["Country"].astype(str) + "||" + out["Year"].astype(str)

    duplicate_key = ["Country", "Year", "Species", "Family"]
    if out.duplicated(duplicate_key).any():
        duplicated = int(out.duplicated(duplicate_key, keep=False).sum())
        raise ValueError(
            f"Found {duplicated} duplicated observed rows for {duplicate_key}. "
            "Aggregate the dataset before running country generalization."
        )

    species_values = sorted(out["Species"].astype(str).unique().tolist())
    family_values = sorted(out["Family"].astype(str).unique().tolist())
    species_to_idx = {value: idx for idx, value in enumerate(species_values)}
    family_to_idx = {value: idx for idx, value in enumerate(family_values)}

    out["species_idx"] = out["Species"].astype(str).map(species_to_idx).astype(int)
    out["family_idx"] = out["Family"].astype(str).map(family_to_idx).astype(int)
    out["row_id"] = np.arange(len(out), dtype=np.int64)

    snapshot_map = {
        snapshot_id: idx
        for idx, snapshot_id in enumerate(sorted(out["snapshot_id"].unique().tolist()))
    }
    out["snapshot_idx"] = out["snapshot_id"].map(snapshot_map).astype(int)

    return out.reset_index(drop=True)


def split_fixed_temporal_vault(
    observed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the immutable historical fitting pool and 2023 to 2024 vault."""
    historical = observed.loc[observed["Year"] <= TRAIN_MAX_YEAR].copy()
    vault = observed.loc[observed["Year"].isin(TEST_YEARS)].copy()
    ignored = observed.loc[
        ~observed.index.isin(historical.index)
        & ~observed.index.isin(vault.index)
    ].copy()

    if historical.empty:
        raise ValueError(
            f"No historical observations with Year <= {TRAIN_MAX_YEAR}."
        )
    if vault.empty:
        raise ValueError(
            f"No observations were found in the fixed test years {TEST_YEARS}."
        )

    missing_test_years = sorted(set(TEST_YEARS) - set(vault["Year"].unique()))
    if missing_test_years:
        raise ValueError(
            "The fixed temporal vault is incomplete. Missing test years: "
            f"{missing_test_years}"
        )
    if set(historical.index) & set(vault.index):
        raise AssertionError("Historical fitting rows overlap the temporal vault.")
    if historical["Year"].gt(TRAIN_MAX_YEAR).any():
        raise AssertionError("A post 2022 row entered the historical fitting pool.")
    if not vault["Year"].isin(TEST_YEARS).all():
        raise AssertionError("A non vault year entered the external test pool.")

    return historical, vault, ignored


def rebuild_modeling_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild entity and row identifiers after excluding non framework years."""
    out = df.copy().reset_index(drop=True)
    species_values = sorted(out["Species"].astype(str).unique().tolist())
    family_values = sorted(out["Family"].astype(str).unique().tolist())
    species_to_idx = {
        value: index for index, value in enumerate(species_values)
    }
    family_to_idx = {
        value: index for index, value in enumerate(family_values)
    }
    out["species_idx"] = (
        out["Species"].astype(str).map(species_to_idx).astype(int)
    )
    out["family_idx"] = (
        out["Family"].astype(str).map(family_to_idx).astype(int)
    )
    out["row_id"] = np.arange(len(out), dtype=np.int64)
    snapshot_values = sorted(out["snapshot_id"].astype(str).unique().tolist())
    snapshot_to_idx = {
        value: index for index, value in enumerate(snapshot_values)
    }
    out["snapshot_idx"] = (
        out["snapshot_id"].astype(str).map(snapshot_to_idx).astype(int)
    )
    return out


def add_residual_columns(df: pd.DataFrame) :
    out = df.copy()
    p = out["p_baseline"].to_numpy(dtype=float)
    p = np.clip(p, EPS, 1.0 - EPS)
    observed = np.clip(out["prop_S"].to_numpy(dtype=float), EPS, 1.0 - EPS)

    out["baseline_logit"] = np.log(p / (1.0 - p))
    out["observed_logit"] = np.log(observed / (1.0 - observed))
    out["residual_prop_S"] = out["prop_S"] - out["p_baseline"]
    out["residual_logit"] = out["observed_logit"] - out["baseline_logit"]
    return out


def build_fold_features(
    train_df: pd.DataFrame,
    external_df: pd.DataFrame,
    *,
    alpha: float,
    beta: float,
    baseline_mode: str,
):
    global_p = smoothed_rate(
        train_df,
        group_cols=None,
        alpha=alpha,
        beta=beta,
    )

    train_features = add_configured_baseline_prediction_leave_one_out(
        train_df=train_df,
        mode=baseline_mode,
        alpha=alpha,
        beta=beta,
        pred_col="p_baseline",
        source_col="baseline_source",
    )

    external_features = add_configured_baseline_prediction(
        train_df=train_df,
        target_df=external_df,
        mode=baseline_mode,
        global_p=global_p,
        alpha=alpha,
        beta=beta,
        pred_col="p_baseline",
        source_col="baseline_source",
    )

    train_features = add_residual_columns(train_features)
    external_features = add_residual_columns(external_features)

    if train_features["p_baseline"].isna().any():
        raise ValueError("Training fold contains missing leave-one-out baseline predictions.")
    if external_features["p_baseline"].isna().any():
        raise ValueError("External fold contains missing train-only baseline predictions.")

    return train_features, external_features, float(global_p)


# -----------------------------------------------------------------------------
# Random-mask training dataset
# -----------------------------------------------------------------------------


class RandomMaskSnapshotDataset(Dataset):
    """One sample per Country-Year snapshot with epoch-dependent disjoint masks."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        context_frac: float,
        min_input_cells: int,
        min_target_cells: int,
        seed: int,
    ) :
        self.df = df.copy()
        self.context_frac = float(context_frac)
        self.min_input_cells = int(min_input_cells)
        self.min_target_cells = int(min_target_cells)
        self.seed = int(seed)
        self.epoch = 0

        self.snapshot_indices: List[np.ndarray] = []
        self.snapshot_ids: List[str] = []

        for snapshot_id, group in self.df.groupby("snapshot_id", sort=True):
            indices = group.index.to_numpy(dtype=np.int64)
            if len(indices) < self.min_input_cells + self.min_target_cells:
                continue
            self.snapshot_ids.append(str(snapshot_id))
            self.snapshot_indices.append(indices)

        if not self.snapshot_indices:
            raise ValueError(
                "No training snapshots have enough cells for disjoint context and target masks."
            )

    def set_epoch(self, epoch: int) :
        self.epoch = int(epoch)

    def __len__(self) :
        return len(self.snapshot_indices)

    def __getitem__(self, index: int) :
        indices = self.snapshot_indices[index]
        n_cells = len(indices)

        # Separate deterministic random stream for each epoch and snapshot.
        rng = np.random.default_rng(
            self.seed + 1_000_003 * self.epoch + 97_409 * int(index)
        )
        permutation = rng.permutation(n_cells)

        n_input = int(round(self.context_frac * n_cells))
        n_input = max(self.min_input_cells, n_input)
        n_input = min(n_input, n_cells - self.min_target_cells)

        input_indices = indices[permutation[:n_input]]
        target_indices = indices[permutation[n_input:]]

        if len(input_indices) < self.min_input_cells:
            raise RuntimeError("Generated fewer input cells than requested.")
        if len(target_indices) < self.min_target_cells:
            raise RuntimeError("Generated fewer target cells than requested.")

        return {
            "snapshot_id": self.snapshot_ids[index],
            "input": self.df.loc[input_indices].copy(),
            "target": self.df.loc[target_indices].copy(),
        }


def _concat_tensor(
    frames: Sequence[pd.DataFrame],
    column: str,
    dtype: torch.dtype,
):
    values = np.concatenate([frame[column].to_numpy() for frame in frames])
    return torch.as_tensor(values, dtype=dtype)


def collate_snapshot_samples(samples: Sequence[Mapping[str, object]]):
    if not samples:
        raise ValueError("Cannot collate an empty snapshot batch.")

    input_frames = [sample["input"] for sample in samples]
    target_frames = [sample["target"] for sample in samples]

    if not all(isinstance(frame, pd.DataFrame) for frame in input_frames + target_frames):
        raise TypeError("Snapshot samples must contain pandas DataFrames.")

    input_frames = [frame for frame in input_frames if isinstance(frame, pd.DataFrame)]
    target_frames = [frame for frame in target_frames if isinstance(frame, pd.DataFrame)]

    input_batch_idx = np.concatenate(
        [np.full(len(frame), i, dtype=np.int64) for i, frame in enumerate(input_frames)]
    )
    target_batch_idx = np.concatenate(
        [np.full(len(frame), i, dtype=np.int64) for i, frame in enumerate(target_frames)]
    )

    batch: Dict[str, object] = {
        "snapshot_ids": [str(sample["snapshot_id"]) for sample in samples],
        "n_snapshots_in_batch": len(samples),
        "input_snapshot_batch_idx": torch.as_tensor(input_batch_idx, dtype=torch.long),
        "target_snapshot_batch_idx": torch.as_tensor(target_batch_idx, dtype=torch.long),
        "input_species_idx": _concat_tensor(input_frames, "species_idx", torch.long),
        "input_family_idx": _concat_tensor(input_frames, "family_idx", torch.long),
        "input_prop_S": _concat_tensor(input_frames, "prop_S", torch.float32),
        "input_n_total": _concat_tensor(input_frames, "n_total", torch.float32),
        "input_p_baseline": _concat_tensor(input_frames, "p_baseline", torch.float32),
        "input_residual_prop_S": _concat_tensor(
            input_frames, "residual_prop_S", torch.float32
        ),
        "target_species_idx": _concat_tensor(target_frames, "species_idx", torch.long),
        "target_family_idx": _concat_tensor(target_frames, "family_idx", torch.long),
        "target_prop_S": _concat_tensor(target_frames, "prop_S", torch.float32),
        "target_n_S": _concat_tensor(target_frames, "n_S", torch.float32),
        "target_n_total": _concat_tensor(target_frames, "n_total", torch.float32),
        "target_p_baseline": _concat_tensor(
            target_frames, "p_baseline", torch.float32
        ),
        "target_baseline_logit": _concat_tensor(
            target_frames, "baseline_logit", torch.float32
        ),
        "target_residual_prop_S": _concat_tensor(
            target_frames, "residual_prop_S", torch.float32
        ),
        "target_row_id": _concat_tensor(target_frames, "row_id", torch.long),
    }
    return batch


def build_training_loader(
    train_features: pd.DataFrame,
    *,
    context_frac: float,
    min_input_cells: int,
    min_target_cells: int,
    seed: int,
    batch_size: int,
    num_workers: int,
) :
    dataset = RandomMaskSnapshotDataset(
        train_features,
        context_frac=context_frac,
        min_input_cells=min_input_cells,
        min_target_cells=min_target_cells,
        seed=seed,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_snapshot_samples,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset


# -----------------------------------------------------------------------------
# Model construction and fixed-epoch training
# -----------------------------------------------------------------------------


def build_model(
    model_name: str,
    *,
    n_species: int,
    n_families: int,
    args: argparse.Namespace,
    device: torch.device,
) :
    common = dict(
        n_species=n_species,
        n_families=n_families,
        entity_emb_dim=args.entity_emb_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        latent_dim=args.latent_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        dropout=args.dropout,
    )

    if model_name == MODEL_RESIDUAL:
        model = SnapshotEncoderResidualModel(**common)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model.to(device)


def train_fixed_epochs(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    dataset: RandomMaskSnapshotDataset,
    *,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)

        if model_name == MODEL_RESIDUAL:
            train_loss = train_one_epoch_residual_snapshot_encoder(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device=device,
            )
            row = {
                "fold": fold,
                "model_name": model_name,
                "epoch": epoch,
                "train_loss_per_test": float(train_loss),
            }

        else:
            raise ValueError(model_name)

        history.append(row)

        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            message = (
                f"Fold {fold} | {model_name} | epoch {epoch:04d}/{args.epochs} | "
                f"loss/test={row['train_loss_per_test']:.6f}"
            )
            if "train_kl_per_snapshot" in row:
                message += f" | KL/snapshot={row['train_kl_per_snapshot']:.6f}"
            print(message)

    return pd.DataFrame(history)


# -----------------------------------------------------------------------------
# Tensor helpers and model inference
# -----------------------------------------------------------------------------


def tensor_long(values: Iterable[object], device: torch.device) :
    return torch.as_tensor(list(values), dtype=torch.long, device=device)


def tensor_float(values: Iterable[object], device: torch.device) :
    return torch.as_tensor(list(values), dtype=torch.float32, device=device)


def _build_vectorized_loo_tensors(
    snapshot: pd.DataFrame,
    target_positions: Sequence[int],
    device: torch.device,
):
    """Create k pseudo-snapshots, each omitting one different target cell."""

    n_cells = len(snapshot)
    if n_cells < 2:
        raise ValueError("Leave-one-out requires at least two cells in the snapshot.")

    all_positions = np.arange(n_cells, dtype=np.int64)

    input_position_blocks: List[np.ndarray] = []
    input_batch_blocks: List[np.ndarray] = []
    for pseudo_snapshot_idx, target_position in enumerate(target_positions):
        context_positions = all_positions[all_positions != int(target_position)]
        input_position_blocks.append(context_positions)
        input_batch_blocks.append(
            np.full(len(context_positions), pseudo_snapshot_idx, dtype=np.int64)
        )

    input_positions = np.concatenate(input_position_blocks)
    input_batch_idx = np.concatenate(input_batch_blocks)
    target_positions_array = np.asarray(target_positions, dtype=np.int64)

    input_rows = snapshot.iloc[input_positions]
    target_rows = snapshot.iloc[target_positions_array]

    return {
        "input_species_idx": tensor_long(input_rows["species_idx"], device),
        "input_family_idx": tensor_long(input_rows["family_idx"], device),
        "input_prop_S": tensor_float(input_rows["prop_S"], device),
        "input_n_total": tensor_float(input_rows["n_total"], device),
        "input_p_baseline": tensor_float(input_rows["p_baseline"], device),
        "input_residual_prop_S": tensor_float(
            input_rows["residual_prop_S"], device
        ),
        "input_snapshot_batch_idx": torch.as_tensor(
            input_batch_idx, dtype=torch.long, device=device
        ),
        "n_snapshots_in_batch": len(target_positions),
        "target_species_idx": tensor_long(target_rows["species_idx"], device),
        "target_family_idx": tensor_long(target_rows["family_idx"], device),
        "target_snapshot_batch_idx": torch.arange(
            len(target_positions), dtype=torch.long, device=device
        ),
        "target_baseline_logit": tensor_float(
            target_rows["baseline_logit"], device
        ),
    }


@torch.no_grad()
def predict_loo_snapshot(
    model: torch.nn.Module,
    model_name: str,
    snapshot: pd.DataFrame,
    *,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
) :
    """Predict every cell from all other cells in the same external snapshot."""

    snapshot = snapshot.sort_values(["species_idx", "family_idx"]).reset_index(drop=True)
    n_cells = len(snapshot)
    if n_cells < 2:
        return pd.DataFrame()

    model.eval()
    total_context_tests = float(snapshot["n_total"].sum())
    output_rows: List[Dict[str, object]] = []

    for start in range(0, n_cells, args.loo_target_batch_size):
        stop = min(start + args.loo_target_batch_size, n_cells)
        target_positions = list(range(start, stop))
        tensors = _build_vectorized_loo_tensors(snapshot, target_positions, device)

        if model_name == MODEL_RESIDUAL:
            final_logits, _delta_logits, _z_snapshot = model(
                input_species_idx=tensors["input_species_idx"],
                input_family_idx=tensors["input_family_idx"],
                input_p_baseline=tensors["input_p_baseline"],
                input_residual_prop_S=tensors["input_residual_prop_S"],
                input_n_total=tensors["input_n_total"],
                input_snapshot_batch_idx=tensors["input_snapshot_batch_idx"],
                n_snapshots_in_batch=tensors["n_snapshots_in_batch"],
                target_species_idx=tensors["target_species_idx"],
                target_family_idx=tensors["target_family_idx"],
                target_snapshot_batch_idx=tensors["target_snapshot_batch_idx"],
                target_baseline_logit=tensors["target_baseline_logit"],
            )
            p_samples = (
                torch.sigmoid(final_logits).unsqueeze(0).detach().cpu().numpy()
            )
            kl_per_target = np.full(len(target_positions), np.nan)

        else:
            raise ValueError(model_name)

        p_mean = p_samples.mean(axis=0)
        p_sd = p_samples.std(axis=0, ddof=0)
        p_q05 = np.quantile(p_samples, 0.05, axis=0)
        p_q95 = np.quantile(p_samples, 0.95, axis=0)
        p_q025 = np.quantile(p_samples, 0.025, axis=0)
        p_q975 = np.quantile(p_samples, 0.975, axis=0)

        for local_idx, target_position in enumerate(target_positions):
            target = snapshot.iloc[target_position]
            output_rows.append(
                {
                    "fold": fold,
                    "evaluation_protocol": EVALUATION_PROTOCOL,
                    "model_name": model_name,
                    "Country": target["Country"],
                    "Year": int(target["Year"]),
                    "Species": target["Species"],
                    "Family": target["Family"],
                    "snapshot_id": target["snapshot_id"],
                    "row_id": int(target["row_id"]),
                    "species_idx": int(target["species_idx"]),
                    "family_idx": int(target["family_idx"]),
                    "n_S": float(target["n_S"]),
                    "n_total": float(target["n_total"]),
                    "prop_S": float(target["prop_S"]),
                    "p_baseline": float(target["p_baseline"]),
                    "baseline_source": target["baseline_source"],
                    "p_pred": float(np.clip(p_mean[local_idx], EPS, 1.0 - EPS)),
                    "posterior_p_sd": float(p_sd[local_idx]),
                    "epistemic_p_q025": float(p_q025[local_idx]),
                    "epistemic_p_q05": float(p_q05[local_idx]),
                    "epistemic_p_q95": float(p_q95[local_idx]),
                    "epistemic_p_q975": float(p_q975[local_idx]),
                    "kl_snapshot": float(kl_per_target[local_idx]),
                    "context_n_cells": n_cells - 1,
                    "context_n_tests": total_context_tests - float(target["n_total"]),
                    "snapshot_n_observed_cells": n_cells,
                    # Kept temporarily for predictive interval construction.
                    "_p_samples": p_samples[:, local_idx].astype(float),
                }
            )

    return pd.DataFrame(output_rows)


@torch.no_grad()
def collect_random_mask_predictions(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    dataset: RandomMaskSnapshotDataset,
    train_features: pd.DataFrame,
    *,
    epoch: int,
    n_variational_samples: int,
    device: torch.device,
) :
    """One leakage-safe train reconstruction pass used only to estimate fold phi."""

    dataset.set_epoch(epoch)
    model.eval()
    rows: List[pd.DataFrame] = []
    lookup = train_features.set_index("row_id", drop=False)

    for batch in loader:
        tensor_batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

        if model_name == MODEL_RESIDUAL:
            final_logits, _, _ = model(
                input_species_idx=tensor_batch["input_species_idx"],
                input_family_idx=tensor_batch["input_family_idx"],
                input_p_baseline=tensor_batch["input_p_baseline"],
                input_residual_prop_S=tensor_batch["input_residual_prop_S"],
                input_n_total=tensor_batch["input_n_total"],
                input_snapshot_batch_idx=tensor_batch["input_snapshot_batch_idx"],
                n_snapshots_in_batch=tensor_batch["n_snapshots_in_batch"],
                target_species_idx=tensor_batch["target_species_idx"],
                target_family_idx=tensor_batch["target_family_idx"],
                target_snapshot_batch_idx=tensor_batch["target_snapshot_batch_idx"],
                target_baseline_logit=tensor_batch["target_baseline_logit"],
            )
            p_pred = torch.sigmoid(final_logits)

        else:
            raise ValueError(model_name)

        row_ids = batch["target_row_id"].detach().cpu().numpy().astype(int)
        target = lookup.loc[row_ids, ["row_id", "n_S", "n_total", "prop_S"]].copy()
        target["p_pred"] = p_pred.detach().cpu().numpy()
        rows.append(target.reset_index(drop=True))

    if not rows:
        raise RuntimeError("No random-mask training predictions were produced for phi fitting.")
    return pd.concat(rows, ignore_index=True)


# -----------------------------------------------------------------------------
# Dispersion and predictive intervals
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PhiInfo:
    raw_phi: float
    phi: float
    rho: float
    source: str


def phi_info_from_fit(fit: Mapping[str, object], source: str) :
    raw_phi = float(fit["log_phi"])
    phi = float(fit["phi"])
    return PhiInfo(
        raw_phi=raw_phi,
        phi=phi,
        rho=float(1.0 / (phi + 1.0)),
        source=source,
    )


def learned_phi_info(model: torch.nn.Module) :
    if not hasattr(model, "log_phi"):
        return None
    raw_phi = float(getattr(model, "log_phi").detach().cpu())
    phi = float((F.softplus(torch.tensor(raw_phi)) + EPS).item())
    return PhiInfo(
        raw_phi=raw_phi,
        phi=phi,
        rho=float(1.0 / (phi + 1.0)),
        source="learned_during_training",
    )


def fit_phi_on_predictions(df: pd.DataFrame, pred_col: str, source: str) :
    fit = fit_global_phi_for_baseline(df=df, pred_col=pred_col)
    return phi_info_from_fit(fit, source=source)


def add_beta_latent_brackets(
    predictions: pd.DataFrame,
    *,
    alpha_col: str,
    beta_col: str,
) -> pd.DataFrame:
    """Add cell level 95 percent latent Beta brackets and latent SD.

    These columns describe the latent susceptibility rate and exclude the
    additional finite sample Binomial variation.
    """

    out = predictions.copy()
    alpha = pd.to_numeric(out[alpha_col], errors="coerce")
    beta = pd.to_numeric(out[beta_col], errors="coerce")
    valid = (
        np.isfinite(alpha.to_numpy(dtype=float))
        & np.isfinite(beta.to_numpy(dtype=float))
        & alpha.gt(0).to_numpy()
        & beta.gt(0).to_numpy()
    )

    bracket_columns = [
        "latent_rate_ci_lo_95",
        "latent_rate_ci_hi_95",
        "latent_rate_ci_width_95",
        "latent_rate_sd",
    ]
    for column in bracket_columns:
        out[column] = np.nan

    if np.any(valid):
        alpha_values = alpha.to_numpy(dtype=float)[valid]
        beta_values = beta.to_numpy(dtype=float)[valid]
        total = alpha_values + beta_values

        lo = beta_distribution.ppf(0.025, alpha_values, beta_values)
        hi = beta_distribution.ppf(0.975, alpha_values, beta_values)
        sd = np.sqrt(
            (alpha_values * beta_values)
            / (total**2 * (total + 1.0))
        )

        out.loc[valid, "latent_rate_ci_lo_95"] = lo
        out.loc[valid, "latent_rate_ci_hi_95"] = hi
        out.loc[valid, "latent_rate_ci_width_95"] = hi - lo
        out.loc[valid, "latent_rate_sd"] = sd

    out["latent_rate_interval_level"] = 0.95
    out["latent_rate_interval_distribution"] = "beta"
    out["latent_rate_interval_includes_binomial_sampling"] = False
    return out


def add_predictive_intervals(
    predictions: pd.DataFrame,
    phi_info: PhiInfo,
    *,
    n_predictive_samples: int,
    seed: int,
) :
    out = predictions.copy()
    rng = np.random.default_rng(seed)

    q025: List[float] = []
    q05: List[float] = []
    q95: List[float] = []
    q975: List[float] = []

    for p_samples_raw, n_total_raw in zip(
        out["_p_samples"].tolist(),
        out["n_total"].tolist(),
    ):
        p_samples = np.asarray(p_samples_raw, dtype=float)
        p_samples = np.clip(p_samples, EPS, 1.0 - EPS)

        sample_indices = rng.integers(
            0,
            len(p_samples),
            size=n_predictive_samples,
        )
        p_draw = p_samples[sample_indices]

        alpha = np.clip(p_draw * phi_info.phi, EPS, None)
        beta = np.clip((1.0 - p_draw) * phi_info.phi, EPS, None)
        theta = rng.beta(alpha, beta)

        n_total = max(1, int(round(float(n_total_raw))))
        count_draw = rng.binomial(n_total, np.clip(theta, EPS, 1.0 - EPS))
        prop_draw = count_draw / float(n_total)

        q025.append(float(np.quantile(prop_draw, 0.025)))
        q05.append(float(np.quantile(prop_draw, 0.05)))
        q95.append(float(np.quantile(prop_draw, 0.95)))
        q975.append(float(np.quantile(prop_draw, 0.975)))

    out["predictive_prop_q025"] = q025
    out["predictive_prop_q05"] = q05
    out["predictive_prop_q95"] = q95
    out["predictive_prop_q975"] = q975

    out["raw_phi_train"] = phi_info.raw_phi
    out["phi_train"] = phi_info.phi
    out["rho_train"] = phi_info.rho
    out["phi_source"] = phi_info.source
    out["bb_alpha_at_mean"] = out["p_pred"] * phi_info.phi
    out["bb_beta_at_mean"] = (1.0 - out["p_pred"]) * phi_info.phi
    out = add_beta_latent_brackets(
        out,
        alpha_col="bb_alpha_at_mean",
        beta_col="bb_beta_at_mean",
    )

    return out.drop(columns=["_p_samples"])


def make_prior_predictions(
    external_features: pd.DataFrame,
    *,
    fold: int,
    phi_info: PhiInfo,
    args: argparse.Namespace,
)  :
    rows = external_features.copy()
    rows["fold"] = fold
    rows["evaluation_protocol"] = EVALUATION_PROTOCOL
    rows["model_name"] = MODEL_PRIOR
    rows["p_pred"] = rows["p_baseline"].clip(EPS, 1.0 - EPS)
    rows["posterior_p_sd"] = 0.0
    rows["epistemic_p_q025"] = rows["p_pred"]
    rows["epistemic_p_q05"] = rows["p_pred"]
    rows["epistemic_p_q95"] = rows["p_pred"]
    rows["epistemic_p_q975"] = rows["p_pred"]
    rows["kl_snapshot"] = np.nan

    snapshot_sizes = rows.groupby("snapshot_id")["row_id"].transform("size")
    snapshot_tests = rows.groupby("snapshot_id")["n_total"].transform("sum")
    rows["snapshot_n_observed_cells"] = snapshot_sizes
    rows["context_n_cells"] = snapshot_sizes - 1
    rows["context_n_tests"] = snapshot_tests - rows["n_total"]
    rows["_p_samples"] = rows["p_pred"].map(lambda value: np.asarray([value]))

    columns = [
        "fold",
        "evaluation_protocol",
        "model_name",
        "Country",
        "Year",
        "Species",
        "Family",
        "snapshot_id",
        "row_id",
        "species_idx",
        "family_idx",
        "n_S",
        "n_total",
        "prop_S",
        "p_baseline",
        "baseline_source",
        "p_pred",
        "posterior_p_sd",
        "epistemic_p_q025",
        "epistemic_p_q05",
        "epistemic_p_q95",
        "epistemic_p_q975",
        "kl_snapshot",
        "context_n_cells",
        "context_n_tests",
        "snapshot_n_observed_cells",
        "_p_samples",
    ]
    rows = rows[columns].copy()

    # The caller restricts this table to the same leave-one-out-evaluable
    # snapshots used by the neural encoders, so all models share one test set.
    return add_predictive_intervals(
        rows,
        phi_info,
        n_predictive_samples=args.n_predictive_samples,
        seed=args.seed + 10_000 * fold,
    )


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def weighted_mean(values: np.ndarray, weights: np.ndarray) :
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def beta_binomial_metrics(
    df: pd.DataFrame,
    *,
    pred_col: str,
    raw_phi: float,
) :
    if df.empty or not np.isfinite(raw_phi):
        return float("nan"), float("nan")

    with torch.no_grad():
        per_test = beta_binomial_nll_from_prob(
            p=df[pred_col].to_numpy(dtype=float),
            n_s=df["n_S"].to_numpy(dtype=float),
            n_total=df["n_total"].to_numpy(dtype=float),
            log_phi=torch.tensor(float(raw_phi), dtype=torch.float32),
            reduction="mean_per_test",
        )
        per_cell = beta_binomial_nll_from_prob(
            p=df[pred_col].to_numpy(dtype=float),
            n_s=df["n_S"].to_numpy(dtype=float),
            n_total=df["n_total"].to_numpy(dtype=float),
            log_phi=torch.tensor(float(raw_phi), dtype=torch.float32),
            reduction="mean_per_cell",
        )
    return float(per_test.item()), float(per_cell.item())


def compute_metrics(df: pd.DataFrame) :
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["prop_S", "p_pred", "n_S", "n_total"]
    )
    clean = clean[clean["n_total"] > 0].copy()

    if clean.empty:
        return {
            "n_cells": 0,
            "n_tests": 0,
            "weighted_mae": np.nan,
            "weighted_rmse": np.nan,
            "sqrt_n_weighted_mae": np.nan,
            "sqrt_n_weighted_rmse": np.nan,
            "n_weighted_mae": np.nan,
            "n_weighted_rmse": np.nan,
            "error_weighting": "sqrt_n_total",
            "unweighted_mae": np.nan,
            "unweighted_rmse": np.nan,
            "mean_signed_error_obs_minus_pred": np.nan,
            "weighted_signed_error_obs_minus_pred": np.nan,
            "n_weighted_signed_error_obs_minus_pred": np.nan,
            "binomial_ce_per_test": np.nan,
            "beta_binomial_nll_per_test": np.nan,
            "beta_binomial_nll_per_cell": np.nan,
            "epistemic_coverage_90": np.nan,
            "epistemic_coverage_95": np.nan,
            "predictive_coverage_90": np.nan,
            "predictive_coverage_95": np.nan,
            "mean_epistemic_interval_width_90": np.nan,
            "mean_epistemic_interval_width_95": np.nan,
            "mean_predictive_interval_width_90": np.nan,
            "mean_predictive_interval_width_95": np.nan,
            "mean_posterior_p_sd": np.nan,
        }

    y = clean["prop_S"].to_numpy(dtype=float)
    p = np.clip(clean["p_pred"].to_numpy(dtype=float), EPS, 1.0 - EPS)
    n_s = clean["n_S"].to_numpy(dtype=float)
    n_total = clean["n_total"].to_numpy(dtype=float)
    signed = y - p
    abs_error = np.abs(signed)
    sq_error = signed**2

    ce = -(n_s * np.log(p) + (n_total - n_s) * np.log1p(-p))
    raw_phi = float(clean["raw_phi_train"].iloc[0])
    bb_per_test, bb_per_cell = beta_binomial_metrics(
        clean,
        pred_col="p_pred",
        raw_phi=raw_phi,
    )

    def coverage(lower: str, upper: str) :
        valid = clean[["prop_S", lower, upper]].dropna()
        if valid.empty:
            return float("nan")
        return float(
            (
                (valid["prop_S"] >= valid[lower])
                & (valid["prop_S"] <= valid[upper])
            ).mean()
        )

    sqrt_n_total = np.sqrt(n_total)

    result: Dict[str, object] = {
        "n_cells": int(len(clean)),
        "n_tests": int(round(float(n_total.sum()))),
        "n_snapshots": int(clean["snapshot_id"].nunique()),
        "n_years": int(clean["Year"].nunique()),
        "weighted_mae": weighted_mean(abs_error, sqrt_n_total),
        "weighted_rmse": float(
            math.sqrt(weighted_mean(sq_error, sqrt_n_total))
        ),
        "sqrt_n_weighted_mae": weighted_mean(abs_error, sqrt_n_total),
        "sqrt_n_weighted_rmse": float(
            math.sqrt(weighted_mean(sq_error, sqrt_n_total))
        ),
        "n_weighted_mae": weighted_mean(abs_error, n_total),
        "n_weighted_rmse": float(math.sqrt(weighted_mean(sq_error, n_total))),
        "error_weighting": "sqrt_n_total",
        "unweighted_mae": float(abs_error.mean()),
        "unweighted_rmse": float(math.sqrt(sq_error.mean())),
        "mean_signed_error_obs_minus_pred": float(signed.mean()),
        "weighted_signed_error_obs_minus_pred": weighted_mean(
            signed, sqrt_n_total
        ),
        "n_weighted_signed_error_obs_minus_pred": weighted_mean(
            signed, n_total
        ),
        "binomial_ce_per_test": float(ce.sum() / n_total.sum()),
        "beta_binomial_nll_per_test": bb_per_test,
        "beta_binomial_nll_per_cell": bb_per_cell,
        "epistemic_coverage_90": coverage("epistemic_p_q05", "epistemic_p_q95"),
        "epistemic_coverage_95": coverage("epistemic_p_q025", "epistemic_p_q975"),
        "predictive_coverage_90": coverage(
            "predictive_prop_q05", "predictive_prop_q95"
        ),
        "predictive_coverage_95": coverage(
            "predictive_prop_q025", "predictive_prop_q975"
        ),
        "mean_epistemic_interval_width_90": float(
            (clean["epistemic_p_q95"] - clean["epistemic_p_q05"]).mean()
        ),
        "mean_epistemic_interval_width_95": float(
            (clean["epistemic_p_q975"] - clean["epistemic_p_q025"]).mean()
        ),
        "mean_predictive_interval_width_90": float(
            (clean["predictive_prop_q95"] - clean["predictive_prop_q05"]).mean()
        ),
        "mean_predictive_interval_width_95": float(
            (clean["predictive_prop_q975"] - clean["predictive_prop_q025"]).mean()
        ),
        "mean_posterior_p_sd": float(clean["posterior_p_sd"].mean()),
        "raw_phi_train": raw_phi,
        "phi_train": float(clean["phi_train"].iloc[0]),
        "rho_train": float(clean["rho_train"].iloc[0]),
        "phi_source": clean["phi_source"].iloc[0],
        "mean_context_n_cells": float(clean["context_n_cells"].mean()),
        "mean_context_n_tests": float(clean["context_n_tests"].mean()),
        "fraction_species_seen_in_train": float(clean["species_seen_in_train"].mean()),
        "fraction_family_seen_in_train": float(clean["family_seen_in_train"].mean()),
        "fraction_both_entities_seen_in_train": float(
            clean["both_entities_seen_in_train"].mean()
        ),
    }
    return result


def add_posthoc_country_phi(
    country_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    min_cells: int,
    min_tests: int,
):
    out = country_metrics.copy()
    group_columns = ["Country", "model_name"]
    if "evaluation_set" in predictions.columns:
        group_columns = ["evaluation_set"] + group_columns

    diagnostics: List[Dict[str, object]] = []
    for keys, group in predictions.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, object] = dict(zip(group_columns, keys))
        row.update(
            {
                "phi_country_posthoc": np.nan,
                "rho_country_posthoc": np.nan,
                "raw_phi_country_posthoc": np.nan,
                "posthoc_phi_status": "insufficient_data",
            }
        )
        if len(group) >= min_cells and float(group["n_total"].sum()) >= min_tests:
            try:
                fit = fit_global_phi_for_baseline(group, pred_col="p_pred")
                phi = float(fit["phi"])
                row.update(
                    {
                        "raw_phi_country_posthoc": float(fit["log_phi"]),
                        "phi_country_posthoc": phi,
                        "rho_country_posthoc": float(1.0 / (phi + 1.0)),
                        "posthoc_phi_status": "estimated",
                    }
                )
            except Exception as exc:
                row["posthoc_phi_status"] = f"fit_failed:{type(exc).__name__}"
        diagnostics.append(row)

    diagnostic_df = pd.DataFrame(diagnostics)
    out = out.merge(diagnostic_df, on=group_columns, how="left")
    out["overdispersion_shift_rho"] = (
        out["rho_country_posthoc"] - out["rho_train"]
    )
    out["log_phi_shift"] = np.log(out["phi_country_posthoc"]) - np.log(
        out["phi_train"]
    )
    return out


def summarize_predictions(predictions: pd.DataFrame):
    country_rows: List[Dict[str, object]] = []
    for (evaluation_set, fold, country, model_name), group in predictions.groupby(
        ["evaluation_set", "fold", "Country", "model_name"], sort=True
    ):
        row = {
            "evaluation_set": evaluation_set,
            "fold": int(fold),
            "Country": country,
            "model_name": model_name,
            "evaluation_protocol": EVALUATION_PROTOCOL,
        }
        row.update(compute_metrics(group))
        country_rows.append(row)
    country_metrics = pd.DataFrame(country_rows)

    fold_rows: List[Dict[str, object]] = []
    for (evaluation_set, fold, model_name), group in predictions.groupby(
        ["evaluation_set", "fold", "model_name"], sort=True
    ):
        row = {
            "evaluation_set": evaluation_set,
            "fold": int(fold),
            "model_name": model_name,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "n_external_countries": int(group["Country"].nunique()),
        }
        row.update(compute_metrics(group))
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)

    global_rows: List[Dict[str, object]] = []
    for (evaluation_set, model_name), group in predictions.groupby(
        ["evaluation_set", "model_name"], sort=True
    ):
        row = {
            "evaluation_set": evaluation_set,
            "model_name": model_name,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "summary_type": "micro_all_cells",
            "n_external_countries": int(group["Country"].nunique()),
        }
        row.update(compute_metrics(group))
        global_rows.append(row)

    global_metrics = pd.DataFrame(global_rows)

    macro_columns = [
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
        "predictive_coverage_90",
        "predictive_coverage_95",
        "mean_predictive_interval_width_90",
        "mean_predictive_interval_width_95",
        "mean_posterior_p_sd",
    ]
    macro_rows: List[Dict[str, object]] = []
    for (evaluation_set, model_name), group in country_metrics.groupby(
        ["evaluation_set", "model_name"], sort=True
    ):
        row = {
            "evaluation_set": evaluation_set,
            "model_name": model_name,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "summary_type": "macro_mean_across_countries",
            "n_external_countries": int(group["Country"].nunique()),
        }
        for column in macro_columns:
            row[column] = float(group[column].mean())
        macro_rows.append(row)

    global_metrics = pd.concat(
        [global_metrics, pd.DataFrame(macro_rows)], ignore_index=True, sort=False
    )
    return country_metrics, fold_metrics, global_metrics


def summarize_predictions_by_year(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    country_year_rows: List[Dict[str, object]] = []
    for (evaluation_set, fold, country, year, model_name), group in predictions.groupby(
        ["evaluation_set", "fold", "Country", "Year", "model_name"], sort=True
    ):
        row = {
            "evaluation_set": evaluation_set,
            "fold": int(fold),
            "Country": country,
            "Year": int(year),
            "model_name": model_name,
            "evaluation_protocol": EVALUATION_PROTOCOL,
        }
        row.update(compute_metrics(group))
        country_year_rows.append(row)

    fold_year_rows: List[Dict[str, object]] = []
    for (evaluation_set, fold, year, model_name), group in predictions.groupby(
        ["evaluation_set", "fold", "Year", "model_name"], sort=True
    ):
        row = {
            "evaluation_set": evaluation_set,
            "fold": int(fold),
            "Year": int(year),
            "model_name": model_name,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "n_external_countries": int(group["Country"].nunique()),
        }
        row.update(compute_metrics(group))
        fold_year_rows.append(row)

    global_year_rows: List[Dict[str, object]] = []
    for (evaluation_set, year, model_name), group in predictions.groupby(
        ["evaluation_set", "Year", "model_name"], sort=True
    ):
        row = {
            "evaluation_set": evaluation_set,
            "Year": int(year),
            "model_name": model_name,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "summary_type": "micro_all_cells_within_year",
            "n_external_countries": int(group["Country"].nunique()),
        }
        row.update(compute_metrics(group))
        global_year_rows.append(row)

    return (
        pd.DataFrame(country_year_rows),
        pd.DataFrame(fold_year_rows),
        pd.DataFrame(global_year_rows),
    )


def summarize_fold_mean_std(
    fold_metrics: pd.DataFrame,
    *,
    expected_folds: int,
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
        "beta_binomial_nll_per_cell",
        "binomial_ce_per_test",
        "predictive_coverage_90",
        "predictive_coverage_95",
        "mean_predictive_interval_width_90",
        "mean_predictive_interval_width_95",
        "mean_posterior_p_sd",
    ]
    rows: List[Dict[str, object]] = []
    for (evaluation_set, model_name), group in fold_metrics.groupby(
        ["evaluation_set", "model_name"], sort=True
    ):
        observed_folds = sorted(group["fold"].astype(int).unique().tolist())
        if len(observed_folds) != expected_folds:
            raise RuntimeError(
                f"{evaluation_set}, {model_name} has folds {observed_folds}, "
                f"expected {expected_folds} folds."
            )
        row: Dict[str, object] = {
            "evaluation_set": evaluation_set,
            "model_name": model_name,
            "n_folds": int(len(observed_folds)),
            "folds": json.dumps(observed_folds),
            "n_external_countries_total": int(group["n_external_countries"].sum()),
            "n_cells_total": int(group["n_cells"].sum()),
            "n_tests_total": int(group["n_tests"].sum()),
            "uncertainty_definition": "sample_standard_deviation_across_five_folds",
        }
        for column in metric_columns:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if len(values) != expected_folds:
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


# -----------------------------------------------------------------------------
# Colleague-compatible country folds
# -----------------------------------------------------------------------------


def _ordered_countries(df: pd.DataFrame) -> list[str]:
    if "country_input_order" in df.columns:
        table = (
            df[["Country", "country_input_order"]]
            .drop_duplicates("Country")
            .sort_values("country_input_order")
        )
        return table["Country"].astype(str).tolist()
    return df["Country"].astype(str).drop_duplicates().tolist()


def _normalise_loaded_fold_numbers(values: pd.Series, n_folds: int) -> pd.Series:
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


def build_colleague_fold_assignment(
    observed: pd.DataFrame,
    *,
    n_folds: int,
    random_state: int,
    folds_path: Optional[Path],
) -> pd.DataFrame:
    ordered_countries = _ordered_countries(observed)
    expected_countries = set(ordered_countries)

    if folds_path is not None:
        folds_path = Path(folds_path)
        if not folds_path.exists():
            raise FileNotFoundError(f"Country fold file not found: {folds_path}")

        if folds_path.suffix.lower() == ".json":
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
            raise ValueError("The supplied fold file contains no country assignments.")
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
        assignment["fold"] = _normalise_loaded_fold_numbers(
            assignment["loaded_fold"], n_folds
        )
        assignment["colleague_fold"] = assignment["fold"] - 1
        assignment["fold_source"] = str(folds_path)
        return assignment[
            ["Country", "fold", "colleague_fold", "fold_source"]
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
                    "Country": str(country),
                    "fold": int(colleague_fold + 1),
                    "colleague_fold": int(colleague_fold),
                    "fold_source": (
                        "reproduced_KFold_on_country_first_occurrence_order"
                    ),
                }
            )
    assignment = pd.DataFrame(rows)
    if set(assignment["Country"]) != expected_countries:
        raise AssertionError("Not all countries were assigned to a fold.")
    return assignment.sort_values(["fold", "Country"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Fold orchestration
# -----------------------------------------------------------------------------


def save_checkpoint(
    model: torch.nn.Module,
    *,
    model_name: str,
    fold: int,
    train_countries: Sequence[str],
    external_countries: Sequence[str],
    args: argparse.Namespace,
    n_species: int,
    n_families: int,
    output_path: Path,
) -> None:
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_state_dict": state,
            "model_name": model_name,
            "fold": fold,
            "args": vars(args),
            "n_species": n_species,
            "n_families": n_families,
            "train_countries": list(train_countries),
            "external_countries": list(external_countries),
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "training_data_max_year": int(TRAIN_MAX_YEAR),
            "historical_external_test_year_rule": "Year <= 2022",
            "temporal_vault_test_years": list(TEST_YEARS),
            "one_training_run_per_fold": True,
            "fixed_epochs": args.epochs,
            "train_max_year": TRAIN_MAX_YEAR,
            "test_years": list(TEST_YEARS),
            "training_outcomes_from_test_years_used": False,
        },
        output_path,
    )


def add_training_support_flags(
    predictions: pd.DataFrame,
    train_df: pd.DataFrame,
)  :
    out = predictions.copy()
    seen_species = set(train_df["Species"].astype(str))
    seen_families = set(train_df["Family"].astype(str))
    seen_pairs = set(
        train_df[["Species", "Family"]].astype(str).itertuples(index=False, name=None)
    )

    out["species_seen_in_train"] = out["Species"].astype(str).isin(seen_species)
    out["family_seen_in_train"] = out["Family"].astype(str).isin(seen_families)
    out["species_family_seen_in_train"] = [
        (str(species), str(family)) in seen_pairs
        for species, family in out[["Species", "Family"]].itertuples(index=False, name=None)
    ]
    out["both_entities_seen_in_train"] = (
        out["species_seen_in_train"] & out["family_seen_in_train"]
    )
    return out


def choose_model_phi(
    model: torch.nn.Module,
    model_name: str,
    *,
    train_loader: DataLoader,
    train_dataset: RandomMaskSnapshotDataset,
    train_features: pd.DataFrame,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
) :
    learned = learned_phi_info(model)
    if learned is not None:
        return learned

    calibration_epoch = args.epochs + 10_000 + fold
    train_predictions = collect_random_mask_predictions(
        model=model,
        model_name=model_name,
        loader=train_loader,
        dataset=train_dataset,
        train_features=train_features,
        epoch=calibration_epoch,
        n_variational_samples=min(args.n_eval_samples, 32),
        device=device,
    )
    return fit_phi_on_predictions(
        train_predictions,
        pred_col="p_pred",
        source="fitted_on_train_random_mask_reconstruction",
    )


def run_fold(
    fold: int,
    train_df: pd.DataFrame,
    external_df: pd.DataFrame,
    *,
    train_country_set: set[str],
    external_country_set: set[str],
    args: argparse.Namespace,
    device: torch.device,
    n_species: int,
    n_families: int,
):
    fold_seed = args.seed + 100_000 * fold
    set_seed(fold_seed)

    train_countries = sorted(map(str, train_country_set))
    external_countries = sorted(map(str, external_country_set))

    if train_country_set & external_country_set:
        raise RuntimeError(
            f"Country leakage detected between fitting and external fold {fold}."
        )
    if train_df.empty:
        raise ValueError(f"Fold {fold} has no historical training observations.")
    if external_df.empty:
        raise ValueError(f"Fold {fold} has no external observations.")
    if train_df["Year"].gt(TRAIN_MAX_YEAR).any():
        bad_years = sorted(train_df.loc[
            train_df["Year"].gt(TRAIN_MAX_YEAR), "Year"
        ].unique().tolist())
        raise AssertionError(
            f"Fold {fold} training contains forbidden years: {bad_years}"
        )
    allowed_external_year = (
        external_df["Year"].le(TRAIN_MAX_YEAR)
        | external_df["Year"].isin(TEST_YEARS)
    )
    if not allowed_external_year.all():
        bad_years = sorted(
            external_df.loc[~allowed_external_year, "Year"].unique().tolist()
        )
        raise AssertionError(
            f"Fold {fold} external test contains unsupported years: {bad_years}"
        )
    historical_external = external_df.loc[
        external_df["Year"].le(TRAIN_MAX_YEAR)
    ]
    vault_external = external_df.loc[
        external_df["Year"].isin(TEST_YEARS)
    ]
    if historical_external.empty:
        raise ValueError(
            f"Fold {fold} has no historical external observations through 2022."
        )
    if vault_external.empty:
        raise ValueError(
            f"Fold {fold} has no external observations in 2023 and 2024."
        )
    missing_fold_test_years = sorted(
        set(TEST_YEARS) - set(vault_external["Year"].unique().tolist())
    )
    if missing_fold_test_years:
        raise ValueError(
            f"Fold {fold} has no external observations for vault years "
            f"{missing_fold_test_years}."
        )
    if not set(train_df["Country"].astype(str)).issubset(train_country_set):
        raise AssertionError("Historical training rows contain an external country.")
    if not set(external_df["Country"].astype(str)).issubset(external_country_set):
        raise AssertionError("Vault rows contain a non external country.")

    print("\n" + "=" * 88)
    print(f"Fold {fold}/{args.n_folds}")
    print(f"Assigned training countries ({len(train_countries)}): {train_countries}")
    print(f"Assigned external countries ({len(external_countries)}): {external_countries}")
    print(
        f"Historical fitting cells through {TRAIN_MAX_YEAR}: {len(train_df)} | "
        f"historical external cells: {len(historical_external)} | "
        f"vault external cells in {TEST_YEARS}: {len(vault_external)}"
    )

    train_features, external_features, global_p = build_fold_features(
        train_df=train_df,
        external_df=external_df,
        alpha=args.alpha,
        beta=args.beta,
        baseline_mode=args.baseline_mode,
    )

    fold_dir = args.output_dir / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    excluded_snapshots = (
        train_features.groupby(["Country", "Year", "snapshot_id"])
        .size()
        .rename("n_observed_cells")
        .reset_index()
    )
    excluded_snapshots = excluded_snapshots.loc[
        excluded_snapshots["n_observed_cells"]
        < args.min_input_cells + args.min_target_cells
    ].copy()
    excluded_snapshots["fold"] = fold
    excluded_snapshots["dataset_role"] = "historical_train"
    excluded_snapshots["reason"] = (
        "insufficient_cells_for_random_mask_training"
    )

    external_snapshot_sizes = (
        external_features.groupby(["Country", "Year", "snapshot_id"])
        .size()
        .rename("n_observed_cells")
        .reset_index()
    )
    external_excluded = external_snapshot_sizes.loc[
        external_snapshot_sizes["n_observed_cells"] < 2
    ].copy()
    external_excluded["fold"] = fold
    external_excluded["evaluation_set"] = np.where(
        external_excluded["Year"].le(TRAIN_MAX_YEAR),
        EVALUATION_SET_HISTORICAL,
        EVALUATION_SET_VAULT,
    )
    external_excluded["dataset_role"] = external_excluded["evaluation_set"]
    external_excluded["reason"] = "insufficient_cells_for_leave_one_out"
    excluded_snapshots = pd.concat(
        [excluded_snapshots, external_excluded],
        ignore_index=True,
        sort=False,
    )

    prior_phi = fit_phi_on_predictions(
        train_features,
        pred_col="p_baseline",
        source="fitted_on_historical_train_leave_one_out_prior",
    )

    external_snapshot_size = external_features.groupby("snapshot_id")[
        "row_id"
    ].transform("size")
    external_loo_evaluable = external_features.loc[
        external_snapshot_size >= 2
    ].copy()
    if external_loo_evaluable.empty:
        raise RuntimeError(
            f"Fold {fold} has no leave one out evaluable external snapshots."
        )

    prior_predictions = make_prior_predictions(
        external_loo_evaluable,
        fold=fold,
        phi_info=prior_phi,
        args=args,
    )
    prior_predictions = add_training_support_flags(prior_predictions, train_df)

    prediction_frames = [prior_predictions]
    history_frames: List[pd.DataFrame] = []
    phi_rows: List[Dict[str, object]] = [
        {
            "fold": fold,
            "model_name": MODEL_PRIOR,
            "raw_phi_train": prior_phi.raw_phi,
            "phi_train": prior_phi.phi,
            "rho_train": prior_phi.rho,
            "phi_source": prior_phi.source,
            "train_max_year": TRAIN_MAX_YEAR,
            "historical_test_year_rule": "Year <= 2022",
            "test_years": json.dumps(list(TEST_YEARS)),
        }
    ]

    model_name = MODEL_ALIASES["residual"]
    set_seed(fold_seed + 1_000 * (1 + list(MODEL_ALIASES).index("residual")))

    train_loader, train_dataset = build_training_loader(
        train_features,
        context_frac=args.train_context_frac,
        min_input_cells=args.min_input_cells,
        min_target_cells=args.min_target_cells,
        seed=fold_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(
        model_name,
        n_species=n_species,
        n_families=n_families,
        args=args,
        device=device,
    )
    print(
        f"\nTraining {model_name} for exactly {args.epochs} epochs "
        f"using years <= {TRAIN_MAX_YEAR}."
    )

    history = train_fixed_epochs(
        model=model,
        model_name=model_name,
        loader=train_loader,
        dataset=train_dataset,
        fold=fold,
        args=args,
        device=device,
    )
    history["dataset_role"] = "historical_train"
    history["train_max_year"] = TRAIN_MAX_YEAR
    history["n_training_cells"] = int(len(train_features))
    history["n_training_countries"] = int(
        train_features["Country"].nunique()
    )
    history_frames.append(history)

    phi_info = choose_model_phi(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        train_dataset=train_dataset,
        train_features=train_features,
        fold=fold,
        args=args,
        device=device,
    )
    phi_rows.append(
        {
            "fold": fold,
            "model_name": model_name,
            "raw_phi_train": phi_info.raw_phi,
            "phi_train": phi_info.phi,
            "rho_train": phi_info.rho,
            "phi_source": phi_info.source,
            "train_max_year": TRAIN_MAX_YEAR,
            "historical_test_year_rule": "Year <= 2022",
            "test_years": json.dumps(list(TEST_YEARS)),
        }
    )

    model_prediction_frames: List[pd.DataFrame] = []
    grouped_external = external_features.groupby(
        ["Country", "Year", "snapshot_id"], sort=True
    )
    for (_country, _year, _snapshot_id), snapshot in grouped_external:
        if len(snapshot) < 2:
            continue
        snapshot_predictions = predict_loo_snapshot(
            model=model,
            model_name=model_name,
            snapshot=snapshot,
            fold=fold,
            args=args,
            device=device,
        )
        if not snapshot_predictions.empty:
            model_prediction_frames.append(snapshot_predictions)

    if not model_prediction_frames:
        raise RuntimeError(
            f"No external predictions produced for {model_name}, fold {fold}."
        )

    model_predictions = pd.concat(model_prediction_frames, ignore_index=True)
    model_predictions = add_predictive_intervals(
        model_predictions,
        phi_info,
        n_predictive_samples=args.n_predictive_samples,
        seed=fold_seed + 17_000 + len(prediction_frames),
    )
    model_predictions = add_training_support_flags(model_predictions, train_df)
    prediction_frames.append(model_predictions)

    if args.save_fold_models:
        save_checkpoint(
            model,
            model_name=model_name,
            fold=fold,
            train_countries=train_countries,
            external_countries=external_countries,
            args=args,
            n_species=n_species,
            n_families=n_families,
            output_path=fold_dir / f"{model_name}_encoder_model.pt",
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fold_predictions = pd.concat(
        prediction_frames, ignore_index=True, sort=False
    )
    fold_predictions["colleague_fold"] = int(fold - 1)
    fold_predictions["global_train_prior"] = global_p
    fold_predictions["evaluation_set"] = np.where(
        fold_predictions["Year"].le(TRAIN_MAX_YEAR),
        EVALUATION_SET_HISTORICAL,
        EVALUATION_SET_VAULT,
    )
    fold_predictions["dataset_role"] = fold_predictions["evaluation_set"]
    fold_predictions["train_max_year"] = TRAIN_MAX_YEAR
    fold_predictions["test_year"] = fold_predictions["Year"].astype(int)
    fold_predictions["test_years_configured"] = json.dumps(list(TEST_YEARS))
    fold_predictions["target_outcome_used_for_training"] = False
    fold_predictions["country_seen_in_parameter_fitting"] = False
    fold_predictions["same_frozen_model_used_for_both_tests"] = True

    if not (
        fold_predictions["Year"].le(TRAIN_MAX_YEAR)
        | fold_predictions["Year"].isin(TEST_YEARS)
    ).all():
        raise AssertionError("A prediction was produced outside the two test sets.")
    observed_sets = set(fold_predictions["evaluation_set"].unique().tolist())
    expected_sets = {EVALUATION_SET_HISTORICAL, EVALUATION_SET_VAULT}
    if observed_sets != expected_sets:
        raise RuntimeError(
            f"Fold {fold} produced evaluation sets {observed_sets}, "
            f"expected {expected_sets}."
        )
    missing_prediction_years = sorted(
        set(TEST_YEARS)
        - set(
            fold_predictions.loc[
                fold_predictions["evaluation_set"].eq(EVALUATION_SET_VAULT),
                "Year",
            ].unique().tolist()
        )
    )
    if missing_prediction_years:
        raise RuntimeError(
            f"Fold {fold} produced no evaluable vault predictions for years "
            f"{missing_prediction_years}."
        )

    fold_history = (
        pd.concat(history_frames, ignore_index=True, sort=False)
        if history_frames
        else pd.DataFrame()
    )
    fold_phi = pd.DataFrame(phi_rows)
    return fold_predictions, fold_history, fold_phi, excluded_snapshots


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)

    observed = standardize_observed_data(args.input_path)
    historical_pool, vault_pool, ignored_year_rows = split_fixed_temporal_vault(
        observed
    )
    modeling_observed = rebuild_modeling_identifiers(
        pd.concat([historical_pool, vault_pool], ignore_index=True, sort=False)
    )
    historical_pool = modeling_observed.loc[
        modeling_observed["Year"].le(TRAIN_MAX_YEAR)
    ].copy()
    vault_pool = modeling_observed.loc[
        modeling_observed["Year"].isin(TEST_YEARS)
    ].copy()
    countries = sorted(modeling_observed["Country"].astype(str).unique().tolist())

    n_species = int(modeling_observed["species_idx"].max()) + 1
    n_families = int(modeling_observed["family_idx"].max()) + 1

    print("Country generalization completion benchmark")
    print("Evaluation protocol:", EVALUATION_PROTOCOL)
    print("Device:", device)
    print("Historical fitting cutoff:", TRAIN_MAX_YEAR)
    print("Historical external test rule: Year <= 2022")
    print("Temporal vault years:", TEST_YEARS)
    print("Historical cells:", len(historical_pool))
    print("Vault cells:", len(vault_pool))
    print("Ignored cells from other years:", len(ignored_year_rows))
    print("Countries:", len(countries))
    print("Species:", n_species, "Families:", n_families)
    print("Fixed epochs:", args.epochs)
    print("Models:", [MODEL_PRIOR] + [MODEL_ALIASES[name] for name in args.models])

    fold_assignment = build_colleague_fold_assignment(
        modeling_observed,
        n_folds=args.n_folds,
        random_state=args.fold_random_state,
        folds_path=args.country_folds_path,
    )
    if fold_assignment["fold"].nunique() != 5:
        raise RuntimeError("The supplied GNN fold file does not contain five folds.")

    assignment_rows: List[Dict[str, object]] = []
    fold_predictions: List[pd.DataFrame] = []
    fold_histories: List[pd.DataFrame] = []
    fold_phi_rows: List[pd.DataFrame] = []
    excluded_frames: List[pd.DataFrame] = []

    all_country_set = set(modeling_observed["Country"].astype(str))
    for fold in sorted(fold_assignment["fold"].unique().tolist()):
        external_country_set = set(
            fold_assignment.loc[
                fold_assignment["fold"].eq(fold), "Country"
            ].astype(str)
        )
        train_country_set = all_country_set - external_country_set

        train_df = historical_pool.loc[
            historical_pool["Country"].astype(str).isin(train_country_set)
        ].copy()
        historical_external_df = historical_pool.loc[
            historical_pool["Country"].astype(str).isin(external_country_set)
        ].copy()
        vault_external_df = vault_pool.loc[
            vault_pool["Country"].astype(str).isin(external_country_set)
        ].copy()
        external_df = pd.concat(
            [historical_external_df, vault_external_df],
            ignore_index=True,
            sort=False,
        )

        if not train_country_set.isdisjoint(external_country_set):
            raise RuntimeError(f"Country leakage detected in fold {fold}.")
        if train_df["Year"].gt(TRAIN_MAX_YEAR).any():
            raise RuntimeError(f"Post 2022 training row detected in fold {fold}.")
        if not historical_external_df["Year"].le(TRAIN_MAX_YEAR).all():
            raise RuntimeError(f"Invalid historical external row in fold {fold}.")
        if not vault_external_df["Year"].isin(TEST_YEARS).all():
            raise RuntimeError(f"Invalid vault external row in fold {fold}.")

        for country in sorted(external_country_set):
            historical_country = historical_external_df.loc[
                historical_external_df["Country"].astype(str).eq(country)
            ]
            vault_country = vault_external_df.loc[
                vault_external_df["Country"].astype(str).eq(country)
            ]
            fold_source = fold_assignment.loc[
                fold_assignment["Country"].eq(country), "fold_source"
            ].iloc[0]
            assignment_rows.append(
                {
                    "Country": country,
                    "external_fold": int(fold),
                    "colleague_fold": int(fold - 1),
                    "fold_source": fold_source,
                    "train_max_year": TRAIN_MAX_YEAR,
                    "historical_test_year_rule": "Year <= 2022",
                    "vault_test_years": json.dumps(list(TEST_YEARS)),
                    "n_historical_external_cells": int(len(historical_country)),
                    "n_historical_external_tests": int(
                        round(float(historical_country["n_total"].sum()))
                    ),
                    "n_historical_external_years": int(
                        historical_country["Year"].nunique()
                    ),
                    "n_historical_external_snapshots": int(
                        historical_country["snapshot_id"].nunique()
                    ),
                    "n_vault_cells": int(len(vault_country)),
                    "n_vault_tests": int(
                        round(float(vault_country["n_total"].sum()))
                    ),
                    "n_vault_years": int(vault_country["Year"].nunique()),
                    "n_vault_snapshots": int(vault_country["snapshot_id"].nunique()),
                    "has_2023_test_data": bool(vault_country["Year"].eq(2023).any()),
                    "has_2024_test_data": bool(vault_country["Year"].eq(2024).any()),
                }
            )

        predictions_fold, history_fold, phi_fold, excluded_fold = run_fold(
            fold=fold,
            train_df=train_df,
            external_df=external_df,
            train_country_set=train_country_set,
            external_country_set=external_country_set,
            args=args,
            device=device,
            n_species=n_species,
            n_families=n_families,
        )
        fold_predictions.append(predictions_fold)
        fold_histories.append(history_fold)
        fold_phi_rows.append(phi_fold)
        excluded_frames.append(excluded_fold)

    predictions = pd.concat(fold_predictions, ignore_index=True, sort=False)
    history = pd.concat(fold_histories, ignore_index=True, sort=False)
    phi_table = pd.concat(fold_phi_rows, ignore_index=True, sort=False)
    excluded = pd.concat(excluded_frames, ignore_index=True, sort=False)
    assignment = pd.DataFrame(assignment_rows).sort_values("Country")

    assignment_counts = assignment["Country"].value_counts()
    if not assignment_counts.eq(1).all() or set(assignment["Country"]) != set(countries):
        raise RuntimeError(
            "Invalid country fold assignment: every country must appear exactly "
            "once as external."
        )
    if predictions["target_outcome_used_for_training"].fillna(False).any():
        raise RuntimeError("An external target outcome was marked as used for training.")
    if predictions["country_seen_in_parameter_fitting"].fillna(True).any():
        raise RuntimeError("An external country was marked as seen in parameter fitting.")

    fold_check = predictions[["Country", "fold"]].drop_duplicates().merge(
        assignment[["Country", "external_fold"]],
        on="Country",
        how="left",
        validate="many_to_one",
    )
    if fold_check["external_fold"].isna().any() or not fold_check["fold"].eq(
        fold_check["external_fold"]
    ).all():
        raise RuntimeError(
            "At least one prediction was produced by the wrong country fold model."
        )

    predictions["signed_error_obs_minus_pred"] = (
        predictions["prop_S"] - predictions["p_pred"]
    )
    predictions["abs_error"] = predictions["signed_error_obs_minus_pred"].abs()
    predictions["sq_error"] = predictions["signed_error_obs_minus_pred"] ** 2
    predictions["predictive_interval_width_90"] = (
        predictions["predictive_prop_q95"] - predictions["predictive_prop_q05"]
    )
    predictions["predictive_interval_width_95"] = (
        predictions["predictive_prop_q975"] - predictions["predictive_prop_q025"]
    )

    country_metrics, fold_metrics, pooled_summary = summarize_predictions(predictions)
    country_year_metrics, fold_year_metrics, year_summary = (
        summarize_predictions_by_year(predictions)
    )
    fold_mean_std = summarize_fold_mean_std(
        fold_metrics,
        expected_folds=args.n_folds,
    )

    country_metrics = add_posthoc_country_phi(
        country_metrics,
        predictions,
        min_cells=args.min_country_cells_for_posthoc_phi,
        min_tests=args.min_country_tests_for_posthoc_phi,
    )
    country_metrics = country_metrics.merge(
        assignment,
        left_on=["Country", "fold"],
        right_on=["Country", "external_fold"],
        how="left",
        suffixes=("", "_assignment"),
    )
    country_year_metrics = country_year_metrics.merge(
        assignment,
        left_on=["Country", "fold"],
        right_on=["Country", "external_fold"],
        how="left",
        suffixes=("", "_assignment"),
    )

    historical_predictions = predictions.loc[
        predictions["evaluation_set"].eq(EVALUATION_SET_HISTORICAL)
    ].copy()
    vault_predictions = predictions.loc[
        predictions["evaluation_set"].eq(EVALUATION_SET_VAULT)
    ].copy()
    historical_fold_metrics = fold_metrics.loc[
        fold_metrics["evaluation_set"].eq(EVALUATION_SET_HISTORICAL)
    ].copy()
    vault_fold_metrics = fold_metrics.loc[
        fold_metrics["evaluation_set"].eq(EVALUATION_SET_VAULT)
    ].copy()

    output_paths = {
        "predictions_all": args.output_dir / "country_generalization_leave_one_out_predictions.csv",
        "predictions_historical": args.output_dir / "country_generalization_historical_leave_one_out_predictions.csv",
        "predictions_vault": args.output_dir / "country_generalization_vault_leave_one_out_predictions.csv",
        "metrics_by_country_year": args.output_dir / "country_generalization_metrics_by_country_year.csv",
        "metrics_by_country": args.output_dir / "country_generalization_metrics_by_country.csv",
        "metrics_by_fold_year": args.output_dir / "country_generalization_metrics_by_fold_year.csv",
        "metrics_by_fold": args.output_dir / "country_generalization_metrics_by_fold.csv",
        "metrics_mean_std": args.output_dir / "country_generalization_metrics_mean_std_across_folds.csv",
        "historical_metrics_by_fold": args.output_dir / "country_generalization_historical_metrics_by_fold.csv",
        "vault_metrics_by_fold": args.output_dir / "country_generalization_vault_metrics_by_fold.csv",
        "metrics_by_year": args.output_dir / "country_generalization_metrics_by_year.csv",
        "pooled_summary": args.output_dir / "country_generalization_pooled_summary.csv",
        "fold_assignment": args.output_dir / "country_generalization_fold_assignment.csv",
        "training_history": args.output_dir / "country_generalization_historical_training_history.csv",
        "phi_train": args.output_dir / "country_generalization_historical_phi_train.csv",
        "excluded_snapshots": args.output_dir / "country_generalization_excluded_snapshots.csv",
        "ignored_year_rows": args.output_dir / "country_generalization_ignored_year_rows.csv",
        "metadata": args.output_dir / "country_generalization_dual_test_metadata.json",
    }

    predictions.to_csv(output_paths["predictions_all"], index=False)
    historical_predictions.to_csv(output_paths["predictions_historical"], index=False)
    vault_predictions.to_csv(output_paths["predictions_vault"], index=False)
    country_year_metrics.to_csv(output_paths["metrics_by_country_year"], index=False)
    country_metrics.to_csv(output_paths["metrics_by_country"], index=False)
    fold_year_metrics.to_csv(output_paths["metrics_by_fold_year"], index=False)
    fold_metrics.to_csv(output_paths["metrics_by_fold"], index=False)
    fold_mean_std.to_csv(output_paths["metrics_mean_std"], index=False)
    historical_fold_metrics.to_csv(output_paths["historical_metrics_by_fold"], index=False)
    vault_fold_metrics.to_csv(output_paths["vault_metrics_by_fold"], index=False)
    year_summary.to_csv(output_paths["metrics_by_year"], index=False)
    pooled_summary.to_csv(output_paths["pooled_summary"], index=False)
    assignment.to_csv(output_paths["fold_assignment"], index=False)
    history.to_csv(output_paths["training_history"], index=False)
    phi_table.to_csv(output_paths["phi_train"], index=False)
    excluded.to_csv(output_paths["excluded_snapshots"], index=False)
    ignored_year_rows.to_csv(output_paths["ignored_year_rows"], index=False)

    metadata = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "input_path": str(args.input_path),
        "train_max_year": TRAIN_MAX_YEAR,
        "historical_external_test_rule": (
            "assigned external countries and Year less than or equal to 2022"
        ),
        "temporal_vault_external_test_rule": (
            "the same assigned external countries and Year in 2023 or 2024"
        ),
        "one_training_run_per_fold": True,
        "same_frozen_model_used_for_both_tests": True,
        "external_outcomes_used_for_training": False,
        "n_folds": args.n_folds,
        "country_folds_path": str(args.country_folds_path),
        "country_fold_protocol": "exact supplied GNN country folds",
        "primary_error_weighting": "sqrt(n_total)",
        "reported_uncertainty": (
            "sample standard deviation of the five fold level metrics"
        ),
        "fold_standard_deviation_ddof": 1,
        "bootstrap_used": False,
        "fixed_epochs": args.epochs,
        "early_stopping": False,
        "validation_set": False,
        "train_context_frac": args.train_context_frac,
        "external_test_context": (
            "all other observed cells in the same external Country Year snapshot"
        ),
        "models": [MODEL_PRIOR] + [MODEL_ALIASES[name] for name in args.models],
        "n_countries": len(countries),
        "n_historical_cells": len(historical_pool),
        "n_vault_cells": len(vault_pool),
        "n_ignored_year_cells": len(ignored_year_rows),
        "outputs": {key: path.name for key, path in output_paths.items()},
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    output_paths["metadata"].write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nSaved dual test country generalization outputs:")
    for path in output_paths.values():
        print(" ", path)

    display_columns = [
        "evaluation_set",
        "model_name",
        "n_folds",
        "weighted_mae_mean_plus_minus_std",
        "weighted_rmse_mean_plus_minus_std",
        "beta_binomial_nll_per_test_mean_plus_minus_std",
    ]
    display_columns = [
        column for column in display_columns if column in fold_mean_std.columns
    ]
    print("\nPrimary five fold mean plus standard deviation summary:")
    print(fold_mean_std[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()