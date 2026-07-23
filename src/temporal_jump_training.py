"""Training and evaluation for frozen residual encoder direction heads."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from src.temporal_residual_jump_model import TemporalResidualJumpHeadsModel


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _forward(
    model: TemporalResidualJumpHeadsModel,
    batch: dict[str, object],
):
    return model(
        input_species_idx=batch["input_species_idx"],
        input_family_idx=batch["input_family_idx"],
        input_p_baseline=batch["input_p_baseline"],
        input_residual_prop_S=batch["input_residual_prop_S"],
        input_n_total=batch["input_n_total"],
        input_snapshot_batch_idx=batch["input_snapshot_batch_idx"],
        n_snapshots_in_batch=batch["n_snapshots_in_batch"],
        target_species_idx=batch["target_species_idx"],
        target_family_idx=batch["target_family_idx"],
        target_snapshot_batch_idx=batch["target_snapshot_batch_idx"],
        target_baseline_logit=batch["target_baseline_logit"],
        target_current_prop_S=batch["target_current_prop_S"],
    )


def estimate_jump_pos_weights(dataset) -> tuple[float, float, dict[str, int]]:
    total = 0
    down_positive = 0
    up_positive = 0
    for item_index in range(len(dataset)):
        item = dataset[item_index]
        mask = item["jump_loss_mask"].bool()
        down = item["jump_down_label"].bool() & mask
        up = item["jump_up_label"].bool() & mask
        total += int(mask.sum().item())
        down_positive += int(down.sum().item())
        up_positive += int(up.sum().item())

    if total < 1:
        raise ValueError("No direction targets are available in the training split.")

    down_negative = total - down_positive
    up_negative = total - up_positive
    down_weight = down_negative / max(down_positive, 1)
    up_weight = up_negative / max(up_positive, 1)
    return (
        float(down_weight),
        float(up_weight),
        {
            "n_jump_targets": int(total),
            "n_down_positive": int(down_positive),
            "n_up_positive": int(up_positive),
        },
    )


def _masked_bce_sum_and_count(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float,
) -> tuple[torch.Tensor, int]:
    mask = mask.bool()
    count = int(mask.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    loss_sum = F.binary_cross_entropy_with_logits(
        logits[mask],
        labels[mask],
        pos_weight=torch.tensor(
            float(pos_weight),
            dtype=logits.dtype,
            device=logits.device,
        ),
        reduction="sum",
    )
    return loss_sum, count


def train_one_epoch_jump_heads(
    *,
    model: TemporalResidualJumpHeadsModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
    down_pos_weight: float,
    up_pos_weight: float,
    down_loss_weight: float,
    up_loss_weight: float,
) -> dict[str, float]:
    model.train()
    down_sum = 0.0
    up_sum = 0.0
    combined_sum = 0.0
    total_targets = 0

    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        _, _, _, down_logit, up_logit = _forward(model, batch)

        down_loss_sum, down_count = _masked_bce_sum_and_count(
            down_logit,
            batch["jump_down_label"],
            batch["jump_loss_mask"],
            down_pos_weight,
        )
        up_loss_sum, up_count = _masked_bce_sum_and_count(
            up_logit,
            batch["jump_up_label"],
            batch["jump_loss_mask"],
            up_pos_weight,
        )
        if down_count != up_count:
            raise RuntimeError("Down and up heads received different target counts.")
        if down_count == 0:
            continue

        down_loss = down_loss_sum / down_count
        up_loss = up_loss_sum / up_count
        combined_loss = (
            float(down_loss_weight) * down_loss
            + float(up_loss_weight) * up_loss
        )
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.head_parameters()),
            max_norm=float(gradient_clip),
        )
        optimizer.step()

        total_targets += down_count
        down_sum += float(down_loss_sum.detach().cpu())
        up_sum += float(up_loss_sum.detach().cpu())
        combined_sum += float(combined_loss.detach().cpu()) * down_count

    if total_targets < 1:
        raise RuntimeError("Training loader contained no direction targets.")
    return {
        "down_bce": down_sum / total_targets,
        "up_bce": up_sum / total_targets,
        "combined_bce": combined_sum / total_targets,
        "n_jump_targets": int(total_targets),
    }


def _safe_binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    valid = np.isfinite(labels) & np.isfinite(scores)
    labels = labels[valid].astype(int)
    scores = scores[valid]
    if len(labels) < 1 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def classification_metrics_from_frame(
    predictions: pd.DataFrame,
) -> dict[str, float]:
    frame = predictions.loc[
        predictions["jump_loss_mask"].fillna(False)
    ].copy()
    if frame.empty:
        return {
            "n_jump_targets": 0,
            "n_down_positive": 0,
            "n_up_positive": 0,
            "down_auroc": np.nan,
            "down_average_precision": np.nan,
            "up_auroc": np.nan,
            "up_average_precision": np.nan,
        }

    down_auroc, down_ap = _safe_binary_metrics(
        frame["jump_down_label"].to_numpy(float),
        frame["down_prob"].to_numpy(float),
    )
    up_auroc, up_ap = _safe_binary_metrics(
        frame["jump_up_label"].to_numpy(float),
        frame["up_prob"].to_numpy(float),
    )
    return {
        "n_jump_targets": int(len(frame)),
        "n_down_positive": int(frame["jump_down_label"].sum()),
        "n_up_positive": int(frame["jump_up_label"].sum()),
        "down_auroc": down_auroc,
        "down_average_precision": down_ap,
        "up_auroc": up_auroc,
        "up_average_precision": up_ap,
    }


def selection_score(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name == "down_average_precision":
        value = metrics.get("down_average_precision", np.nan)
    elif metric_name == "mean_average_precision":
        values = np.asarray(
            [
                metrics.get("down_average_precision", np.nan),
                metrics.get("up_average_precision", np.nan),
            ],
            dtype=float,
        )
        value = np.nanmean(values) if np.isfinite(values).any() else np.nan
    elif metric_name == "mean_auroc":
        values = np.asarray(
            [
                metrics.get("down_auroc", np.nan),
                metrics.get("up_auroc", np.nan),
            ],
            dtype=float,
        )
        value = np.nanmean(values) if np.isfinite(values).any() else np.nan
    elif metric_name == "negative_combined_bce":
        value = -float(metrics.get("combined_bce", np.nan))
    else:
        raise ValueError(f"Unknown checkpoint metric: {metric_name}")

    if not np.isfinite(value):
        fallback = -float(metrics.get("combined_bce", np.nan))
        return fallback if np.isfinite(fallback) else -math.inf
    return float(value)


@torch.no_grad()
def evaluate_jump_heads(
    *,
    model: TemporalResidualJumpHeadsModel,
    loader,
    device: torch.device,
    down_pos_weight: float,
    up_pos_weight: float,
    down_loss_weight: float,
    up_loss_weight: float,
    return_predictions: bool = False,
) -> tuple[dict[str, float], pd.DataFrame | None]:
    model.eval()
    parts: list[pd.DataFrame] = []
    down_sum = 0.0
    up_sum = 0.0
    combined_sum = 0.0
    total_targets = 0

    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        final_logits, delta_logits, _, down_logit, up_logit = _forward(
            model, batch
        )
        down_prob = torch.sigmoid(down_logit)
        up_prob = torch.sigmoid(up_logit)

        down_loss_sum, down_count = _masked_bce_sum_and_count(
            down_logit,
            batch["jump_down_label"],
            batch["jump_loss_mask"],
            down_pos_weight,
        )
        up_loss_sum, up_count = _masked_bce_sum_and_count(
            up_logit,
            batch["jump_up_label"],
            batch["jump_loss_mask"],
            up_pos_weight,
        )
        if down_count != up_count:
            raise RuntimeError("Down and up heads received different target counts.")
        if down_count > 0:
            down_mean = down_loss_sum / down_count
            up_mean = up_loss_sum / up_count
            combined_mean = (
                float(down_loss_weight) * down_mean
                + float(up_loss_weight) * up_mean
            )
            total_targets += down_count
            down_sum += float(down_loss_sum.cpu())
            up_sum += float(up_loss_sum.cpu())
            combined_sum += float(combined_mean.cpu()) * down_count

        target_batch_index = (
            batch["target_snapshot_batch_idx"].detach().cpu().numpy().astype(int)
        )
        countries = np.asarray(
            [raw_batch["countries"][index] for index in target_batch_index],
            dtype=object,
        )
        input_years = np.asarray(
            [raw_batch["input_years"][index] for index in target_batch_index],
            dtype=int,
        )
        target_years = np.asarray(
            [raw_batch["target_years"][index] for index in target_batch_index],
            dtype=int,
        )
        pair_ids = np.asarray(
            [raw_batch["pair_ids"][index] for index in target_batch_index],
            dtype=object,
        )

        def array(name: str):
            return batch[name].detach().cpu().numpy()

        parts.append(
            pd.DataFrame(
                {
                    "pair_id": pair_ids,
                    "Country": countries,
                    "input_year": input_years,
                    "target_year": target_years,
                    "cell_row_id": array("target_cell_row_id").astype(int),
                    "n_S": array("target_n_S"),
                    "n_total": array("target_n_total"),
                    "prop_S": array("target_prop_S"),
                    "p_current": array("target_current_prop_S"),
                    "n_current": array("target_current_n_total"),
                    "current_cell_observed": array(
                        "target_current_observed"
                    ).astype(bool),
                    "jump_well_sampled": array("jump_well_sampled").astype(bool),
                    "jump_loss_mask": array("jump_loss_mask").astype(bool),
                    "jump_observed_delta": array("jump_observed_delta"),
                    "jump_down_label": array("jump_down_label"),
                    "jump_up_label": array("jump_up_label"),
                    "p_pred_frozen": torch.sigmoid(final_logits).cpu().numpy(),
                    "delta_logit_frozen": delta_logits.cpu().numpy(),
                    "down_logit": down_logit.cpu().numpy(),
                    "down_prob": down_prob.cpu().numpy(),
                    "up_logit": up_logit.cpu().numpy(),
                    "up_prob": up_prob.cpu().numpy(),
                }
            )
        )

    if not parts:
        raise RuntimeError("Evaluation loader produced no predictions.")
    predictions = pd.concat(parts, ignore_index=True)
    metrics = classification_metrics_from_frame(predictions)
    metrics["down_bce"] = (
        down_sum / total_targets if total_targets > 0 else np.nan
    )
    metrics["up_bce"] = up_sum / total_targets if total_targets > 0 else np.nan
    metrics["combined_bce"] = (
        combined_sum / total_targets if total_targets > 0 else np.nan
    )

    if return_predictions:
        return metrics, predictions
    return metrics, None
