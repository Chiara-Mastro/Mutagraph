import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import math


def beta_binomial_log_prob_from_logits(
    logits_mu,
    n_s,
    n_total,
    log_phi,
    eps: float = 1e-6,
):
    """
    Per-cell beta-binomial log probability from logits.

    This mirrors beta_binomial_nll_from_logits, but returns unreduced log
    probabilities so MC predictive likelihood can use logsumexp.
    """
    mu = torch.sigmoid(logits_mu).clamp(eps, 1.0 - eps)
    phi = torch.nn.functional.softplus(log_phi) + eps

    alpha = mu * phi
    beta = (1.0 - mu) * phi

    k = n_s
    n = n_total

    return (
        torch.lgamma(n + 1.0)
        - torch.lgamma(k + 1.0)
        - torch.lgamma(n - k + 1.0)
        + torch.lgamma(k + alpha)
        + torch.lgamma(n - k + beta)
        - torch.lgamma(n + alpha + beta)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
    )

def beta_binomial_nll_from_prob(
    p,
    n_s,
    n_total,
    log_phi,
    reduction="mean_per_test",
    eps=1e-6,
):
    """
    Beta-binomial negative log-likelihood for fixed predicted probabilities.

    This is used for baselines where the mean prediction p is already fixed,
    for example p_hierarchical.

    The beta-binomial parameters are:

        alpha = p * phi
        beta  = (1 - p) * phi

    where phi controls overdispersion.
    """
    p = torch.as_tensor(p, dtype=torch.float32).clamp(eps, 1.0 - eps)
    n_s = torch.as_tensor(n_s, dtype=torch.float32)
    n_total = torch.as_tensor(n_total, dtype=torch.float32)

    phi = F.softplus(log_phi) + eps

    alpha = p * phi
    beta = (1.0 - p) * phi

    k = n_s
    n = n_total

    log_prob = (
        torch.lgamma(n + 1.0)
        - torch.lgamma(k + 1.0)
        - torch.lgamma(n - k + 1.0)
        + torch.lgamma(k + alpha)
        + torch.lgamma(n - k + beta)
        - torch.lgamma(n + alpha + beta)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
    )

    nll = -log_prob

    if reduction == "sum":
        return nll.sum()

    if reduction == "mean_per_cell":
        return nll.mean()

    if reduction == "mean_per_test":
        return nll.sum() / n_total.sum().clamp_min(1.0)

    raise ValueError(f"Unknown reduction: {reduction}")

def beta_binomial_nll_from_logits(
    logits_mu,
    n_s,
    n_total,
    log_phi,
    reduction="mean_per_test",
    eps=1e-6,
):
    """
    Beta-binomial negative log likelihood, ignoring no constants? No,
    here we include the combinatorial term too.

    n_s ~ BetaBinomial(n_total, alpha, beta)

    alpha = mu * phi
    beta = (1 - mu) * phi
    """
    mu = torch.sigmoid(logits_mu).clamp(eps, 1.0 - eps)
    phi = F.softplus(log_phi) + eps

    alpha = mu * phi
    beta = (1.0 - mu) * phi

    k = n_s
    n = n_total

    log_prob = (
        torch.lgamma(n + 1.0)
        - torch.lgamma(k + 1.0)
        - torch.lgamma(n - k + 1.0)
        + torch.lgamma(k + alpha)
        + torch.lgamma(n - k + beta)
        - torch.lgamma(n + alpha + beta)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
    )

    nll = -log_prob

    if reduction == "mean":
        return nll.mean()

    if reduction == "sum":
        return nll.sum()

    if reduction == "mean_per_test":
        return nll.sum() / n_total.sum().clamp_min(1.0)

    raise ValueError(f"Unknown reduction: {reduction}")

def binomial_nll_from_logits(logits, n_s, n_total, reduction="mean_per_test"):
    """
    Binomial negative log-likelihood from logits.

    We ignore the combinatorial constant log(n choose k), because it does not
    depend on the model prediction and therefore does not affect training.

    Loss per cell:
        - [k * log(p) + (n-k) * log(1-p)]

    Using logits is numerically safer than computing p first.
    """
    log_p = F.logsigmoid(logits)
    log_1_minus_p = F.logsigmoid(-logits)

    nll = -(
        n_s * log_p
        + (n_total - n_s) * log_1_minus_p
    )

    if reduction == "sum":
        return nll.sum()

    if reduction == "mean_per_cell":
        return nll.mean()

    if reduction == "mean_per_test":
        return nll.sum() / n_total.sum().clamp_min(1.0)

    raise ValueError(f"Unknown reduction: {reduction}")

def compute_metrics_from_arrays(prop_s, pred_p, n_s, n_total, eps=1e-6):
    """
    Compute point and probabilistic metrics for susceptibility predictions.
    """
    prop_s = np.asarray(prop_s, dtype=float)
    pred_p = np.asarray(pred_p, dtype=float)
    n_s = np.asarray(n_s, dtype=float)
    n_total = np.asarray(n_total, dtype=float)

    pred_p = np.clip(pred_p, eps, 1.0 - eps)

    weighted_mae = np.average(np.abs(prop_s - pred_p), weights=n_total)
    weighted_rmse = np.sqrt(np.average((prop_s - pred_p) ** 2, weights=n_total))

    unweighted_mae = np.mean(np.abs(prop_s - pred_p))
    unweighted_rmse = np.sqrt(np.mean((prop_s - pred_p) ** 2))

    binomial_ce_per_test = -np.sum(
        n_s * np.log(pred_p)
        + (n_total - n_s) * np.log(1.0 - pred_p)
    ) / np.sum(n_total)

    return {
        "n_cells": int(len(prop_s)),
        "n_tests": int(np.sum(n_total)),
        "weighted_mae": float(weighted_mae),
        "weighted_rmse": float(weighted_rmse),
        "unweighted_mae": float(unweighted_mae),
        "unweighted_rmse": float(unweighted_rmse),
        "binomial_ce_per_test": float(binomial_ce_per_test),
    }

def move_batch_to_device(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
    }

def move_snapshot_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

def forward_snapshot_batch(model, batch):
    logits, z_snapshot = model(
        input_species_idx=batch["input_species_idx"],
        input_family_idx=batch["input_family_idx"],
        input_prop_S=batch["input_prop_S"],
        input_n_total=batch["input_n_total"],
        input_snapshot_batch_idx=batch["input_snapshot_batch_idx"],
        n_snapshots_in_batch=batch["n_snapshots_in_batch"],
        target_species_idx=batch["target_species_idx"],
        target_family_idx=batch["target_family_idx"],
        target_snapshot_batch_idx=batch["target_snapshot_batch_idx"],
    )

    return logits, z_snapshot

def forward_residual_snapshot_batch(model, batch):
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
    )


def forward_temporal_residual_model(
    model,
    batch: dict[str, object],
) :
    return model(
        input_species_idx=batch["input_species_idx"],
        input_family_idx=batch["input_family_idx"],
        input_p_baseline=batch["input_p_baseline"],
        input_residual_prop_S=batch["input_residual_prop_S"],
        input_n_total=batch["input_n_total"],
        input_snapshot_batch_idx=batch[
            "input_snapshot_batch_idx"
        ],
        n_snapshots_in_batch=batch["n_snapshots_in_batch"],
        target_species_idx=batch["target_species_idx"],
        target_family_idx=batch["target_family_idx"],
        target_snapshot_batch_idx=batch[
            "target_snapshot_batch_idx"
        ],
        target_baseline_logit=batch["target_baseline_logit"],
    )


def forward_variational_residual_snapshot_batch(model, batch, sample: bool = True):
    """
    Forward pass for Variational_residual_encoder_model.

    Expected model output:
        final_logits, delta_logits, z_snapshot, mu_snapshot,
        logvar_snapshot, kl_per_snapshot
    """
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
        sample_latent=sample,
    )

def train_one_epoch_variational_residual_snapshot_encoder(
    model,
    loader,
    optimizer,
    device,
    kl_weight: float,
    n_samples: int = 1,
):
    """
    Residual training with a variational KL penalty.

    If n_samples > 1, draw multiple z samples per batch and average the
    beta-binomial NLL before backprop. This directly tests whether the
    single-sample ELBO estimator was injecting too much decoder-gradient noise.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")

    model.train()

    total_loss_weighted = 0.0
    total_nll_weighted = 0.0
    total_kl_weighted = 0.0
    total_tests = 0.0
    total_snapshots = 0.0

    for batch in loader:
        batch = move_snapshot_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        nll_terms = []
        kl_terms = []

        for _ in range(n_samples):
            (
                final_logits,
                _delta_logits,
                _z_snapshot,
                _mu_snapshot,
                _logvar_snapshot,
                kl_per_snapshot,
            ) = forward_variational_residual_snapshot_batch(
                model=model,
                batch=batch,
                sample=True,
            )

            nll = beta_binomial_nll_from_logits(
                logits_mu=final_logits,
                n_s=batch["target_n_S"],
                n_total=batch["target_n_total"],
                log_phi=model.log_phi,
                reduction="mean_per_test",
            )

            nll_terms.append(nll)
            kl_terms.append(kl_per_snapshot.mean())

        mean_nll = torch.stack(nll_terms).mean()
        mean_kl = torch.stack(kl_terms).mean()
        loss = mean_nll + kl_weight * mean_kl

        loss.backward()
        optimizer.step()

        batch_tests = batch["target_n_total"].sum().item()
        batch_snapshots = float(batch["n_snapshots_in_batch"])

        total_loss_weighted += loss.item() * batch_tests
        total_nll_weighted += mean_nll.item() * batch_tests
        total_kl_weighted += mean_kl.item() * batch_snapshots
        total_tests += batch_tests
        total_snapshots += batch_snapshots

    return {
        "loss_per_test": total_loss_weighted / max(total_tests, 1.0),
        "nll_per_test": total_nll_weighted / max(total_tests, 1.0),
        "kl_per_snapshot": total_kl_weighted / max(total_snapshots, 1.0),
        "kl_weight": float(kl_weight),
        "n_train_samples": int(n_samples),
    }

def train_one_epoch_snapshot_encoder(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_tests = 0.0

    for batch in loader:
        batch = move_snapshot_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        logits, _ = forward_snapshot_batch(model, batch)

        loss = beta_binomial_nll_from_logits(
            logits_mu=logits,
            n_s=batch["target_n_S"],
            n_total=batch["target_n_total"],
            log_phi=model.log_phi,
            reduction="mean_per_test",
        )

        loss.backward()
        optimizer.step()

        batch_tests = batch["target_n_total"].sum().item()
        total_loss += loss.item() * batch_tests
        total_tests += batch_tests

    return total_loss / max(total_tests, 1.0)

def train_one_epoch_residual_snapshot_encoder(
    model,
    loader,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    total_tests = 0.0

    for batch in loader:
        batch = move_snapshot_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        final_logits, _, _ = forward_residual_snapshot_batch(
            model,
            batch,
        )

        loss = beta_binomial_nll_from_logits(
            logits_mu=final_logits,
            n_s=batch["target_n_S"],
            n_total=batch["target_n_total"],
            log_phi=model.log_phi,
            reduction="mean_per_test",
        )

        loss.backward()
        optimizer.step()

        batch_tests = batch["target_n_total"].sum().item()
        total_loss += loss.item() * batch_tests
        total_tests += batch_tests

    return total_loss / max(total_tests, 1.0)

def train_one_epoch_temporal_residual(
    model,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
) :
    model.train()
    total_weighted_loss = 0.0
    total_tests = 0.0

    for raw_batch in loader:
        batch = tensor_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)

        final_logits, _delta_logits, _z_snapshot = forward_temporal_residual_model(
            model,
            batch,
        )
        probability = torch.sigmoid(final_logits)

        loss = beta_binomial_nll_from_prob(
            p=probability,
            n_s=batch["target_n_S"],
            n_total=batch["target_n_total"],
            log_phi=model.log_phi,
            reduction="mean_per_test",
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip,
        )
        optimizer.step()

        batch_tests = float(
            batch["target_n_total"].sum().detach().cpu()
        )
        total_weighted_loss += float(loss.detach().cpu()) * batch_tests
        total_tests += batch_tests

    if total_tests <= 0:
        raise RuntimeError("Training loader contained no target tests.")
    return total_weighted_loss / total_tests


def train_one_epoch_shared_dynamics(model, loader, optimizer, device):
    """
    One epoch of (Country, t) -> (Country, t+1) transition training.

    Loss is the beta-binomial NLL of the decoded target-year predictions
    against the observed n_S/n_total in year t+1. There is no KL term here
    -- this model is deterministic (a point estimate of z_hat_{t+1}, not a
    posterior over it).
    """
    model.train()

    total_loss_weighted = 0.0
    total_tests = 0.0

    for batch in loader:
        batch = move_snapshot_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        final_logits, _delta_logits, _z_t, _z_next = model(
            input_species_idx=batch["input_species_idx"],
            input_family_idx=batch["input_family_idx"],
            input_residual_logit=batch["input_residual_logit"],
            input_n_total=batch["input_n_total"],
            input_snapshot_batch_idx=batch["input_snapshot_batch_idx"],
            n_snapshots_in_batch=batch["n_snapshots_in_batch"],
            target_species_idx=batch["target_species_idx"],
            target_family_idx=batch["target_family_idx"],
            target_snapshot_batch_idx=batch["target_snapshot_batch_idx"],
            target_baseline_logit=batch["target_baseline_logit"],
        )

        loss = beta_binomial_nll_from_logits(
            logits_mu=final_logits,
            n_s=batch["target_n_S"],
            n_total=batch["target_n_total"],
            log_phi=model.log_phi,
            reduction="mean_per_test",
        )

        loss.backward()
        optimizer.step()

        batch_tests = batch["target_n_total"].sum().item()
        total_loss_weighted += loss.item() * batch_tests
        total_tests += batch_tests

    return {"loss_per_test": total_loss_weighted / max(total_tests, 1.0)}

@torch.no_grad()
def evaluate_snapshot_encoder(model, loader, device, eps=1e-6):
    model.eval()

    all_logits = []
    all_n_s = []
    all_n_total = []
    all_prop_s = []

    total_loss = 0.0
    total_tests = 0.0

    for batch in loader:
        batch = move_snapshot_batch_to_device(batch, device)

        logits, _ = forward_snapshot_batch(model, batch)

        loss = beta_binomial_nll_from_logits(
            logits_mu=logits,
            n_s=batch["target_n_S"],
            n_total=batch["target_n_total"],
            log_phi=model.log_phi,
            reduction="mean_per_test",
        )

        batch_tests = batch["target_n_total"].sum().item()
        total_loss += loss.item() * batch_tests
        total_tests += batch_tests

        all_logits.append(logits.detach().cpu())
        all_n_s.append(batch["target_n_S"].detach().cpu())
        all_n_total.append(batch["target_n_total"].detach().cpu())
        all_prop_s.append(batch["target_prop_S"].detach().cpu())

    logits = torch.cat(all_logits).numpy()
    n_s = torch.cat(all_n_s).numpy()
    n_total = torch.cat(all_n_total).numpy()
    prop_s = torch.cat(all_prop_s).numpy()

    pred_p = 1.0 / (1.0 + np.exp(-logits))

    metrics = compute_metrics_from_arrays(
        prop_s=prop_s,
        pred_p=pred_p,
        n_s=n_s,
        n_total=n_total,
        eps=eps,
    )

    metrics["loss_per_test"] = total_loss / max(total_tests, 1.0)

    return metrics

def train_one_epoch_base_latent(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_tests = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        logits = model(
            snapshot_idx=batch["snapshot_idx"],
            species_idx=batch["species_idx"],
            family_idx=batch["family_idx"],
        )

        loss = beta_binomial_nll_from_logits(
            logits_mu=logits,
            n_s=batch["n_S"],
            n_total=batch["n_total"],
            log_phi=model.log_phi,
            reduction="mean_per_test",
        )

        loss.backward()
        optimizer.step()

        batch_tests = batch["n_total"].sum().item()
        total_loss += loss.item() * batch_tests
        total_tests += batch_tests

    return {
        "loss_per_test": total_loss / max(total_tests, 1.0)
    }

@torch.no_grad()
def evaluate_base_latent(model, loader, device, eps=1e-6):
    model.eval()

    all_prop_s = []
    all_pred_p = []
    all_n_s = []
    all_n_total = []

    total_loss = 0.0
    total_tests = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        logits = model(
            snapshot_idx=batch["snapshot_idx"],
            species_idx=batch["species_idx"],
            family_idx=batch["family_idx"],
        )

        loss = beta_binomial_nll_from_logits(
            logits_mu=logits,
            n_s=batch["n_S"],
            n_total=batch["n_total"],
            log_phi=model.log_phi,
            reduction="mean_per_test",
        )

        pred_p = torch.sigmoid(logits)

        batch_tests = batch["n_total"].sum().item()
        total_loss += loss.item() * batch_tests
        total_tests += batch_tests

        all_prop_s.append(batch["prop_S"].detach().cpu().numpy())
        all_pred_p.append(pred_p.detach().cpu().numpy())
        all_n_s.append(batch["n_S"].detach().cpu().numpy())
        all_n_total.append(batch["n_total"].detach().cpu().numpy())

    prop_s = np.concatenate(all_prop_s)
    pred_p = np.concatenate(all_pred_p)
    n_s = np.concatenate(all_n_s)
    n_total = np.concatenate(all_n_total)

    metrics = compute_metrics_from_arrays(
        prop_s=prop_s,
        pred_p=pred_p,
        n_s=n_s,
        n_total=n_total,
        eps=eps,
    )

    metrics["loss_per_test"] = total_loss / max(total_tests, 1.0)
    metrics["phi"] = float(torch.nn.functional.softplus(model.log_phi).detach().cpu())
    metrics["log_phi"] = float(model.log_phi.detach().cpu())

    return metrics

@torch.no_grad()
def evaluate_residual_snapshot_encoder(
    model,
    loader,
    device,
    eps=1e-6,
):
    model.eval()

    all_logits = []
    all_n_s = []
    all_n_total = []
    all_prop_s = []

    total_loss = 0.0
    total_tests = 0.0

    for batch in loader:
        batch = move_snapshot_batch_to_device(batch, device)

        final_logits, _, _ = forward_residual_snapshot_batch(
            model,
            batch,
        )

        loss = beta_binomial_nll_from_logits(
            logits_mu=final_logits,
            n_s=batch["target_n_S"],
            n_total=batch["target_n_total"],
            log_phi=model.log_phi,
            reduction="mean_per_test",
        )

        batch_tests = batch["target_n_total"].sum().item()
        total_loss += loss.item() * batch_tests
        total_tests += batch_tests

        all_logits.append(final_logits.detach().cpu())
        all_n_s.append(batch["target_n_S"].detach().cpu())
        all_n_total.append(batch["target_n_total"].detach().cpu())
        all_prop_s.append(batch["target_prop_S"].detach().cpu())

    logits = torch.cat(all_logits).numpy()
    n_s = torch.cat(all_n_s).numpy()
    n_total = torch.cat(all_n_total).numpy()
    prop_s = torch.cat(all_prop_s).numpy()

    pred_p = 1.0 / (1.0 + np.exp(-logits))

    metrics = compute_metrics_from_arrays(
        prop_s=prop_s,
        pred_p=pred_p,
        n_s=n_s,
        n_total=n_total,
        eps=eps,
    )

    metrics["loss_per_test"] = total_loss / max(total_tests, 1.0)

    return metrics

def evaluate_species_family_prior(
    df_split: pd.DataFrame,
    split_name: str,
    log_phi: float | None = None,
):
    pred_p = df_split["p_baseline"].to_numpy(dtype=float)
    prop_s = df_split["prop_S"].to_numpy(dtype=float)
    n_s = df_split["n_S"].to_numpy(dtype=float)
    n_total = df_split["n_total"].to_numpy(dtype=float)

    metrics = compute_metrics_from_arrays(
        prop_s=prop_s,
        pred_p=pred_p,
        n_s=n_s,
        n_total=n_total,
    )

    metrics["split"] = split_name
    metrics["model_name"] = "species_family_prior"

    if log_phi is not None:
        log_phi_tensor = torch.tensor(
            float(log_phi),
            dtype=torch.float32,
        )

        bb_loss = beta_binomial_nll_from_prob(
            p=pred_p,
            n_s=n_s,
            n_total=n_total,
            log_phi=log_phi_tensor,
            reduction="mean_per_test",
        )

        metrics["beta_binomial_nll_per_test"] = float(
            bb_loss.detach().cpu()
        )
        metrics["log_phi"] = float(log_phi)
        metrics["phi"] = float(
            torch.nn.functional.softplus(log_phi_tensor).detach().cpu()
        )

    return metrics

@torch.no_grad()
def evaluate_variational_residual_snapshot_encoder(
    model,
    loader,
    device,
    n_samples: int = 32,
    eps: float = 1e-6,
    point_prediction_mode: str = "mc_mean_prob",
):
    """
    Evaluate the variational residual encoder.

    Probabilistic loss always uses MC-integrated beta-binomial predictive
    likelihood:
        log p(y|x) ~= logmeanexp_k log p(y|z_k)

    Point metrics can be computed three ways:
        mc_mean_prob:
            mean_k sigmoid(logits(z_k)). This is the posterior predictive mean
            probability, but it can shrink point predictions toward 0.5.
        posterior_mean_latent:
            sigmoid(logits(mu)). This tests whether the learned posterior mean
            is point-accurate even if MC probability averaging softens it.
        mc_mean_logit:
            sigmoid(mean_k logits(z_k)). This isolates sigmoid/Jensen effects.
    """
    valid_modes = {"mc_mean_prob", "posterior_mean_latent", "mc_mean_logit"}
    if point_prediction_mode not in valid_modes:
        raise ValueError(
            f"Unknown point_prediction_mode={point_prediction_mode!r}. "
            f"Valid modes: {sorted(valid_modes)}"
        )

    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")

    model.eval()

    all_pred_p_point = []
    all_pred_p_mean = []
    all_pred_p_sd = []
    all_pred_logit_mean = []
    all_pred_p_mu = []
    all_n_s = []
    all_n_total = []
    all_prop_s = []

    total_mc_nll = 0.0
    total_tests = 0.0
    total_kl = 0.0
    total_snapshots = 0.0

    for batch in loader:
        batch = move_snapshot_batch_to_device(batch, device)

        sample_log_probs = []
        sample_pred_p = []
        sample_logits = []
        kl_per_snapshot_for_batch = None

        for _ in range(n_samples):
            (
                final_logits,
                _delta_logits,
                _z_snapshot,
                _mu_snapshot,
                _logvar_snapshot,
                kl_per_snapshot,
            ) = forward_variational_residual_snapshot_batch(
                model=model,
                batch=batch,
                sample=True,
            )

            log_prob = beta_binomial_log_prob_from_logits(
                logits_mu=final_logits,
                n_s=batch["target_n_S"],
                n_total=batch["target_n_total"],
                log_phi=model.log_phi,
                eps=eps,
            )

            sample_log_probs.append(log_prob)
            sample_logits.append(final_logits)
            sample_pred_p.append(torch.sigmoid(final_logits))
            kl_per_snapshot_for_batch = kl_per_snapshot

        log_probs = torch.stack(sample_log_probs, dim=0)
        pred_p_samples = torch.stack(sample_pred_p, dim=0)
        logit_samples = torch.stack(sample_logits, dim=0)

        mc_log_prob = torch.logsumexp(log_probs, dim=0) - math.log(n_samples)
        mc_nll_sum = -mc_log_prob.sum()

        batch_tests = batch["target_n_total"].sum().item()
        total_mc_nll += mc_nll_sum.item()
        total_tests += batch_tests

        if kl_per_snapshot_for_batch is not None:
            total_kl += kl_per_snapshot_for_batch.sum().item()
            total_snapshots += float(batch["n_snapshots_in_batch"])

        pred_p_mean = pred_p_samples.mean(dim=0)
        pred_p_sd = pred_p_samples.std(dim=0, unbiased=False)
        pred_logit_mean = logit_samples.mean(dim=0)
        pred_p_from_mean_logit = torch.sigmoid(pred_logit_mean)

        (
            final_logits_mu,
            _delta_logits_mu,
            _z_snapshot_mu,
            _mu_snapshot,
            _logvar_snapshot,
            _kl_per_snapshot_mu,
        ) = forward_variational_residual_snapshot_batch(
            model=model,
            batch=batch,
            sample=False,
        )
        pred_p_mu = torch.sigmoid(final_logits_mu)

        if point_prediction_mode == "mc_mean_prob":
            pred_p_point = pred_p_mean
        elif point_prediction_mode == "posterior_mean_latent":
            pred_p_point = pred_p_mu
        elif point_prediction_mode == "mc_mean_logit":
            pred_p_point = pred_p_from_mean_logit
        else:
            raise AssertionError("Unreachable point prediction mode.")

        all_pred_p_point.append(pred_p_point.detach().cpu())
        all_pred_p_mean.append(pred_p_mean.detach().cpu())
        all_pred_p_sd.append(pred_p_sd.detach().cpu())
        all_pred_logit_mean.append(pred_logit_mean.detach().cpu())
        all_pred_p_mu.append(pred_p_mu.detach().cpu())
        all_n_s.append(batch["target_n_S"].detach().cpu())
        all_n_total.append(batch["target_n_total"].detach().cpu())
        all_prop_s.append(batch["target_prop_S"].detach().cpu())

    pred_p_point = torch.cat(all_pred_p_point).numpy()
    pred_p_mean = torch.cat(all_pred_p_mean).numpy()
    pred_p_sd = torch.cat(all_pred_p_sd).numpy()
    pred_logit_mean = torch.cat(all_pred_logit_mean).numpy()
    pred_p_mu = torch.cat(all_pred_p_mu).numpy()
    n_s = torch.cat(all_n_s).numpy()
    n_total = torch.cat(all_n_total).numpy()
    prop_s = torch.cat(all_prop_s).numpy()

    metrics = compute_metrics_from_arrays(
        prop_s=prop_s,
        pred_p=pred_p_point,
        n_s=n_s,
        n_total=n_total,
        eps=eps,
    )

    # Always report the integrated MC predictive NLL, regardless of point mode.
    metrics["loss_per_test"] = total_mc_nll / max(total_tests, 1.0)
    metrics["beta_binomial_nll_per_test"] = metrics["loss_per_test"]
    metrics["kl_per_snapshot"] = total_kl / max(total_snapshots, 1.0)
    metrics["posterior_p_sd_mean"] = float(np.mean(pred_p_sd))
    metrics["posterior_p_sd_median"] = float(np.median(pred_p_sd))
    metrics["n_eval_samples"] = int(n_samples)
    metrics["point_prediction_mode"] = point_prediction_mode

    # Extra Jensen/shrinkage diagnostics.
    metrics["mean_abs_mu_minus_mc_prob"] = float(np.mean(np.abs(pred_p_mu - pred_p_mean)))
    metrics["weighted_abs_mu_minus_mc_prob"] = float(
        np.average(np.abs(pred_p_mu - pred_p_mean), weights=n_total)
    )
    metrics["mean_abs_mc_logit_prob_minus_mc_prob"] = float(
        np.mean(np.abs((1.0 / (1.0 + np.exp(-pred_logit_mean))) - pred_p_mean))
    )

    return metrics

@torch.no_grad()
def evaluate_shared_dynamics(model, loader, device, eps: float = 1e-6, return_predictions: bool = False):
    """
    Evaluate the shared-dynamics model's one-step-ahead predictions.

    If return_predictions is True, also returns a long-format DataFrame with
    one row per evaluated target cell (country, t, t_next, species_idx,
    family_idx, n_S, n_total, prop_S, pred_p, signed_residual =
    prop_S - pred_p), which is what
    13_train_shared_dynamics_model.py uses to save anomaly/emergence
    candidates (large negative signed_residual: observed susceptibility far
    BELOW what the model expected).
    """
    model.eval()

    total_nll_weighted = 0.0
    total_tests = 0.0

    all_pred_p, all_n_s, all_n_total, all_prop_s = [], [], [], []
    row_records = []

    for batch in loader:
        batch = move_snapshot_batch_to_device(batch, device)

        final_logits, _delta_logits, _z_t, _z_next = model(
            input_species_idx=batch["input_species_idx"],
            input_family_idx=batch["input_family_idx"],
            input_residual_logit=batch["input_residual_logit"],
            input_n_total=batch["input_n_total"],
            input_snapshot_batch_idx=batch["input_snapshot_batch_idx"],
            n_snapshots_in_batch=batch["n_snapshots_in_batch"],
            target_species_idx=batch["target_species_idx"],
            target_family_idx=batch["target_family_idx"],
            target_snapshot_batch_idx=batch["target_snapshot_batch_idx"],
            target_baseline_logit=batch["target_baseline_logit"],
        )

        nll_sum = beta_binomial_nll_from_logits(
            logits_mu=final_logits,
            n_s=batch["target_n_S"],
            n_total=batch["target_n_total"],
            log_phi=model.log_phi,
            reduction="sum",
        )

        batch_tests = batch["target_n_total"].sum().item()
        total_nll_weighted += nll_sum.item()
        total_tests += batch_tests

        pred_p = torch.sigmoid(final_logits).clamp(eps, 1.0 - eps)

        all_pred_p.append(pred_p.detach().cpu())
        all_n_s.append(batch["target_n_S"].detach().cpu())
        all_n_total.append(batch["target_n_total"].detach().cpu())
        all_prop_s.append(batch["target_prop_S"].detach().cpu())

        if return_predictions:
            target_batch_idx = batch["target_snapshot_batch_idx"].detach().cpu().numpy()
            countries_per_row = [batch["countries"][i] for i in target_batch_idx]
            t_nexts_per_row = [batch["t_nexts"][i] for i in target_batch_idx]
            pred_p_np = pred_p.detach().cpu().numpy()
            n_s_np = batch["target_n_S"].detach().cpu().numpy()
            n_total_np = batch["target_n_total"].detach().cpu().numpy()
            prop_s_np = batch["target_prop_S"].detach().cpu().numpy()
            species_np = batch["target_species_idx"].detach().cpu().numpy()
            family_np = batch["target_family_idx"].detach().cpu().numpy()

            for i in range(len(pred_p_np)):
                row_records.append({
                    "Country": countries_per_row[i],
                    "target_year": t_nexts_per_row[i],
                    "species_idx": int(species_np[i]),
                    "family_idx": int(family_np[i]),
                    "n_S": float(n_s_np[i]),
                    "n_total": float(n_total_np[i]),
                    "prop_S": float(prop_s_np[i]),
                    "pred_p": float(pred_p_np[i]),
                    "signed_residual": float(prop_s_np[i] - pred_p_np[i]),
                })

    pred_p_all = torch.cat(all_pred_p).numpy()
    n_s_all = torch.cat(all_n_s).numpy()
    n_total_all = torch.cat(all_n_total).numpy()
    prop_s_all = torch.cat(all_prop_s).numpy()

    metrics = compute_metrics_from_arrays(
        prop_s=prop_s_all, pred_p=pred_p_all, n_s=n_s_all, n_total=n_total_all, eps=eps,
    )
    metrics["loss_per_test"] = total_nll_weighted / max(total_tests, 1.0)
    metrics["beta_binomial_nll_per_test"] = metrics["loss_per_test"]

    if return_predictions:
        return metrics, pd.DataFrame(row_records)

    return metrics

@torch.no_grad()
def evaluate_temporal_residual_loader(
    model,
    loader,
    device: torch.device,
    return_predictions: bool = False,
) :
    model.eval()

    prediction_parts: list[pd.DataFrame] = []

    for raw_batch in loader:
        batch = tensor_batch_to_device(raw_batch, device)
        final_logits, delta_logits, _z_snapshot = forward_temporal_residual_model(
            model,
            batch,
        )

        probability = torch.sigmoid(final_logits)

        target_batch_index = (
            batch["target_snapshot_batch_idx"]
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )
        countries = np.asarray(
            [
                raw_batch["countries"][index]
                for index in target_batch_index
            ],
            dtype=object,
        )
        input_years = np.asarray(
            [
                raw_batch["input_years"][index]
                for index in target_batch_index
            ],
            dtype=int,
        )
        target_years = np.asarray(
            [
                raw_batch["target_years"][index]
                for index in target_batch_index
            ],
            dtype=int,
        )
        pair_ids = np.asarray(
            [
                raw_batch["pair_ids"][index]
                for index in target_batch_index
            ],
            dtype=object,
        )

        prediction_parts.append(
            pd.DataFrame(
                {
                    "pair_id": pair_ids,
                    "Country": countries,
                    "input_year": input_years,
                    "target_year": target_years,
                    "cell_row_id": (
                        batch["target_cell_row_id"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(int)
                    ),
                    "n_S": (
                        batch["target_n_S"]
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    "n_total": (
                        batch["target_n_total"]
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    "prop_S": (
                        batch["target_prop_S"]
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    "p_pred": probability.detach().cpu().numpy(),
                    "delta_logit": (
                        delta_logits.detach().cpu().numpy()
                    ),
                }
            )
        )

    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    metrics = compute_metrics_from_arrays(
        prop_s=predictions["prop_S"].to_numpy(),
        pred_p=predictions["p_pred"].to_numpy(),
        n_s=predictions["n_S"].to_numpy(),
        n_total=predictions["n_total"].to_numpy(),
    )

    bb_nll = beta_binomial_nll_from_prob(
        p=predictions["p_pred"].to_numpy(),
        n_s=predictions["n_S"].to_numpy(),
        n_total=predictions["n_total"].to_numpy(),
        log_phi=model.log_phi.detach().cpu(),
        reduction="mean_per_test",
    )
    metrics["beta_binomial_nll_per_test"] = float(bb_nll.item())
    metrics["loss_per_test"] = metrics[
        "beta_binomial_nll_per_test"
    ]

    if return_predictions:
        return metrics, predictions
    return metrics, None

def tensor_batch_to_device(
    batch: dict[str, object],
    device: torch.device,
) :
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
