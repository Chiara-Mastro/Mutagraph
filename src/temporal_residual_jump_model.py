"""Frozen temporal residual encoder with two direction heads.

The pretrained residual encoder remains unchanged. Two independent MLP heads
reuse its snapshot representation and its learned Species and Family embedding
modules, together with the susceptibility observed in the input year.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


COMMON_SPECIES_EMBEDDING_NAMES = (
    "species_embedding",
    "species_embeddings",
    "species_emb",
    "bug_embedding",
    "pathogen_embedding",
)
COMMON_FAMILY_EMBEDDING_NAMES = (
    "family_embedding",
    "family_embeddings",
    "family_emb",
    "drug_embedding",
    "antibiotic_embedding",
)


def _embedding_candidates(
    model: nn.Module,
    expected_rows: int,
) -> list[tuple[str, nn.Embedding]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Embedding)
        and int(module.num_embeddings) == int(expected_rows)
    ]


def _resolve_embedding_name(
    model: nn.Module,
    *,
    role: str,
    expected_rows: int,
    explicit_name: str | None,
    common_names: Iterable[str],
) -> str:
    if explicit_name is not None:
        module = model.get_submodule(explicit_name)
        if not isinstance(module, nn.Embedding):
            raise TypeError(
                f"Module {explicit_name!r} selected for {role} is not an Embedding."
            )
        if int(module.num_embeddings) != int(expected_rows):
            raise ValueError(
                f"Embedding {explicit_name!r} has {module.num_embeddings} rows, "
                f"expected {expected_rows} for {role}."
            )
        return explicit_name

    named_modules = dict(model.named_modules())
    for candidate_name in common_names:
        module = named_modules.get(candidate_name)
        if isinstance(module, nn.Embedding) and int(module.num_embeddings) == int(
            expected_rows
        ):
            return candidate_name

    candidates = _embedding_candidates(model, expected_rows)
    role_candidates = [
        (name, module)
        for name, module in candidates
        if role.lower() in name.lower()
    ]
    if len(role_candidates) == 1:
        return role_candidates[0][0]
    if len(candidates) == 1:
        return candidates[0][0]

    available = [
        {
            "name": name,
            "num_embeddings": int(module.num_embeddings),
            "embedding_dim": int(module.embedding_dim),
        }
        for name, module in model.named_modules()
        if isinstance(module, nn.Embedding)
    ]
    raise ValueError(
        f"Could not identify one unambiguous {role} embedding with "
        f"{expected_rows} rows. Available embeddings: {available}. "
        f"Pass the exact module name explicitly."
    )


class TemporalResidualJumpHeadsModel(nn.Module):
    """Attach two independent direction heads to a residual encoder.

    The heads receive only information available in the input year:

    1. the shared snapshot representation returned by the residual encoder
    2. the residual encoder Species embedding for the target cell
    3. the residual encoder Family embedding for the target cell
    4. the susceptibility observed for that cell in the input year

    The continuous forecast output is returned for reporting, but it is not an
    input to either direction head.
    """

    def __init__(
        self,
        *,
        residual_model: nn.Module,
        n_species: int,
        n_families: int,
        latent_dim: int,
        head_hidden_dim: int = 64,
        head_bottleneck_dim: int = 16,
        dropout: float = 0.10,
        species_embedding_name: str | None = None,
        family_embedding_name: str | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if head_hidden_dim < 1 or head_bottleneck_dim < 1:
            raise ValueError("Direction head widths must be positive.")

        self.residual_model = residual_model
        self.freeze_backbone = bool(freeze_backbone)

        self.species_embedding_name = _resolve_embedding_name(
            residual_model,
            role="species",
            expected_rows=n_species,
            explicit_name=species_embedding_name,
            common_names=COMMON_SPECIES_EMBEDDING_NAMES,
        )
        self.family_embedding_name = _resolve_embedding_name(
            residual_model,
            role="family",
            expected_rows=n_families,
            explicit_name=family_embedding_name,
            common_names=COMMON_FAMILY_EMBEDDING_NAMES,
        )

        species_embedding = self._species_embedding()
        family_embedding = self._family_embedding()
        feature_dim = (
            int(latent_dim)
            + int(species_embedding.embedding_dim)
            + int(family_embedding.embedding_dim)
            + 1
        )

        def make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feature_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, head_bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_bottleneck_dim, 1),
            )

        self.down_head = make_head()
        self.up_head = make_head()
        self.jump_feature_dim = feature_dim

        if self.freeze_backbone:
            self.set_backbone_trainable(False)

    @property
    def log_phi(self) -> torch.Tensor:
        return self.residual_model.log_phi

    def _species_embedding(self) -> nn.Embedding:
        module = self.residual_model.get_submodule(self.species_embedding_name)
        if not isinstance(module, nn.Embedding):
            raise TypeError("Resolved Species module is no longer an Embedding.")
        return module

    def _family_embedding(self) -> nn.Embedding:
        module = self.residual_model.get_submodule(self.family_embedding_name)
        if not isinstance(module, nn.Embedding):
            raise TypeError("Resolved Family module is no longer an Embedding.")
        return module

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.residual_model.parameters():
            parameter.requires_grad = bool(trainable)
        self.freeze_backbone = not bool(trainable)
        if self.freeze_backbone:
            self.residual_model.eval()

    def head_parameters(self):
        yield from self.down_head.parameters()
        yield from self.up_head.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.residual_model.eval()
        return self

    def forward(
        self,
        *,
        input_species_idx: torch.Tensor,
        input_family_idx: torch.Tensor,
        input_p_baseline: torch.Tensor,
        input_residual_prop_S: torch.Tensor,
        input_n_total: torch.Tensor,
        input_snapshot_batch_idx: torch.Tensor,
        n_snapshots_in_batch: int,
        target_species_idx: torch.Tensor,
        target_family_idx: torch.Tensor,
        target_snapshot_batch_idx: torch.Tensor,
        target_baseline_logit: torch.Tensor,
        target_current_prop_S: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        final_logits, delta_logits, z_snapshot = self.residual_model(
            input_species_idx=input_species_idx,
            input_family_idx=input_family_idx,
            input_p_baseline=input_p_baseline,
            input_residual_prop_S=input_residual_prop_S,
            input_n_total=input_n_total,
            input_snapshot_batch_idx=input_snapshot_batch_idx,
            n_snapshots_in_batch=n_snapshots_in_batch,
            target_species_idx=target_species_idx,
            target_family_idx=target_family_idx,
            target_snapshot_batch_idx=target_snapshot_batch_idx,
            target_baseline_logit=target_baseline_logit,
        )

        target_snapshot = z_snapshot[target_snapshot_batch_idx]
        target_species = self._species_embedding()(target_species_idx)
        target_family = self._family_embedding()(target_family_idx)
        current_probability = target_current_prop_S.clamp(0.0, 1.0).unsqueeze(-1)

        features = torch.cat(
            [
                target_snapshot,
                target_species,
                target_family,
                current_probability,
            ],
            dim=-1,
        )
        if int(features.shape[-1]) != int(self.jump_feature_dim):
            raise RuntimeError(
                f"Direction feature width is {features.shape[-1]}, expected "
                f"{self.jump_feature_dim}."
            )

        down_logit = self.down_head(features).squeeze(-1)
        up_logit = self.up_head(features).squeeze(-1)
        return final_logits, delta_logits, z_snapshot, down_logit, up_logit
