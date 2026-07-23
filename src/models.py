import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT_FROM_SCRIPT))

from src.CONFIG import (
     STATUS_OBSERVED,
     STATUS_IMPUTE,
     EPS)


class VariationalResidualEncoderModel(nn.Module):
    """
    Variational snapshot encoder for residual AMR landscape completion.

    This is the variational analogue of SnapshotEncoderResidualModel.

    It keeps the residual formulation:

        final_logit = target_baseline_logit + delta_logit

    but replaces the deterministic snapshot latent state z with:

        q(z | observed snapshot edges) = Normal(mu, diag(exp(logvar)))

    Expected forward inputs are compatible with AMRSnapshotResidualDataset and
    snapshot_residual_collate_fn.
    """

    def __init__(
        self,
        n_species: int,
        n_families: int,
        entity_emb_dim: int = 16,
        edge_hidden_dim: int = 64,
        latent_dim: int = 12,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.10,
        min_logvar: float = -8.0,
        max_logvar: float = 4.0,
    ):
        super().__init__()

        self.n_species = int(n_species)
        self.n_families = int(n_families)
        self.entity_emb_dim = int(entity_emb_dim)
        self.edge_hidden_dim = int(edge_hidden_dim)
        self.latent_dim = int(latent_dim)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.dropout = float(dropout)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        self.species_embedding = nn.Embedding(
            self.n_species,
            self.entity_emb_dim,
        )
        self.family_embedding = nn.Embedding(
            self.n_families,
            self.entity_emb_dim,
        )

        # Per observed edge, the encoder sees:
        #   species embedding
        #   family embedding
        #   baseline probability p_baseline
        #   observed residual prop_S - p_baseline
        #   log(1 + n_total), as a soft reliability / sample-size signal
        edge_input_dim = 2 * self.entity_emb_dim + 3

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, self.edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.edge_hidden_dim, self.edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.mu_head = nn.Linear(self.edge_hidden_dim, self.latent_dim)
        self.logvar_head = nn.Linear(self.edge_hidden_dim, self.latent_dim)

        decoder_input_dim = self.latent_dim + 2 * self.entity_emb_dim

        self.residual_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.decoder_hidden_dim, 1),
        )

        # Same interpretation as in the deterministic residual model:
        # beta-binomial overdispersion/concentration parameter is softplus(log_phi).
        self.log_phi = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))

    def _encode_edges(
        self,
        input_species_idx: torch.Tensor,
        input_family_idx: torch.Tensor,
        input_p_baseline: torch.Tensor,
        input_residual_prop_S: torch.Tensor,
        input_n_total: torch.Tensor,
    ) :
        species_emb = self.species_embedding(input_species_idx)
        family_emb = self.family_embedding(input_family_idx)

        numeric_features = torch.stack(
            [
                input_p_baseline.float().clamp(1e-6, 1.0 - 1e-6),
                input_residual_prop_S.float(),
                torch.log1p(input_n_total.float().clamp_min(0.0)),
            ],
            dim=-1,
        )

        edge_features = torch.cat(
            [species_emb, family_emb, numeric_features],
            dim=-1,
        )

        return self.edge_encoder(edge_features)

    @staticmethod
    def _weighted_snapshot_pool(
        edge_h: torch.Tensor,
        input_n_total: torch.Tensor,
        input_snapshot_batch_idx: torch.Tensor,
        n_snapshots_in_batch: int,
    ):
        """
        Weighted mean-pool observed edge embeddings into one vector per snapshot.

        log(1 + n_total) is used as a tempered reliability weight. This avoids a
        single huge-n cell completely dominating the snapshot representation,
        because apparently raw counts enjoy turning neural networks into puppets.
        """
        n_snapshots = int(n_snapshots_in_batch)
        hidden_dim = edge_h.shape[-1]

        weights = torch.log1p(input_n_total.float().clamp_min(0.0)).clamp_min(1.0)
        weights = weights.unsqueeze(-1)

        pooled = edge_h.new_zeros((n_snapshots, hidden_dim))
        denom = edge_h.new_zeros((n_snapshots, 1))

        pooled.index_add_(0, input_snapshot_batch_idx, edge_h * weights)
        denom.index_add_(0, input_snapshot_batch_idx, weights)

        return pooled / denom.clamp_min(1e-6)

    def encode(
        self,
        input_species_idx: torch.Tensor,
        input_family_idx: torch.Tensor,
        input_p_baseline: torch.Tensor,
        input_residual_prop_S: torch.Tensor,
        input_n_total: torch.Tensor,
        input_snapshot_batch_idx: torch.Tensor,
        n_snapshots_in_batch: int,
    ) :
        edge_h = self._encode_edges(
            input_species_idx=input_species_idx,
            input_family_idx=input_family_idx,
            input_p_baseline=input_p_baseline,
            input_residual_prop_S=input_residual_prop_S,
            input_n_total=input_n_total,
        )

        snapshot_h = self._weighted_snapshot_pool(
            edge_h=edge_h,
            input_n_total=input_n_total,
            input_snapshot_batch_idx=input_snapshot_batch_idx,
            n_snapshots_in_batch=n_snapshots_in_batch,
        )

        mu_snapshot = self.mu_head(snapshot_h)
        logvar_snapshot = self.logvar_head(snapshot_h)
        logvar_snapshot = logvar_snapshot.clamp(
            min=self.min_logvar,
            max=self.max_logvar,
        )

        return mu_snapshot, logvar_snapshot

    @staticmethod
    def reparameterize(
        mu_snapshot: torch.Tensor,
        logvar_snapshot: torch.Tensor,
        sample_latent: bool,
    ):
        if not sample_latent:
            return mu_snapshot

        std_snapshot = torch.exp(0.5 * logvar_snapshot)
        eps = torch.randn_like(std_snapshot)
        return mu_snapshot + eps * std_snapshot

    @staticmethod
    def kl_divergence_standard_normal(
        mu_snapshot: torch.Tensor,
        logvar_snapshot: torch.Tensor,
    ) :
        """
        KL[q(z|x) || N(0, I)] per snapshot.
        Shape: [n_snapshots_in_batch]
        """
        return 0.5 * torch.sum(
            torch.exp(logvar_snapshot)
            + mu_snapshot.pow(2)
            - 1.0
            - logvar_snapshot,
            dim=-1,
        )

    def decode(
        self,
        z_snapshot: torch.Tensor,
        target_species_idx: torch.Tensor,
        target_family_idx: torch.Tensor,
        target_snapshot_batch_idx: torch.Tensor,
        target_baseline_logit: torch.Tensor,
    ):
        target_z = z_snapshot[target_snapshot_batch_idx]
        target_species_emb = self.species_embedding(target_species_idx)
        target_family_emb = self.family_embedding(target_family_idx)

        decoder_input = torch.cat(
            [target_z, target_species_emb, target_family_emb],
            dim=-1,
        )

        delta_logits = self.residual_decoder(decoder_input).squeeze(-1)
        final_logits = target_baseline_logit.float() + delta_logits

        return final_logits, delta_logits

    def forward(
        self,
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
        sample_latent: bool | None = None,
    ) :
        """
        Forward pass.
        """
        if sample_latent is None:
            sample_latent = self.training

        mu_snapshot, logvar_snapshot = self.encode(
            input_species_idx=input_species_idx,
            input_family_idx=input_family_idx,
            input_p_baseline=input_p_baseline,
            input_residual_prop_S=input_residual_prop_S,
            input_n_total=input_n_total,
            input_snapshot_batch_idx=input_snapshot_batch_idx,
            n_snapshots_in_batch=n_snapshots_in_batch,
        )

        z_snapshot = self.reparameterize(
            mu_snapshot=mu_snapshot,
            logvar_snapshot=logvar_snapshot,
            sample_latent=bool(sample_latent),
        )

        final_logits, delta_logits = self.decode(
            z_snapshot=z_snapshot,
            target_species_idx=target_species_idx,
            target_family_idx=target_family_idx,
            target_snapshot_batch_idx=target_snapshot_batch_idx,
            target_baseline_logit=target_baseline_logit,
        )

        kl_per_snapshot = self.kl_divergence_standard_normal(
            mu_snapshot=mu_snapshot,
            logvar_snapshot=logvar_snapshot,
        )

        return (
            final_logits,
            delta_logits,
            z_snapshot,
            mu_snapshot,
            logvar_snapshot,
            kl_per_snapshot,
        )

    def sample_predictions(
        self,
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
        n_samples: int = 100,
    ) :
        """
        Draw multiple posterior latent samples and decode target probabilities.

        Useful for 09_export_landscape_predictions.py when exporting posterior
        means, standard deviations, and quantiles.
        """
        mu_snapshot, logvar_snapshot = self.encode(
            input_species_idx=input_species_idx,
            input_family_idx=input_family_idx,
            input_p_baseline=input_p_baseline,
            input_residual_prop_S=input_residual_prop_S,
            input_n_total=input_n_total,
            input_snapshot_batch_idx=input_snapshot_batch_idx,
            n_snapshots_in_batch=n_snapshots_in_batch,
        )

        logits_samples = []
        delta_samples = []

        for _ in range(int(n_samples)):
            z_snapshot = self.reparameterize(
                mu_snapshot=mu_snapshot,
                logvar_snapshot=logvar_snapshot,
                sample_latent=True,
            )
            final_logits, delta_logits = self.decode(
                z_snapshot=z_snapshot,
                target_species_idx=target_species_idx,
                target_family_idx=target_family_idx,
                target_snapshot_batch_idx=target_snapshot_batch_idx,
                target_baseline_logit=target_baseline_logit,
            )
            logits_samples.append(final_logits)
            delta_samples.append(delta_logits)

        logits_samples = torch.stack(logits_samples, dim=0)
        delta_samples = torch.stack(delta_samples, dim=0)
        p_samples = torch.sigmoid(logits_samples)

        kl_per_snapshot = self.kl_divergence_standard_normal(
            mu_snapshot=mu_snapshot,
            logvar_snapshot=logvar_snapshot,
        )

        return {
            "logits_samples": logits_samples,
            "delta_logits_samples": delta_samples,
            "p_samples": p_samples,
            "mu_snapshot": mu_snapshot,
            "logvar_snapshot": logvar_snapshot,
            "kl_per_snapshot": kl_per_snapshot,
        }



class SnapshotEncoderResidualModel(nn.Module):
    """
    Encoder-based latent AMR residual model.

    Unlike the free latent factor model, this model does not learn a free
    parameter for each Country-Year snapshot.

    Instead, it infers the snapshot latent state from the observed train cells:

        observed train edges in snapshot -> encoder -> z_snapshot

    Then it decodes target Species-Family cells using:

        z_snapshot + species embedding + family embedding -> delta_logit

        final_logit = target_baseline_logit + delta_logit

    This is a simple DeepSets-style encoder:
        edge embeddings are computed independently and then averaged within
        each snapshot.
    """

    def __init__(
        self,
        n_species: int,
        n_families: int,
        entity_emb_dim: int = 16,
        edge_hidden_dim: int = 64,
        latent_dim: int = 12,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.species_embedding = nn.Embedding(n_species, entity_emb_dim)
        self.family_embedding = nn.Embedding(n_families, entity_emb_dim)

        # Edge input:
        # species emb + family emb + prop_S + log1p(n_total)
        edge_input_dim = entity_emb_dim + entity_emb_dim + 3

        self.log_phi = nn.Parameter(torch.tensor(4.0))

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.ReLU(),
        )

        self.snapshot_encoder = nn.Sequential(
            nn.Linear(edge_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, latent_dim),
        )

        decoder_input_dim = latent_dim + entity_emb_dim + entity_emb_dim

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, 1),
        )

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.species_embedding.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.family_embedding.weight, mean=0.0, std=0.05)

    def encode_snapshot(
        self,
        input_species_idx,
        input_family_idx,
        input_p_baseline,
        input_residual_prop_S,
        input_n_total,
        input_snapshot_batch_idx,
        n_snapshots_in_batch,
    ):
        species_emb = self.species_embedding(input_species_idx)
        family_emb = self.family_embedding(input_family_idx)

        log_n_total = torch.log1p(input_n_total).unsqueeze(-1)
        p_baseline = input_p_baseline.unsqueeze(-1)
        res_prop_S = input_residual_prop_S.unsqueeze(-1)

        edge_features = torch.cat(
            [
                species_emb,
                family_emb,
                p_baseline, 
                res_prop_S,
                log_n_total,
            ],
            dim=-1,
        )

        edge_hidden = self.edge_encoder(edge_features)

        # Weighted mean aggregation by snapshot.
        # Weights use log1p(n_total): larger cells matter more, but not so much
        # that a huge cell completely eats the snapshot alive. Dataset behavior,
        # as usual, must be supervised like a toddler near scissors.
        edge_weights = torch.log1p(input_n_total).clamp_min(1.0)
        weighted_edge_hidden = edge_hidden * edge_weights.unsqueeze(-1)

        snapshot_sum = torch.zeros(
            n_snapshots_in_batch,
            edge_hidden.shape[-1],
            device=edge_hidden.device,
            dtype=edge_hidden.dtype,
        )

        weight_sum = torch.zeros(
            n_snapshots_in_batch,
            device=edge_hidden.device,
            dtype=edge_hidden.dtype,
        )

        snapshot_sum.index_add_(
            0,
            input_snapshot_batch_idx,
            weighted_edge_hidden,
        )

        weight_sum.index_add_(
            0,
            input_snapshot_batch_idx,
            edge_weights,
        )

        snapshot_mean = snapshot_sum / weight_sum.clamp_min(1e-6).unsqueeze(-1)

        z_snapshot = self.snapshot_encoder(snapshot_mean)

        return z_snapshot

    def decode_targets(
        self,
        z_snapshot,
        target_species_idx,
        target_family_idx,
        target_snapshot_batch_idx,
        target_baseline_logit,
    ):
        target_species_emb = self.species_embedding(target_species_idx)
        target_family_emb = self.family_embedding(target_family_idx)

        target_z = z_snapshot[target_snapshot_batch_idx]

        decoder_input = torch.cat(
            [
                target_z,
                target_species_emb,
                target_family_emb,
            ],
            dim=-1,
        )

        delta_logits = self.decoder(decoder_input).squeeze(-1)

        final_logits = target_baseline_logit + delta_logits

        return final_logits, delta_logits

    def forward(
        self,
        input_species_idx,
        input_family_idx,
        input_p_baseline,
        input_residual_prop_S,
        input_n_total,
        input_snapshot_batch_idx,
        n_snapshots_in_batch,
        target_species_idx,
        target_family_idx,
        target_snapshot_batch_idx,
        target_baseline_logit,
    ):
        z_snapshot = self.encode_snapshot(
            input_species_idx=input_species_idx,
            input_family_idx=input_family_idx,
            input_p_baseline=input_p_baseline,
            input_residual_prop_S=input_residual_prop_S,
            input_n_total=input_n_total,
            input_snapshot_batch_idx=input_snapshot_batch_idx,
            n_snapshots_in_batch=n_snapshots_in_batch,
        )

        final_logits, delta_logits = self.decode_targets(
            z_snapshot=z_snapshot,
            target_species_idx=target_species_idx,
            target_family_idx=target_family_idx,
            target_snapshot_batch_idx=target_snapshot_batch_idx,
            target_baseline_logit=target_baseline_logit,
        )

        return final_logits, delta_logits, z_snapshot


class SnapshotEncoderCompletionModel(nn.Module):
    """
    Encoder-based latent AMR completion model.

    Unlike the free latent factor model, this model does not learn a free
    parameter for each Country-Year snapshot.

    Instead, it infers the snapshot latent state from the observed train cells:

        observed train edges in snapshot -> encoder -> z_snapshot

    Then it decodes target Species-Family cells using:

        z_snapshot + species embedding + family embedding -> logit p_S

    This is a simple DeepSets-style encoder:
        edge embeddings are computed independently and then averaged within
        each snapshot.
    """

    def __init__(
        self,
        n_species: int,
        n_families: int,
        entity_emb_dim: int = 16,
        edge_hidden_dim: int = 64,
        latent_dim: int = 12,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.species_embedding = nn.Embedding(n_species, entity_emb_dim)
        self.family_embedding = nn.Embedding(n_families, entity_emb_dim)

        # Edge input:
        # species emb + family emb + prop_S + log1p(n_total)
        edge_input_dim = entity_emb_dim + entity_emb_dim + 2

        self.log_phi = nn.Parameter(torch.tensor(4.0))

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.ReLU(),
        )

        self.snapshot_encoder = nn.Sequential(
            nn.Linear(edge_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, latent_dim),
        )

        decoder_input_dim = latent_dim + entity_emb_dim + entity_emb_dim

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, 1),
        )

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.species_embedding.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.family_embedding.weight, mean=0.0, std=0.05)

    def encode_snapshot(
        self,
        input_species_idx,
        input_family_idx,
        input_prop_S,
        input_n_total,
        input_snapshot_batch_idx,
        n_snapshots_in_batch,
    ):
        species_emb = self.species_embedding(input_species_idx)
        family_emb = self.family_embedding(input_family_idx)

        log_n_total = torch.log1p(input_n_total).unsqueeze(-1)
        prop_S = input_prop_S.unsqueeze(-1)

        edge_features = torch.cat(
            [
                species_emb,
                family_emb,
                prop_S,
                log_n_total,
            ],
            dim=-1,
        )

        edge_hidden = self.edge_encoder(edge_features)

        # Weighted mean aggregation by snapshot.
        # Weights use log1p(n_total): larger cells matter more, but not so much
        # that a huge cell completely eats the snapshot alive. Dataset behavior,
        # as usual, must be supervised like a toddler near scissors.
        edge_weights = torch.log1p(input_n_total).clamp_min(1.0)
        weighted_edge_hidden = edge_hidden * edge_weights.unsqueeze(-1)

        snapshot_sum = torch.zeros(
            n_snapshots_in_batch,
            edge_hidden.shape[-1],
            device=edge_hidden.device,
            dtype=edge_hidden.dtype,
        )

        weight_sum = torch.zeros(
            n_snapshots_in_batch,
            device=edge_hidden.device,
            dtype=edge_hidden.dtype,
        )

        snapshot_sum.index_add_(
            0,
            input_snapshot_batch_idx,
            weighted_edge_hidden,
        )

        weight_sum.index_add_(
            0,
            input_snapshot_batch_idx,
            edge_weights,
        )

        snapshot_mean = snapshot_sum / weight_sum.clamp_min(1e-6).unsqueeze(-1)

        z_snapshot = self.snapshot_encoder(snapshot_mean)

        return z_snapshot

    def decode_targets(
        self,
        z_snapshot,
        target_species_idx,
        target_family_idx,
        target_snapshot_batch_idx,
    ):
        target_species_emb = self.species_embedding(target_species_idx)
        target_family_emb = self.family_embedding(target_family_idx)

        target_z = z_snapshot[target_snapshot_batch_idx]

        decoder_input = torch.cat(
            [
                target_z,
                target_species_emb,
                target_family_emb,
            ],
            dim=-1,
        )

        logits = self.decoder(decoder_input).squeeze(-1)

        return logits

    def forward(
        self,
        input_species_idx,
        input_family_idx,
        input_prop_S,
        input_n_total,
        input_snapshot_batch_idx,
        n_snapshots_in_batch,
        target_species_idx,
        target_family_idx,
        target_snapshot_batch_idx,
    ):
        z_snapshot = self.encode_snapshot(
            input_species_idx=input_species_idx,
            input_family_idx=input_family_idx,
            input_prop_S=input_prop_S,
            input_n_total=input_n_total,
            input_snapshot_batch_idx=input_snapshot_batch_idx,
            n_snapshots_in_batch=n_snapshots_in_batch,
        )

        logits = self.decode_targets(
            z_snapshot=z_snapshot,
            target_species_idx=target_species_idx,
            target_family_idx=target_family_idx,
            target_snapshot_batch_idx=target_snapshot_batch_idx,
        )

        return logits, z_snapshot



class BaseLatentCompletionModel(nn.Module):
    """
    Base latent AMR completion model.

    The model learns:
        - one latent embedding for each Country-Year snapshot;
        - one embedding for each Species;
        - one embedding for each Family.

    It predicts:
        the logits because it's better in the loss function than predicting probabilities directly.
        p_S = P(susceptible | snapshot, species, family)
    """

    def __init__(
        self,
        n_snapshots: int,
        n_species: int,
        n_families: int,
        latent_dim: int = 12,
        entity_emb_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.snapshot_embedding = nn.Embedding(n_snapshots, latent_dim)
        self.species_embedding = nn.Embedding(n_species, entity_emb_dim)
        self.family_embedding = nn.Embedding(n_families, entity_emb_dim)

        # Global beta-binomial concentration parameter.
        # Higher phi means less overdispersion, lower phi means more overdispersion.
        self.log_phi = nn.Parameter(torch.tensor(4.0))

        input_dim = latent_dim + entity_emb_dim + entity_emb_dim

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.snapshot_embedding.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.species_embedding.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.family_embedding.weight, mean=0.0, std=0.05)

    def forward(self, snapshot_idx, species_idx, family_idx):
        z_snapshot = self.snapshot_embedding(snapshot_idx)
        z_species = self.species_embedding(species_idx)
        z_family = self.family_embedding(family_idx)

        x = torch.cat(
            [z_snapshot, z_species, z_family],
            dim=-1,
        )

        logits = self.decoder(x).squeeze(-1)

        return logits
    


class SharedDynamicsResidualModel(nn.Module):
    """
    Shared-dynamics temporal residual model.

    Snapshot (Country, t)'s observed residual cells are encoded into a
    latent AMR state z_{c,t}, using the same  edge-encoder-then-weighted-pool architecture as
    SnapshotEncoderResidualModel.encode_snapshot (species/family embedding + scalar features 
    -> per-edge hidden -> log1p(n_total)-weighted mean pool -> snapshot_encoder MLP).

    A single SHARED transition function F_theta maps z_{c,t} -> z_hat_{c,t+1}:

        z_hat_{c,t+1} = z_{c,t} + MLP(z_{c,t})

    The residual-connection form means the untrained/default behavior is
    persistence (z unchanged from t to t+1); F_theta only has to learn the
    DEVIATION from persistence, which matters because plain persistence
    (LOCF) and short rolling windows are already strong baselines.

    The decoder predicts target Species-Family residual corrections from
    z_hat_{c,t+1}, using the same architecture as
    SnapshotEncoderResidualModel.decode_targets:

        final_logit_{c,t+1,s,f} = baseline_logit_{s,f,t+1} + delta_logit

    Training pairs are (Country, t) -> (Country, t+1); encoder, transition,
    and decoder weights are shared across every country -- no country ever
    gets its own parameters, which is the "shared dynamics" assumption.
    """

    def __init__(
        self,
        n_species: int,
        n_families: int,
        entity_emb_dim: int = 16,
        edge_hidden_dim: int = 64,
        latent_dim: int = 12,
        decoder_hidden_dim: int = 64,
        transition_hidden_dim: int = 32,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.species_embedding = nn.Embedding(n_species, entity_emb_dim)
        self.family_embedding = nn.Embedding(n_families, entity_emb_dim)

        # Edge input: species emb + family emb + residual_logit + log1p(n_total)
        edge_input_dim = entity_emb_dim + entity_emb_dim + 2

        self.log_phi = nn.Parameter(torch.tensor(4.0))

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.ReLU(),
        )

        self.snapshot_encoder = nn.Sequential(
            nn.Linear(edge_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, latent_dim),
        )

        self.transition = nn.Sequential(
            nn.Linear(latent_dim, transition_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transition_hidden_dim, latent_dim),
        )

        decoder_input_dim = latent_dim + entity_emb_dim + entity_emb_dim

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, 1),
        )

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.species_embedding.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.family_embedding.weight, mean=0.0, std=0.05)

        # Start the transition near-identity (small final-layer weights and
        # zero bias) so training starts close to "predict persistence" --
        # the residual connection then only has to learn a small correction,
        # not overcome a large random initial disruption to z.
        last_linear = self.transition[-1]
        nn.init.normal_(last_linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(last_linear.bias)

    def encode_snapshot(
        self,
        input_species_idx,
        input_family_idx,
        input_residual_logit,
        input_n_total,
        input_snapshot_batch_idx,
        n_snapshots_in_batch,
    ):
        species_emb = self.species_embedding(input_species_idx)
        family_emb = self.family_embedding(input_family_idx)

        log_n_total = torch.log1p(input_n_total).unsqueeze(-1)
        residual_logit = input_residual_logit.unsqueeze(-1)

        edge_features = torch.cat(
            [species_emb, family_emb, residual_logit, log_n_total],
            dim=-1,
        )

        edge_hidden = self.edge_encoder(edge_features)

        edge_weights = torch.log1p(input_n_total).clamp_min(1.0)
        weighted_edge_hidden = edge_hidden * edge_weights.unsqueeze(-1)

        snapshot_sum = torch.zeros(
            n_snapshots_in_batch,
            edge_hidden.shape[-1],
            device=edge_hidden.device,
            dtype=edge_hidden.dtype,
        )
        weight_sum = torch.zeros(
            n_snapshots_in_batch,
            device=edge_hidden.device,
            dtype=edge_hidden.dtype,
        )

        snapshot_sum.index_add_(0, input_snapshot_batch_idx, weighted_edge_hidden)
        weight_sum.index_add_(0, input_snapshot_batch_idx, edge_weights)

        snapshot_mean = snapshot_sum / weight_sum.clamp_min(1e-6).unsqueeze(-1)

        z_t = self.snapshot_encoder(snapshot_mean)

        return z_t

    def transition_step(self, z_t):
        return z_t + self.transition(z_t)

    def decode_targets(
        self,
        z_next,
        target_species_idx,
        target_family_idx,
        target_snapshot_batch_idx,
        target_baseline_logit,
    ):
        target_species_emb = self.species_embedding(target_species_idx)
        target_family_emb = self.family_embedding(target_family_idx)

        target_z = z_next[target_snapshot_batch_idx]

        decoder_input = torch.cat([target_z, target_species_emb, target_family_emb], dim=-1)

        delta_logits = self.decoder(decoder_input).squeeze(-1)
        final_logits = target_baseline_logit + delta_logits

        return final_logits, delta_logits

    def forward(
        self,
        input_species_idx,
        input_family_idx,
        input_residual_logit,
        input_n_total,
        input_snapshot_batch_idx,
        n_snapshots_in_batch,
        target_species_idx,
        target_family_idx,
        target_snapshot_batch_idx,
        target_baseline_logit,
    ):
        z_t = self.encode_snapshot(
            input_species_idx=input_species_idx,
            input_family_idx=input_family_idx,
            input_residual_logit=input_residual_logit,
            input_n_total=input_n_total,
            input_snapshot_batch_idx=input_snapshot_batch_idx,
            n_snapshots_in_batch=n_snapshots_in_batch,
        )

        z_next = self.transition_step(z_t)

        final_logits, delta_logits = self.decode_targets(
            z_next=z_next,
            target_species_idx=target_species_idx,
            target_family_idx=target_family_idx,
            target_snapshot_batch_idx=target_snapshot_batch_idx,
            target_baseline_logit=target_baseline_logit,
        )

        return final_logits, delta_logits, z_t, z_next

    
