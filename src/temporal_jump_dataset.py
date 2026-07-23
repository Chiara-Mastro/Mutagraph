"""Temporal residual dataset with prospective jump labels.

Each item contains a complete Country snapshot at year t and all observed
cells at year t plus one. Direction labels are defined only from information in
the two observed snapshots and are used solely as supervised targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


REQUIRED_CELL_COLUMNS = [
    "cell_row_id",
    "Country",
    "Year",
    "species_idx",
    "family_idx",
    "n_S",
    "n_total",
    "prop_S",
    "p_baseline",
    "baseline_logit",
    "residual_prop_S",
]

REQUIRED_PAIR_COLUMNS = [
    "pair_id",
    "Country",
    "input_year",
    "target_year",
    "split",
]


def _int_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise")
    return values.to_numpy(dtype=np.int64)


def _float_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise")
    array = values.to_numpy(dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Column {column!r} contains non finite values.")
    return array


def _passes_minimum(
    values: np.ndarray,
    minimum: float,
    rule: str,
) -> np.ndarray:
    if rule == "inclusive":
        return values >= minimum
    if rule == "exclusive":
        return values > minimum
    raise ValueError(f"Unknown minimum rule: {rule}")


class TemporalJumpDataset(Dataset):
    def __init__(
        self,
        cells: pd.DataFrame,
        pairs: pd.DataFrame,
        *,
        jump_threshold: float = 0.10,
        jump_min_tests: float = 80.0,
        jump_min_tests_rule: str = "inclusive",
        jump_label_policy: str = "gnn_stable",
    ) -> None:
        missing_cells = [
            column for column in REQUIRED_CELL_COLUMNS
            if column not in cells.columns
        ]
        missing_pairs = [
            column for column in REQUIRED_PAIR_COLUMNS
            if column not in pairs.columns
        ]
        if missing_cells:
            raise ValueError(f"Cell table is missing columns: {missing_cells}")
        if missing_pairs:
            raise ValueError(f"Pair table is missing columns: {missing_pairs}")
        if pairs.empty:
            raise ValueError("Pair table is empty.")
        if jump_threshold <= 0:
            raise ValueError("jump_threshold must be positive.")
        if jump_min_tests < 0:
            raise ValueError("jump_min_tests must be nonnegative.")
        if jump_label_policy not in {"eligible_only", "gnn_stable"}:
            raise ValueError(
                "jump_label_policy must be eligible_only or gnn_stable."
            )

        self.groups = {
            (str(country), int(year)): group.reset_index(drop=True)
            for (country, year), group in cells.groupby(
                ["Country", "Year"], sort=True
            )
        }
        self.pairs = pairs.reset_index(drop=True).copy()
        self.jump_threshold = float(jump_threshold)
        self.jump_min_tests = float(jump_min_tests)
        self.jump_min_tests_rule = str(jump_min_tests_rule)
        self.jump_label_policy = str(jump_label_policy)

        missing_snapshots: list[str] = []
        for row in self.pairs.itertuples(index=False):
            input_key = (str(row.Country), int(row.input_year))
            target_key = (str(row.Country), int(row.target_year))
            if input_key not in self.groups:
                missing_snapshots.append(
                    f"{row.pair_id}: missing input snapshot"
                )
            if target_key not in self.groups:
                missing_snapshots.append(
                    f"{row.pair_id}: missing target snapshot"
                )
        if missing_snapshots:
            raise ValueError(
                "Pair table refers to absent snapshots. Examples:\n"
                + "\n".join(missing_snapshots[:20])
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.pairs.iloc[index]
        country = str(pair["Country"])
        input_year = int(pair["input_year"])
        target_year = int(pair["target_year"])

        input_group = self.groups[(country, input_year)]
        target_group = self.groups[(country, target_year)]

        current_table = input_group[
            ["species_idx", "family_idx", "prop_S", "n_total"]
        ].rename(
            columns={
                "prop_S": "current_prop_S",
                "n_total": "current_n_total",
            }
        )

        aligned = target_group[
            ["species_idx", "family_idx"]
        ].copy()
        aligned["target_order"] = np.arange(len(aligned), dtype=np.int64)
        aligned = aligned.merge(
            current_table,
            on=["species_idx", "family_idx"],
            how="left",
            validate="one_to_one",
            sort=False,
        ).sort_values("target_order")

        current_observed = aligned["current_prop_S"].notna().to_numpy(bool)
        current_prop = pd.to_numeric(
            aligned["current_prop_S"], errors="coerce"
        ).fillna(0.5).to_numpy(dtype=np.float32)
        current_n = pd.to_numeric(
            aligned["current_n_total"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float32)

        target_prop = _float_array(target_group, "prop_S")
        target_n = _float_array(target_group, "n_total")
        observed_delta = target_prop - current_prop

        current_large = _passes_minimum(
            current_n, self.jump_min_tests, self.jump_min_tests_rule
        )
        target_large = _passes_minimum(
            target_n, self.jump_min_tests, self.jump_min_tests_rule
        )
        well_sampled = current_observed & current_large & target_large

        down_label = well_sampled & (
            observed_delta <= -self.jump_threshold
        )
        up_label = well_sampled & (
            observed_delta >= self.jump_threshold
        )

        if self.jump_label_policy == "eligible_only":
            loss_mask = well_sampled
        else:
            # This reproduces the colleague convention in which undersampled
            # observed transitions are retained as stable negatives.
            loss_mask = current_observed

        return {
            "pair_id": str(pair["pair_id"]),
            "country": country,
            "input_year": input_year,
            "target_year": target_year,
            "input_species_idx": torch.as_tensor(
                _int_array(input_group, "species_idx"), dtype=torch.long
            ),
            "input_family_idx": torch.as_tensor(
                _int_array(input_group, "family_idx"), dtype=torch.long
            ),
            "input_p_baseline": torch.as_tensor(
                _float_array(input_group, "p_baseline"), dtype=torch.float32
            ),
            "input_residual_prop_S": torch.as_tensor(
                _float_array(input_group, "residual_prop_S"),
                dtype=torch.float32,
            ),
            "input_n_total": torch.as_tensor(
                _float_array(input_group, "n_total"), dtype=torch.float32
            ),
            "target_cell_row_id": torch.as_tensor(
                _int_array(target_group, "cell_row_id"), dtype=torch.long
            ),
            "target_species_idx": torch.as_tensor(
                _int_array(target_group, "species_idx"), dtype=torch.long
            ),
            "target_family_idx": torch.as_tensor(
                _int_array(target_group, "family_idx"), dtype=torch.long
            ),
            "target_n_S": torch.as_tensor(
                _float_array(target_group, "n_S"), dtype=torch.float32
            ),
            "target_n_total": torch.as_tensor(
                target_n, dtype=torch.float32
            ),
            "target_prop_S": torch.as_tensor(
                target_prop, dtype=torch.float32
            ),
            "target_baseline_logit": torch.as_tensor(
                _float_array(target_group, "baseline_logit"),
                dtype=torch.float32,
            ),
            "target_current_prop_S": torch.as_tensor(
                current_prop, dtype=torch.float32
            ),
            "target_current_n_total": torch.as_tensor(
                current_n, dtype=torch.float32
            ),
            "target_current_observed": torch.as_tensor(
                current_observed, dtype=torch.bool
            ),
            "jump_well_sampled": torch.as_tensor(
                well_sampled, dtype=torch.bool
            ),
            "jump_loss_mask": torch.as_tensor(
                loss_mask, dtype=torch.bool
            ),
            "jump_observed_delta": torch.as_tensor(
                observed_delta, dtype=torch.float32
            ),
            "jump_down_label": torch.as_tensor(
                down_label.astype(np.float32), dtype=torch.float32
            ),
            "jump_up_label": torch.as_tensor(
                up_label.astype(np.float32), dtype=torch.float32
            ),
        }


def temporal_jump_collate_fn(
    batch: list[dict[str, object]],
) -> dict[str, object]:
    tensor_keys = [
        "input_species_idx",
        "input_family_idx",
        "input_p_baseline",
        "input_residual_prop_S",
        "input_n_total",
        "target_cell_row_id",
        "target_species_idx",
        "target_family_idx",
        "target_n_S",
        "target_n_total",
        "target_prop_S",
        "target_baseline_logit",
        "target_current_prop_S",
        "target_current_n_total",
        "target_current_observed",
        "jump_well_sampled",
        "jump_loss_mask",
        "jump_observed_delta",
        "jump_down_label",
        "jump_up_label",
    ]
    collected: dict[str, list[torch.Tensor]] = {
        key: [] for key in tensor_keys
    }
    input_batch_idx: list[torch.Tensor] = []
    target_batch_idx: list[torch.Tensor] = []
    pair_ids: list[str] = []
    countries: list[str] = []
    input_years: list[int] = []
    target_years: list[int] = []

    for batch_index, item in enumerate(batch):
        pair_ids.append(str(item["pair_id"]))
        countries.append(str(item["country"]))
        input_years.append(int(item["input_year"]))
        target_years.append(int(item["target_year"]))

        n_input = len(item["input_species_idx"])
        n_target = len(item["target_species_idx"])
        input_batch_idx.append(
            torch.full((n_input,), batch_index, dtype=torch.long)
        )
        target_batch_idx.append(
            torch.full((n_target,), batch_index, dtype=torch.long)
        )
        for key in tensor_keys:
            collected[key].append(item[key])

    output: dict[str, object] = {
        "pair_ids": pair_ids,
        "countries": countries,
        "input_years": input_years,
        "target_years": target_years,
        "input_snapshot_batch_idx": torch.cat(input_batch_idx),
        "target_snapshot_batch_idx": torch.cat(target_batch_idx),
        "n_snapshots_in_batch": len(batch),
    }
    for key, values in collected.items():
        output[key] = torch.cat(values)
    return output


def build_temporal_jump_dataloaders(
    cells: pd.DataFrame,
    pairs: pd.DataFrame,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    jump_threshold: float = 0.10,
    jump_min_tests: float = 80.0,
    jump_min_tests_rule: str = "inclusive",
    jump_label_policy: str = "gnn_stable",
) -> tuple[dict[str, DataLoader], dict[str, TemporalJumpDataset]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least one.")

    loaders: dict[str, DataLoader] = {}
    datasets: dict[str, TemporalJumpDataset] = {}

    for split in ["train", "val", "test"]:
        split_pairs = pairs.loc[pairs["split"].eq(split)].copy()
        if split_pairs.empty:
            raise ValueError(f"No temporal pairs for split {split!r}.")

        dataset = TemporalJumpDataset(
            cells=cells,
            pairs=split_pairs,
            jump_threshold=jump_threshold,
            jump_min_tests=jump_min_tests,
            jump_min_tests_rule=jump_min_tests_rule,
            jump_label_policy=jump_label_policy,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=temporal_jump_collate_fn,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
        )
        datasets[split] = dataset
        loaders[split] = loader

    return loaders, datasets
