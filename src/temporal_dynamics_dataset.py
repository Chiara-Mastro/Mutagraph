"""
(Country, t) -> (Country, t+1) transition-pair dataset for the shared-dynamics
temporal residual model.

Each item is one Country whose observed cells in calendar year t (the INPUT
snapshot) are used by the encoder, and whose observed cells in calendar year
t+1 (the TARGET snapshot) are what the decoder, after one shared transition
step, is trained/evaluated against. Pairs require t and t+1 to be literally
consecutive calendar years for that country. build_transition_pairs() reports
how many candidate pairs were dropped for this reason so the gap isn't invisible.

Splitting is by TARGET year t+1 (not by pair index), matching one-step-ahead
forecasting evaluation: a single model is trained once on all pairs whose
target year is "early enough", then evaluated one-step-ahead, unchanged, on
each later target year -- this is different from src/temporal_baselines.py's baselines,
which are refit from scratch at every target year. We have a fixed, shared transition
function evaluated one-step-ahead across several future years.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


REQUIRED_COLUMNS = [
    "Country", "Year", "species_idx", "family_idx", "n_S", "n_total",
    "prop_S", "baseline_logit", "residual_logit",
]


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



def build_transition_pairs(df: pd.DataFrame):
    """
    Return (pairs, n_dropped_for_gap), where pairs is a list of
    (country, t, t_next) with t_next == t + 1 and both years having at
    least one observed cell for that country.
    """
    years_by_country = (
        df.groupby("Country")["Year"].apply(lambda s: sorted(s.unique().tolist()))
    )

    pairs = []
    n_dropped = 0

    for country, years in years_by_country.items():
        year_set = set(years)
        for t in years:
            t_next = t + 1
            if t_next in year_set:
                pairs.append((country, t, t_next))
            else:
                n_dropped += 1

    return pairs, n_dropped


class CountryYearTransitionDataset(Dataset):
    """
    One item per (Country, t, t_next) pair: all observed cells for that
    country in year t (encoder input) and all observed cells for that
    country in year t_next (decoder target).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pairs: list,
        min_input_cells: int = 1,
        min_target_cells: int = 1,
    ):
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.groups = {
            (country, year): group.reset_index(drop=True)
            for (country, year), group in df.groupby(["Country", "Year"], sort=True)
        }

        self.pairs = []
        for country, t, t_next in pairs:
            input_group = self.groups.get((country, t))
            target_group = self.groups.get((country, t_next))

            if input_group is None or target_group is None:
                continue
            if len(input_group) < min_input_cells:
                continue
            if len(target_group) < min_target_cells:
                continue

            self.pairs.append((country, t, t_next))

        if not self.pairs:
            raise ValueError(
                "No transition pairs survived min_input_cells/min_target_cells "
                "filtering. Lower these thresholds or check the input data."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        country, t, t_next = self.pairs[idx]
        input_group = self.groups[(country, t)]
        target_group = self.groups[(country, t_next)]

        return {
            "country": country,
            "t": t,
            "t_next": t_next,
            "input_species_idx": torch.tensor(input_group["species_idx"].to_numpy(), dtype=torch.long),
            "input_family_idx": torch.tensor(input_group["family_idx"].to_numpy(), dtype=torch.long),
            "input_residual_logit": torch.tensor(input_group["residual_logit"].to_numpy(), dtype=torch.float32),
            "input_n_total": torch.tensor(input_group["n_total"].to_numpy(), dtype=torch.float32),
            "target_species_idx": torch.tensor(target_group["species_idx"].to_numpy(), dtype=torch.long),
            "target_family_idx": torch.tensor(target_group["family_idx"].to_numpy(), dtype=torch.long),
            "target_n_S": torch.tensor(target_group["n_S"].to_numpy(), dtype=torch.float32),
            "target_n_total": torch.tensor(target_group["n_total"].to_numpy(), dtype=torch.float32),
            "target_prop_S": torch.tensor(target_group["prop_S"].to_numpy(), dtype=torch.float32),
            "target_baseline_logit": torch.tensor(target_group["baseline_logit"].to_numpy(), dtype=torch.float32),
        }


def transition_collate_fn(batch):
    input_species_idx, input_family_idx = [], []
    input_residual_logit, input_n_total, input_batch_idx = [], [], []

    target_species_idx, target_family_idx = [], []
    target_n_S, target_n_total, target_prop_S = [], [], []
    target_baseline_logit, target_batch_idx = [], []

    countries, ts, t_nexts = [], [], []

    for batch_idx, item in enumerate(batch):
        countries.append(item["country"])
        ts.append(item["t"])
        t_nexts.append(item["t_next"])

        n_input = len(item["input_species_idx"])
        n_target = len(item["target_species_idx"])

        input_species_idx.append(item["input_species_idx"])
        input_family_idx.append(item["input_family_idx"])
        input_residual_logit.append(item["input_residual_logit"])
        input_n_total.append(item["input_n_total"])
        input_batch_idx.append(torch.full((n_input,), batch_idx, dtype=torch.long))

        target_species_idx.append(item["target_species_idx"])
        target_family_idx.append(item["target_family_idx"])
        target_n_S.append(item["target_n_S"])
        target_n_total.append(item["target_n_total"])
        target_prop_S.append(item["target_prop_S"])
        target_baseline_logit.append(item["target_baseline_logit"])
        target_batch_idx.append(torch.full((n_target,), batch_idx, dtype=torch.long))

    return {
        "countries": countries,
        "ts": ts,
        "t_nexts": t_nexts,

        "input_species_idx": torch.cat(input_species_idx),
        "input_family_idx": torch.cat(input_family_idx),
        "input_residual_logit": torch.cat(input_residual_logit),
        "input_n_total": torch.cat(input_n_total),
        "input_snapshot_batch_idx": torch.cat(input_batch_idx),

        "target_species_idx": torch.cat(target_species_idx),
        "target_family_idx": torch.cat(target_family_idx),
        "target_n_S": torch.cat(target_n_S),
        "target_n_total": torch.cat(target_n_total),
        "target_prop_S": torch.cat(target_prop_S),
        "target_baseline_logit": torch.cat(target_baseline_logit),
        "target_snapshot_batch_idx": torch.cat(target_batch_idx),

        "n_snapshots_in_batch": len(batch),
    }


def build_transition_dataloaders(
    df: pd.DataFrame,
    batch_size: int,
    val_year: int,
    test_min_year: int,
    min_input_cells: int = 1,
    min_target_cells: int = 1,
):
    """
    Split (Country, t, t_next) pairs by target year t_next:
        train: t_next <  val_year
        val:   t_next == val_year
        test:  t_next >= test_min_year

    val_year must be < test_min_year. Pairs with val_year <= t_next <
    test_min_year (if any) belong to neither split and are reported, not
    silently used -- e.g. deliberately left as a buffer, or simply an
    off-by-one in the two cutoffs.
    """
    if val_year >= test_min_year:
        raise ValueError("val_year must be strictly less than test_min_year.")

    pairs, n_dropped_for_gap = build_transition_pairs(df)
    print(f"Transition pairs: {len(pairs)} total, {n_dropped_for_gap} dropped "
          "for non-consecutive-year gaps.")

    split_pairs = {"train": [], "val": [], "test": []}
    n_unused = 0

    for country, t, t_next in pairs:
        if t_next < val_year:
            split_pairs["train"].append((country, t, t_next))
        elif t_next == val_year:
            split_pairs["val"].append((country, t, t_next))
        elif t_next >= test_min_year:
            split_pairs["test"].append((country, t, t_next))
        else:
            n_unused += 1

    if n_unused:
        print(f"Note: {n_unused} pairs fall strictly between val_year and "
              "test_min_year and are unused by either split.")

    datasets = {}
    loaders = {}

    for split in ["train", "val", "test"]:
        if not split_pairs[split]:
            raise ValueError(
                f"No transition pairs for split={split!r}. Adjust "
                "--val-year/--test-min-year or check the Year range."
            )

        datasets[split] = CountryYearTransitionDataset(
            df=df,
            pairs=split_pairs[split],
            min_input_cells=min_input_cells,
            min_target_cells=min_target_cells,
        )

        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=transition_collate_fn,
        )

        print(f"  {split}: {len(datasets[split])} pairs "
              f"(target years {sorted({p[2] for p in split_pairs[split]})})")

    return loaders, datasets



class TemporalResidualEncoderDataset(Dataset):
    def __init__(
        self,
        cells: pd.DataFrame,
        pairs: pd.DataFrame,
    ) -> None:
        missing_cells = [
            column
            for column in REQUIRED_CELL_COLUMNS
            if column not in cells.columns
        ]
        missing_pairs = [
            column
            for column in REQUIRED_PAIR_COLUMNS
            if column not in pairs.columns
        ]
        if missing_cells:
            raise ValueError(
                f"Cell table is missing columns: {missing_cells}"
            )
        if missing_pairs:
            raise ValueError(
                f"Pair table is missing columns: {missing_pairs}"
            )
        if pairs.empty:
            raise ValueError("Pair table is empty.")

        self.groups = {
            (str(country), int(year)): group.reset_index(drop=True)
            for (country, year), group in cells.groupby(
                ["Country", "Year"],
                sort=True,
            )
        }

        self.pairs = pairs.reset_index(drop=True).copy()

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

        return {
            "pair_id": str(pair["pair_id"]),
            "country": country,
            "input_year": input_year,
            "target_year": target_year,
            "input_species_idx": torch.tensor(
                input_group["species_idx"].to_numpy(),
                dtype=torch.long,
            ),
            "input_family_idx": torch.tensor(
                input_group["family_idx"].to_numpy(),
                dtype=torch.long,
            ),
            "input_p_baseline": torch.tensor(
                input_group["p_baseline"].to_numpy(),
                dtype=torch.float32,
            ),
            "input_residual_prop_S": torch.tensor(
                input_group["residual_prop_S"].to_numpy(),
                dtype=torch.float32,
            ),
            "input_n_total": torch.tensor(
                input_group["n_total"].to_numpy(),
                dtype=torch.float32,
            ),
            "target_cell_row_id": torch.tensor(
                target_group["cell_row_id"].to_numpy(),
                dtype=torch.long,
            ),
            "target_species_idx": torch.tensor(
                target_group["species_idx"].to_numpy(),
                dtype=torch.long,
            ),
            "target_family_idx": torch.tensor(
                target_group["family_idx"].to_numpy(),
                dtype=torch.long,
            ),
            "target_n_S": torch.tensor(
                target_group["n_S"].to_numpy(),
                dtype=torch.float32,
            ),
            "target_n_total": torch.tensor(
                target_group["n_total"].to_numpy(),
                dtype=torch.float32,
            ),
            "target_prop_S": torch.tensor(
                target_group["prop_S"].to_numpy(),
                dtype=torch.float32,
            ),
            "target_baseline_logit": torch.tensor(
                target_group["baseline_logit"].to_numpy(),
                dtype=torch.float32,
            ),
        }


def temporal_residual_collate_fn(
    batch: list[dict[str, object]],
) -> dict[str, object]:
    input_species_idx = []
    input_family_idx = []
    input_p_baseline = []
    input_residual_prop_S = []
    input_n_total = []
    input_batch_idx = []

    target_cell_row_id = []
    target_species_idx = []
    target_family_idx = []
    target_n_S = []
    target_n_total = []
    target_prop_S = []
    target_baseline_logit = []
    target_batch_idx = []

    pair_ids = []
    countries = []
    input_years = []
    target_years = []

    for batch_index, item in enumerate(batch):
        pair_ids.append(item["pair_id"])
        countries.append(item["country"])
        input_years.append(item["input_year"])
        target_years.append(item["target_year"])

        n_input = len(item["input_species_idx"])
        n_target = len(item["target_species_idx"])

        input_species_idx.append(item["input_species_idx"])
        input_family_idx.append(item["input_family_idx"])
        input_p_baseline.append(item["input_p_baseline"])
        input_residual_prop_S.append(
            item["input_residual_prop_S"]
        )
        input_n_total.append(item["input_n_total"])
        input_batch_idx.append(
            torch.full(
                (n_input,),
                batch_index,
                dtype=torch.long,
            )
        )

        target_cell_row_id.append(item["target_cell_row_id"])
        target_species_idx.append(item["target_species_idx"])
        target_family_idx.append(item["target_family_idx"])
        target_n_S.append(item["target_n_S"])
        target_n_total.append(item["target_n_total"])
        target_prop_S.append(item["target_prop_S"])
        target_baseline_logit.append(
            item["target_baseline_logit"]
        )
        target_batch_idx.append(
            torch.full(
                (n_target,),
                batch_index,
                dtype=torch.long,
            )
        )

    return {
        "pair_ids": pair_ids,
        "countries": countries,
        "input_years": input_years,
        "target_years": target_years,
        "input_species_idx": torch.cat(input_species_idx),
        "input_family_idx": torch.cat(input_family_idx),
        "input_p_baseline": torch.cat(input_p_baseline),
        "input_residual_prop_S": torch.cat(
            input_residual_prop_S
        ),
        "input_n_total": torch.cat(input_n_total),
        "input_snapshot_batch_idx": torch.cat(input_batch_idx),
        "target_cell_row_id": torch.cat(target_cell_row_id),
        "target_species_idx": torch.cat(target_species_idx),
        "target_family_idx": torch.cat(target_family_idx),
        "target_n_S": torch.cat(target_n_S),
        "target_n_total": torch.cat(target_n_total),
        "target_prop_S": torch.cat(target_prop_S),
        "target_baseline_logit": torch.cat(
            target_baseline_logit
        ),
        "target_snapshot_batch_idx": torch.cat(
            target_batch_idx
        ),
        "n_snapshots_in_batch": len(batch),
    }


def build_temporal_residual_encoder_dataloaders(
    cells: pd.DataFrame,
    pairs: pd.DataFrame,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[dict[str, DataLoader], dict[str, Dataset]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least one.")

    loaders: dict[str, DataLoader] = {}
    datasets: dict[str, Dataset] = {}

    for split in ["train", "val", "test"]:
        split_pairs = pairs.loc[pairs["split"].eq(split)].copy()
        if split_pairs.empty:
            raise ValueError(f"No temporal pairs for split {split!r}.")

        dataset = TemporalResidualEncoderDataset(
            cells=cells,
            pairs=split_pairs,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=temporal_residual_collate_fn,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
        )

        datasets[split] = dataset
        loaders[split] = loader

    return loaders, datasets



