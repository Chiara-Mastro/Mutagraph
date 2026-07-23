# src/amr_config.py

from pathlib import Path

DEFAULT_PROJECT_ROOT = Path(".../AMR_latent")

DEFAULT_INPUT_PATH = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_grid_status.csv"
)

DEFAULT_OUTPUT_PATH = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "processed"
    / "v1"
    / "amr_species_family_country_year.csv"
)

REQUIRED_COLUMNS = [
    "Species",
    "Family",
    "Country",
    "Year",
    "status",
    "n_S",
    "n_total",
]

KEY_COLUMNS = [
    "Species",
    "Family",
    "Country",
    "Year",
]

STATUS_OBSERVED = "observed"
STATUS_IMPUTE = "to_impute"
STATUS_INTRINSIC_RESISTANCE = "do_not_impute_intrinsic"

EPS = 1e-6

BASELINE_LEVELS = {
    "species_family": [
        (["Species", "Family"], "species_family"),
        (["Family"], "family"),
        (["Species"], "species"),
    ],
    "country_species_family": [
        (["Country", "Species", "Family"], "country_species_family"),
        (["Species", "Family"], "species_family"),
        (["Family"], "family"),
        (["Species"], "species"),
    ],
}