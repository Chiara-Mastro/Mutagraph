# scripts/01_create_standard_dataset.py
# pyright: reportMissingImports=false

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT_FROM_SCRIPT))

from src.input import load_input_table, standardize_columns, apply_year_filter, aggregate_if_needed, add_model_columns, final_qc


def parse_args():
    parser = argparse.ArgumentParser(description=("Create the standard AMR modeling table aggregated by Species, Family, Country, Year."))
    parser.add_argument("--input-path", type=Path, required=True, help="Input CSV with columns: Species, Family, Country, Year, status, n_S, n_total.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output CSV path for the standardized aggregated table.")
    parser.add_argument("--min-year", type=int, default=None, help="Optional minimum year to keep.")
    parser.add_argument("--max-year", type=int, default=None, help="Optional maximum year to keep.")
    return parser.parse_args()



def main():
    args = parse_args()

    df = load_input_table(args.input_path)
    df = standardize_columns(df)
    df = apply_year_filter(df, args.min_year, args.max_year)

    output_df = aggregate_if_needed(df)
    output_df = add_model_columns(output_df)
    final_qc(output_df)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.suffix.lower() != ".csv":
        args.output_path.mkdir(parents=True, exist_ok=True)
        args.output_path = args.output_path / "amr_disaggregated.csv"
    else:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(args.output_path, index=False)

    print("Saved standardized aggregated table:")
    print(args.output_path)
    print("Shape:", output_df.shape)
    print(output_df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

