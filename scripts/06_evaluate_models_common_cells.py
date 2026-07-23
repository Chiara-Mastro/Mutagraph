#!/usr/bin/env python3
"""Evaluate completion and temporal AMR models with one common protocol.

This single script produces two independent comparison tables.

Completion table
    GNN reconstruction, Species Family prior, and residual encoder evaluated
    on the exact same observed cells.

Temporal table
    GNN next year projection, Species Family temporal mean, LOCF, rolling
    mean, EWMA residual, and temporal residual encoder evaluated on the exact
    same observed transitions.

For both tasks, metrics are computed inside each external country fold and
then summarized across the five folds. The default report scope is the fixed
2023 and 2024 vault only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln

UNIFIED_SCRIPT_VERSION = "2026_07_23_common_protocol_v2"


completion_KEY_COLUMNS = ['Country', 'Year', 'Species', 'Family']
completion_EXPECTED_MODELS = ['GNN', 'species_family_prior', 'snapshot_encoder_residual_model']
completion_DISPLAY_NAMES = {'GNN': 'GNN', 'species_family_prior': 'Species Family prior', 'snapshot_encoder_residual_model': 'Residual encoder'}
completion_PRIMARY_METRICS = ['weighted_mae', 'weighted_rmse', 'beta_binomial_nll_per_test']
completion_HISTORICAL_TASK = 'country_generalization_through_2022'
completion_VAULT_TASK = 'country_and_year_generalization_2023_2024'
completion_EPS = 1e-08
completion_OBSERVED_PROP_TOLERANCE = 1e-06
completion_SCRIPT_VERSION = '2026_07_23_v4'

class completion_ProtocolError(ValueError):
    """Raised when source predictions cannot support a fair comparison."""

def completion_require_columns(frame: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise completion_ProtocolError(f'{table_name} is missing columns {missing}. Available columns: {list(frame.columns)}')

def completion_clean_text(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = out[column].astype(str).str.strip()
    return out

def completion_numeric_series(frame: pd.DataFrame, column: str, table_name: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors='coerce')
    invalid = values.isna() | ~np.isfinite(values.to_numpy(dtype=float))
    if invalid.any():
        examples = frame.loc[invalid, column].head(10).tolist()
        raise completion_ProtocolError(f'{table_name}.{column} contains invalid values: {examples}')
    return values

def completion_canonical_observed_proportion(frame: pd.DataFrame, source_column: str, table_name: str) -> pd.Series:
    """Return n_S divided by n_total after checking the exported proportion.

    GNN exports commonly store observed proportions in float32, while script 02
    recomputes the same ratio in float64. The source value is checked with a
    tolerance appropriate for float32 and the exact count ratio is then used as
    the common outcome for every model.
    """
    counts_ratio = frame['n_S'].to_numpy(dtype=float) / frame['n_total'].to_numpy(dtype=float)
    source = frame[source_column].to_numpy(dtype=float)
    close = np.isclose(source, counts_ratio, rtol=completion_OBSERVED_PROP_TOLERANCE, atol=completion_OBSERVED_PROP_TOLERANCE)
    if not close.all():
        bad = frame.loc[~close, completion_KEY_COLUMNS + ['n_S', 'n_total', source_column]].copy()
        bad['prop_from_counts'] = counts_ratio[~close]
        bad['absolute_difference'] = np.abs(source[~close] - counts_ratio[~close])
        raise completion_ProtocolError(f'{table_name}.{source_column} is materially inconsistent with n_S divided by n_total. Examples:\n' + bad.head(20).to_string(index=False))
    return pd.Series(counts_ratio, index=frame.index, dtype=float)

def completion_assert_unique(frame: pd.DataFrame, columns: list[str], table_name: str) -> None:
    duplicated = frame.duplicated(columns, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, columns].head(20)
        raise completion_ProtocolError(f'{table_name} contains duplicate keys:\n' + examples.to_string(index=False))

def completion_normalize_fold_values(values: pd.Series, expected_folds: int, table_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors='coerce')
    if numeric.isna().any():
        raise completion_ProtocolError(f'{table_name} contains missing fold values.')
    numeric = numeric.astype(int)
    unique = sorted(numeric.unique().tolist())
    zero_based = list(range(expected_folds))
    one_based = list(range(1, expected_folds + 1))
    if unique == zero_based:
        return numeric + 1
    if unique == one_based:
        return numeric
    if len(unique) == expected_folds:
        mapping = {original: position for position, original in enumerate(unique, start=1)}
        return numeric.map(mapping).astype(int)
    raise completion_ProtocolError(f'{table_name} contains fold values {unique}, expected {zero_based} or {one_based}.')

def completion_load_gnn_reconstruction(path: Path, expected_folds: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = completion_KEY_COLUMNS + ['n_S', 'n_total', 'prop_S_observed', 'prop_S_pred', 'alpha_cal', 'beta_cal', 'fold_model']
    completion_require_columns(frame, required, 'gnn_reconstruction')
    frame = completion_clean_text(frame, ['Country', 'Species', 'Family'])
    for column in ['Year', 'n_S', 'n_total', 'prop_S_observed', 'prop_S_pred', 'alpha_cal', 'beta_cal', 'fold_model']:
        frame[column] = completion_numeric_series(frame, column, 'gnn_reconstruction')
    frame['Year'] = frame['Year'].astype(int)
    frame['fold'] = completion_normalize_fold_values(frame['fold_model'], expected_folds, 'gnn_reconstruction')
    completion_assert_unique(frame, completion_KEY_COLUMNS, 'gnn_reconstruction')
    if not frame['prop_S_observed'].between(0, 1).all():
        raise completion_ProtocolError('GNN observed proportions must lie between zero and one.')
    frame['prop_S_observed'] = completion_canonical_observed_proportion(frame, source_column='prop_S_observed', table_name='gnn_reconstruction')
    if not frame['prop_S_pred'].between(0, 1).all():
        raise completion_ProtocolError('GNN predictions must lie between zero and one.')
    if (frame['n_total'] <= 0).any():
        raise completion_ProtocolError('GNN n_total must be positive.')
    if (frame['n_S'] < 0).any() or (frame['n_S'] > frame['n_total']).any():
        raise completion_ProtocolError('GNN n_S must lie between zero and n_total.')
    if (frame['alpha_cal'] <= 0).any() or (frame['beta_cal'] <= 0).any():
        raise completion_ProtocolError('GNN alpha_cal and beta_cal must be positive.')
    out = frame[completion_KEY_COLUMNS + ['fold', 'n_S', 'n_total', 'prop_S_observed', 'prop_S_pred', 'alpha_cal', 'beta_cal']].copy()
    out = out.rename(columns={'prop_S_observed': 'observed_prop_S', 'prop_S_pred': 'p_pred', 'alpha_cal': 'alpha_pred', 'beta_cal': 'beta_pred'})
    out['model_name'] = 'GNN'
    out['model_display_name'] = completion_DISPLAY_NAMES['GNN']
    out['source_table'] = str(path)
    return out

def completion_completion_alpha_beta(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if {'bb_alpha_at_mean', 'bb_beta_at_mean'}.issubset(frame.columns):
        alpha = pd.to_numeric(frame['bb_alpha_at_mean'], errors='coerce')
        beta = pd.to_numeric(frame['bb_beta_at_mean'], errors='coerce')
        if alpha.notna().all() and beta.notna().all():
            return (alpha, beta)
    if 'phi_train' not in frame.columns:
        raise completion_ProtocolError('Completion predictions need bb_alpha_at_mean and bb_beta_at_mean, or phi_train.')
    phi = pd.to_numeric(frame['phi_train'], errors='coerce')
    p = pd.to_numeric(frame['p_pred'], errors='coerce')
    return (p * phi, (1.0 - p) * phi)

def completion_load_completion_predictions(path: Path, expected_folds: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = completion_KEY_COLUMNS + ['model_name', 'n_S', 'n_total', 'prop_S', 'p_pred', 'fold']
    completion_require_columns(frame, required, 'completion_predictions')
    frame = completion_clean_text(frame, ['Country', 'Species', 'Family', 'model_name'])
    frame = frame.loc[frame['model_name'].isin(completion_EXPECTED_MODELS[1:])].copy()
    missing_models = sorted(set(completion_EXPECTED_MODELS[1:]) - set(frame['model_name'].unique()))
    if missing_models:
        raise completion_ProtocolError(f'Completion predictions are missing models: {missing_models}')
    for column in ['Year', 'n_S', 'n_total', 'prop_S', 'p_pred', 'fold']:
        frame[column] = completion_numeric_series(frame, column, 'completion_predictions')
    frame['Year'] = frame['Year'].astype(int)
    frame['fold'] = completion_normalize_fold_values(frame['fold'], expected_folds, 'completion_predictions')
    frame['alpha_pred'], frame['beta_pred'] = completion_completion_alpha_beta(frame)
    if frame[['alpha_pred', 'beta_pred']].isna().any(axis=None):
        raise completion_ProtocolError('Completion alpha or beta contains missing values.')
    if (frame['alpha_pred'] <= 0).any() or (frame['beta_pred'] <= 0).any():
        raise completion_ProtocolError('Completion alpha and beta must be positive.')
    if not frame['prop_S'].between(0, 1).all():
        raise completion_ProtocolError('Completion observed proportions must lie between zero and one.')
    frame['prop_S'] = completion_canonical_observed_proportion(frame, source_column='prop_S', table_name='completion_predictions')
    if not frame['p_pred'].between(0, 1).all():
        raise completion_ProtocolError('Completion predictions must lie between zero and one.')
    completion_assert_unique(frame, completion_KEY_COLUMNS + ['model_name'], 'completion_predictions')
    frame = frame.rename(columns={'prop_S': 'observed_prop_S'})
    frame['model_display_name'] = frame['model_name'].map(completion_DISPLAY_NAMES)
    frame['source_table'] = str(path)
    return frame[completion_KEY_COLUMNS + ['fold', 'model_name', 'model_display_name', 'n_S', 'n_total', 'observed_prop_S', 'p_pred', 'alpha_pred', 'beta_pred', 'source_table']].copy()

def completion_load_fold_assignments(path: Path, expected_folds: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == '.json':
        payload = json.loads(path.read_text(encoding='utf8'))
        rows: list[dict[str, object]] = []
        if not isinstance(payload, dict):
            raise completion_ProtocolError('Fold JSON must contain a dictionary.')
        if all((isinstance(value, list) for value in payload.values())):
            for raw_fold, countries in payload.items():
                for country in countries:
                    rows.append({'Country': str(country).strip(), 'raw_fold': raw_fold})
        elif all((not isinstance(value, (dict, list)) for value in payload.values())):
            for country, raw_fold in payload.items():
                rows.append({'Country': str(country).strip(), 'raw_fold': raw_fold})
        else:
            raise completion_ProtocolError('Fold JSON must map folds to country lists or countries to folds.')
        frame = pd.DataFrame(rows)
    else:
        source = pd.read_csv(path)
        completion_require_columns(source, ['Country'], 'country_folds')
        fold_column = next((column for column in ['fold_model', 'colleague_fold', 'external_fold', 'fold'] if column in source.columns), None)
        if fold_column is None:
            raise completion_ProtocolError('Country fold CSV needs fold_model, colleague_fold, external_fold, or fold.')
        frame = source[['Country', fold_column]].rename(columns={fold_column: 'raw_fold'})
        frame['Country'] = frame['Country'].astype(str).str.strip()
    frame = frame.drop_duplicates().copy()
    duplicated = frame.duplicated('Country', keep=False)
    if duplicated.any():
        raise completion_ProtocolError('Country fold file assigns a country more than once.')
    frame['fold'] = completion_normalize_fold_values(frame['raw_fold'], expected_folds, 'country_folds')
    return frame[['Country', 'fold']].copy()

def completion_verify_fold_assignments(predictions: pd.DataFrame, assignments: pd.DataFrame, table_name: str) -> None:
    observed = predictions[['Country', 'fold']].drop_duplicates()
    country_counts = observed.groupby('Country')['fold'].nunique()
    if country_counts.gt(1).any():
        bad = country_counts.loc[country_counts.gt(1)].index.tolist()[:20]
        raise completion_ProtocolError(f'{table_name} assigns countries to multiple folds: {bad}')
    check = observed.merge(assignments, on='Country', how='left', suffixes=('_observed', '_expected'), validate='one_to_one')
    if check['fold_expected'].isna().any():
        missing = check.loc[check['fold_expected'].isna(), 'Country'].tolist()[:20]
        raise completion_ProtocolError(f'{table_name} contains countries absent from fold file: {missing}')
    mismatch = check['fold_observed'].ne(check['fold_expected'])
    if mismatch.any():
        raise completion_ProtocolError(f'{table_name} disagrees with the supplied country fold file.')

def completion_key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[completion_KEY_COLUMNS].drop_duplicates())

def completion_create_common_cell_table(gnn: pd.DataFrame, completion: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_tables = {'GNN': gnn, 'species_family_prior': completion.loc[completion['model_name'].eq('species_family_prior')].copy(), 'snapshot_encoder_residual_model': completion.loc[completion['model_name'].eq('snapshot_encoder_residual_model')].copy()}
    common_keys = completion_key_index(model_tables['GNN'])
    coverage_rows: list[dict[str, object]] = []
    for model_name, frame in model_tables.items():
        model_keys = completion_key_index(frame)
        common_keys = common_keys.intersection(model_keys)
        coverage_rows.append({'model_name': model_name, 'model_display_name': completion_DISPLAY_NAMES[model_name], 'n_source_cells': int(len(model_keys))})
    if len(common_keys) == 0:
        raise completion_ProtocolError('The three models have no cells in common.')
    common_key_frame = common_keys.to_frame(index=False)
    standardized_parts: list[pd.DataFrame] = []
    canonical = gnn.merge(common_key_frame, on=completion_KEY_COLUMNS, how='inner', validate='one_to_one')[completion_KEY_COLUMNS + ['fold', 'n_S', 'n_total', 'observed_prop_S']].copy()
    for model_name, frame in model_tables.items():
        part = frame.merge(common_key_frame, on=completion_KEY_COLUMNS, how='inner', validate='one_to_one').copy()
        comparison = part.merge(canonical, on=completion_KEY_COLUMNS, how='inner', suffixes=('', '_canonical'), validate='one_to_one')
        if not comparison['fold'].eq(comparison['fold_canonical']).all():
            raise completion_ProtocolError(f'Fold assignments disagree for model {model_name}.')
        for column in ['n_S', 'n_total']:
            left = comparison[column].to_numpy(dtype=float)
            right = comparison[f'{column}_canonical'].to_numpy(dtype=float)
            close = np.isclose(left, right, rtol=0.0, atol=1e-06)
            if not close.all():
                examples = comparison.loc[~close, completion_KEY_COLUMNS + [column, f'{column}_canonical']].head(20)
                raise completion_ProtocolError(f'Observed count column {column} disagrees for model {model_name}. Examples:\n' + examples.to_string(index=False))
        comparison['observed_prop_S_canonical'] = comparison['n_S_canonical'].to_numpy(dtype=float) / comparison['n_total_canonical'].to_numpy(dtype=float)
        standardized_parts.append(comparison[completion_KEY_COLUMNS + ['fold_canonical', 'n_S_canonical', 'n_total_canonical', 'observed_prop_S_canonical', 'model_name', 'model_display_name', 'p_pred', 'alpha_pred', 'beta_pred', 'source_table']].rename(columns={'fold_canonical': 'fold', 'n_S_canonical': 'n_S', 'n_total_canonical': 'n_total', 'observed_prop_S_canonical': 'observed_prop_S'}))
    standardized = pd.concat(standardized_parts, ignore_index=True, sort=False)
    expected_rows = len(common_keys) * len(completion_EXPECTED_MODELS)
    if len(standardized) != expected_rows:
        raise completion_ProtocolError(f'Expected {expected_rows} standardized rows, found {len(standardized)}.')
    for row in coverage_rows:
        row['n_common_cells'] = int(len(common_keys))
        row['fraction_retained'] = float(len(common_keys) / row['n_source_cells']) if row['n_source_cells'] > 0 else np.nan
    coverage = pd.DataFrame(coverage_rows)
    return (standardized, coverage)

def completion_beta_binomial_nll(n_s: np.ndarray, n_total: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    k = np.asarray(n_s, dtype=float)
    n = np.asarray(n_total, dtype=float)
    a = np.maximum(np.asarray(alpha, dtype=float), completion_EPS)
    b = np.maximum(np.asarray(beta, dtype=float), completion_EPS)
    return -(gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0) + betaln(k + a, n - k + b) - betaln(a, b))

def completion_compute_metrics(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        raise completion_ProtocolError('Cannot calculate metrics on an empty table.')
    observed = frame['observed_prop_S'].to_numpy(dtype=float)
    predicted = np.clip(frame['p_pred'].to_numpy(dtype=float), completion_EPS, 1.0 - completion_EPS)
    n_s = frame['n_S'].to_numpy(dtype=float)
    n_total = frame['n_total'].to_numpy(dtype=float)
    alpha = frame['alpha_pred'].to_numpy(dtype=float)
    beta = frame['beta_pred'].to_numpy(dtype=float)
    error = observed - predicted
    absolute_error = np.abs(error)
    squared_error = error ** 2
    weights = np.sqrt(n_total)
    nll = completion_beta_binomial_nll(n_s, n_total, alpha, beta)
    return {'n_cells': int(len(frame)), 'n_tests': int(round(float(n_total.sum()))), 'n_countries': int(frame['Country'].nunique()), 'weighted_mae': float(np.average(absolute_error, weights=weights)), 'weighted_rmse': float(math.sqrt(np.average(squared_error, weights=weights))), 'beta_binomial_nll_per_test': float(np.sum(nll) / np.sum(n_total)), 'unweighted_mae': float(np.mean(absolute_error)), 'unweighted_rmse': float(math.sqrt(np.mean(squared_error))), 'mean_signed_error_observed_minus_predicted': float(np.mean(error))}

def completion_assign_evaluation_task(frame: pd.DataFrame, historical_max_year: int, vault_years: list[int]) -> pd.DataFrame:
    out = frame.copy()
    out['evaluation_task'] = np.select([out['Year'].le(historical_max_year), out['Year'].isin(vault_years)], [completion_HISTORICAL_TASK, completion_VAULT_TASK], default='outside_configured_tasks')
    outside = out['evaluation_task'].eq('outside_configured_tasks')
    if outside.any():
        years = sorted(out.loc[outside, 'Year'].unique().tolist())
        print('Ignoring rows outside historical and vault tasks. Years: ' + str(years))
        out = out.loc[~outside].copy()
    expected = {completion_HISTORICAL_TASK, completion_VAULT_TASK}
    observed = set(out['evaluation_task'].unique())
    missing = sorted(expected - observed)
    if missing:
        raise completion_ProtocolError(f'No common rows are available for tasks: {missing}')
    return out

def completion_calculate_fold_metrics(common: pd.DataFrame, expected_folds: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = common.groupby(['evaluation_task', 'model_name', 'model_display_name', 'fold'], sort=True)
    for (evaluation_task, model_name, display_name, fold), group in grouped:
        row: dict[str, object] = {'evaluation_family': 'completion_reconstruction', 'evaluation_task': evaluation_task, 'model_name': model_name, 'model_display_name': display_name, 'fold': int(fold), 'year_min': int(group['Year'].min()), 'year_max': int(group['Year'].max()), 'n_years': int(group['Year'].nunique())}
        row.update(completion_compute_metrics(group))
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(['evaluation_task', 'model_name', 'fold']).reset_index(drop=True)
    counts = result.groupby(['evaluation_task', 'model_name'])['fold'].nunique()
    bad = counts.loc[counts.ne(expected_folds)]
    if not bad.empty:
        raise completion_ProtocolError(f'Every model and task must contain all configured folds. Observed: {bad.to_dict()}')
    support_check = result.groupby(['evaluation_task', 'fold']).agg(n_cells_values=('n_cells', lambda values: values.nunique()), n_tests_values=('n_tests', lambda values: values.nunique()), n_countries_values=('n_countries', lambda values: values.nunique()))
    if support_check[['n_cells_values', 'n_tests_values', 'n_countries_values']].ne(1).any(axis=None):
        raise completion_ProtocolError('Models do not have identical support inside every task and fold.')
    return result

def completion_summarize_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = fold_metrics.groupby(['evaluation_family', 'evaluation_task', 'model_name', 'model_display_name'], sort=True)
    for keys, group in grouped:
        evaluation_family, evaluation_task, model_name, display_name = keys
        row: dict[str, object] = {'evaluation_family': evaluation_family, 'evaluation_task': evaluation_task, 'model_name': model_name, 'model_display_name': display_name, 'n_folds': int(group['fold'].nunique()), 'n_cells_total': int(group['n_cells'].sum()), 'n_tests_total': int(group['n_tests'].sum()), 'n_countries_total': int(group['n_countries'].sum()), 'year_min': int(group['year_min'].min()), 'year_max': int(group['year_max'].max()), 'error_weighting': 'sqrt_n_total', 'uncertainty_definition': 'sample standard deviation across fold metrics', 'standard_deviation_ddof': 1, 'cell_set_rule': 'exact intersection across all three models'}
        for metric in completion_PRIMARY_METRICS:
            values = pd.to_numeric(group[metric], errors='coerce').dropna()
            mean = float(values.mean()) if not values.empty else np.nan
            sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f'{metric}_mean'] = mean
            row[f'{metric}_sd'] = sd
            row[f'{metric}_mean_plus_minus_sd'] = f'{mean:.6f} ± {sd:.6f}' if np.isfinite(mean) and np.isfinite(sd) else ''
        rows.append(row)
    order = {name: index for index, name in enumerate(completion_EXPECTED_MODELS)}
    result = pd.DataFrame(rows)
    result['model_order'] = result['model_name'].map(order)
    result = result.sort_values(['evaluation_task', 'model_order']).drop(columns='model_order')
    return result.reset_index(drop=True)

temporal_TRANSITION_KEYS = ['Country', 'input_year', 'target_year', 'Species', 'Family']
temporal_GNN_RECONSTRUCTION_KEYS = ['Country', 'Year', 'Species', 'Family']
temporal_DEFAULT_TEMPORAL_MODELS = ['species_family_mean', 'rolling_mean_k', 'temporal_residual_encoder']
temporal_DISPLAY_NAMES = {'GNN': 'GNN', 'species_family_mean': 'Species Family temporal mean', 'temporal_residual_encoder': 'Temporal residual encoder', 'locf': 'LOCF', 'rolling_mean_k': 'Rolling mean k', 'ewma_residual': 'EWMA residual'}
temporal_PRIMARY_METRICS = ['weighted_mae', 'weighted_rmse', 'beta_binomial_nll_per_test']
temporal_HISTORICAL_TASK = 'country_generalization_through_2022'
temporal_VAULT_TASK = 'country_and_year_generalization_2023_2024'
temporal_EPS = 1e-08
temporal_OBSERVED_PROP_TOLERANCE = 1e-06
temporal_COUNT_TOLERANCE = 1e-06
temporal_SCRIPT_VERSION = '2026_07_23_temporal_v1'

class temporal_ProtocolError(ValueError):
    """Raised when source tables cannot support a fair comparison."""

def temporal_require_columns(frame: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise temporal_ProtocolError(f'{table_name} is missing columns {missing}. Available columns: {list(frame.columns)}')

def temporal_clean_text(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = out[column].astype(str).str.strip()
    return out

def temporal_numeric_series(frame: pd.DataFrame, column: str, table_name: str, *, allow_missing: bool=False) -> pd.Series:
    values = pd.to_numeric(frame[column], errors='coerce')
    invalid = ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    if allow_missing:
        invalid = invalid & values.notna().to_numpy()
    else:
        invalid = invalid | values.isna().to_numpy()
    if invalid.any():
        examples = frame.loc[invalid, column].head(10).tolist()
        raise temporal_ProtocolError(f'{table_name}.{column} contains invalid values: {examples}')
    return values

def temporal_boolean_series(values: pd.Series, column_name: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    mapped = values.astype(str).str.strip().str.lower().map({'true': True, 'false': False, '1': True, '0': False})
    if mapped.isna().any():
        examples = values.loc[mapped.isna()].head(10).tolist()
        raise temporal_ProtocolError(f'{column_name} contains invalid boolean values: {examples}')
    return mapped.astype(bool)

def temporal_assert_unique(frame: pd.DataFrame, columns: list[str], table_name: str) -> None:
    duplicated = frame.duplicated(columns, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, columns].head(20)
        raise temporal_ProtocolError(f'{table_name} contains duplicate keys:\n' + examples.to_string(index=False))

def temporal_normalize_fold_values(values: pd.Series, expected_folds: int, table_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors='coerce')
    if numeric.isna().any():
        raise temporal_ProtocolError(f'{table_name} contains missing fold values.')
    numeric = numeric.astype(int)
    unique = sorted(numeric.unique().tolist())
    zero_based = list(range(expected_folds))
    one_based = list(range(1, expected_folds + 1))
    if unique == zero_based:
        return numeric + 1
    if unique == one_based:
        return numeric
    if len(unique) == expected_folds:
        mapping = {original: position for position, original in enumerate(unique, start=1)}
        return numeric.map(mapping).astype(int)
    raise temporal_ProtocolError(f'{table_name} contains fold values {unique}, expected {zero_based} or {one_based}.')

def temporal_canonical_observed_proportion(frame: pd.DataFrame, source_column: str, table_name: str) -> pd.Series:
    ratio = frame['n_S'].to_numpy(dtype=float) / frame['n_total'].to_numpy(dtype=float)
    source = frame[source_column].to_numpy(dtype=float)
    close = np.isclose(source, ratio, rtol=temporal_OBSERVED_PROP_TOLERANCE, atol=temporal_OBSERVED_PROP_TOLERANCE)
    if not close.all():
        bad_columns = [column for column in temporal_TRANSITION_KEYS + ['n_S', 'n_total', source_column] if column in frame.columns]
        bad = frame.loc[~close, bad_columns].copy()
        bad['prop_from_counts'] = ratio[~close]
        bad['absolute_difference'] = np.abs(source[~close] - ratio[~close])
        raise temporal_ProtocolError(f'{table_name}.{source_column} is inconsistent with n_S divided by n_total. Examples:\n' + bad.head(20).to_string(index=False))
    return pd.Series(ratio, index=frame.index, dtype=float)

def temporal_load_gnn_reconstruction_targets(path: Path, expected_folds: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = temporal_GNN_RECONSTRUCTION_KEYS + ['n_S', 'n_total', 'prop_S_observed', 'fold_model']
    temporal_require_columns(frame, required, 'gnn_reconstruction')
    frame = temporal_clean_text(frame, ['Country', 'Species', 'Family'])
    for column in ['Year', 'n_S', 'n_total', 'prop_S_observed', 'fold_model']:
        frame[column] = temporal_numeric_series(frame, column, 'gnn_reconstruction')
    frame['Year'] = frame['Year'].astype(int)
    frame['fold'] = temporal_normalize_fold_values(frame['fold_model'], expected_folds, 'gnn_reconstruction')
    temporal_assert_unique(frame, temporal_GNN_RECONSTRUCTION_KEYS, 'gnn_reconstruction')
    if (frame['n_total'] <= 0).any():
        raise temporal_ProtocolError('GNN reconstruction n_total must be positive.')
    if (frame['n_S'] < 0).any() or (frame['n_S'] > frame['n_total']).any():
        raise temporal_ProtocolError('GNN reconstruction n_S must lie between zero and n_total.')
    if not frame['prop_S_observed'].between(0, 1).all():
        raise temporal_ProtocolError('GNN reconstruction observed proportions must lie in zero to one.')
    frame['observed_prop_S'] = temporal_canonical_observed_proportion(frame, source_column='prop_S_observed', table_name='gnn_reconstruction')
    return frame[temporal_GNN_RECONSTRUCTION_KEYS + ['fold', 'n_S', 'n_total', 'observed_prop_S']].copy()

def temporal_load_gnn_temporal_predictions(projection_path: Path, reconstruction_targets: pd.DataFrame, expected_folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not projection_path.exists():
        raise FileNotFoundError(projection_path)
    frame = pd.read_csv(projection_path)
    required = ['Country', 'year_from', 'year_to', 'Species', 'Family', 'prop_S_pred_next', 'alpha_cal', 'beta_cal', 'fold_model', 'next_year_in_data']
    temporal_require_columns(frame, required, 'gnn_projection')
    frame = temporal_clean_text(frame, ['Country', 'Species', 'Family'])
    for column in ['year_from', 'year_to', 'prop_S_pred_next', 'alpha_cal', 'beta_cal', 'fold_model']:
        frame[column] = temporal_numeric_series(frame, column, 'gnn_projection')
    frame['year_from'] = frame['year_from'].astype(int)
    frame['year_to'] = frame['year_to'].astype(int)
    frame['fold'] = temporal_normalize_fold_values(frame['fold_model'], expected_folds, 'gnn_projection')
    frame['next_year_in_data'] = temporal_boolean_series(frame['next_year_in_data'], 'next_year_in_data')
    frame = frame.rename(columns={'year_from': 'input_year', 'year_to': 'target_year'})
    temporal_assert_unique(frame, temporal_TRANSITION_KEYS, 'gnn_projection')
    nonconsecutive = frame['target_year'].ne(frame['input_year'] + 1)
    if nonconsecutive.any():
        examples = frame.loc[nonconsecutive, temporal_TRANSITION_KEYS].head(20)
        raise temporal_ProtocolError('GNN projection contains nonconsecutive transitions:\n' + examples.to_string(index=False))
    if not frame['prop_S_pred_next'].between(0, 1).all():
        raise temporal_ProtocolError('GNN temporal predictions must lie in zero to one.')
    if (frame['alpha_cal'] <= 0).any() or (frame['beta_cal'] <= 0).any():
        raise temporal_ProtocolError('GNN temporal alpha_cal and beta_cal must be positive.')
    target = reconstruction_targets.rename(columns={'Year': 'target_year', 'fold': 'target_fold', 'n_S': 'target_n_S', 'n_total': 'target_n_total', 'observed_prop_S': 'target_observed_prop_S'})
    target = target[['Country', 'target_year', 'Species', 'Family', 'target_fold', 'target_n_S', 'target_n_total', 'target_observed_prop_S']]
    joined = frame.merge(target, on=['Country', 'target_year', 'Species', 'Family'], how='left', validate='one_to_one', indicator=True)
    joined['exact_target_observation_available'] = joined['_merge'].eq('both')
    joined['fold_matches_target'] = joined['target_fold'].isna() | joined['fold'].eq(joined['target_fold'])
    disagreement = joined['exact_target_observation_available'] & ~joined['fold_matches_target']
    if disagreement.any():
        examples = joined.loc[disagreement, temporal_TRANSITION_KEYS + ['fold', 'target_fold']].head(20)
        raise temporal_ProtocolError('GNN projection and reconstruction folds disagree:\n' + examples.to_string(index=False))
    matching = joined.groupby(['target_year', 'next_year_in_data', 'exact_target_observation_available'], dropna=False, sort=True).size().reset_index(name='n_rows')
    matched = joined.loc[joined['next_year_in_data'] & joined['exact_target_observation_available']].copy()
    if matched.empty:
        raise temporal_ProtocolError('No GNN projection rows have an exact observed target cell.')
    matched['n_S'] = matched['target_n_S']
    matched['n_total'] = matched['target_n_total']
    matched['observed_prop_S'] = matched['n_S'].to_numpy(dtype=float) / matched['n_total'].to_numpy(dtype=float)
    matched['p_pred'] = matched['prop_S_pred_next']
    matched['alpha_pred'] = matched['alpha_cal']
    matched['beta_pred'] = matched['beta_cal']
    matched['model_name'] = 'GNN'
    matched['model_display_name'] = temporal_DISPLAY_NAMES['GNN']
    matched['source_table'] = str(projection_path)
    return (matched[temporal_TRANSITION_KEYS + ['fold', 'model_name', 'model_display_name', 'n_S', 'n_total', 'observed_prop_S', 'p_pred', 'alpha_pred', 'beta_pred', 'source_table']].copy(), matching)

def temporal_temporal_alpha_beta(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    candidate_pairs = [('bb_alpha', 'bb_beta'), ('alpha_cal', 'beta_cal'), ('bb_alpha_at_mean', 'bb_beta_at_mean')]
    for alpha_column, beta_column in candidate_pairs:
        if {alpha_column, beta_column}.issubset(frame.columns):
            alpha = pd.to_numeric(frame[alpha_column], errors='coerce')
            beta = pd.to_numeric(frame[beta_column], errors='coerce')
            if alpha.notna().all() and beta.notna().all():
                return (alpha.astype(float), beta.astype(float))
    phi_column = next((column for column in ['phi', 'phi_train', 'phi_fixed'] if column in frame.columns), None)
    if phi_column is None:
        raise temporal_ProtocolError('Temporal predictions need bb_alpha and bb_beta, alpha_cal and beta_cal, or a phi column.')
    phi = pd.to_numeric(frame[phi_column], errors='coerce')
    p = pd.to_numeric(frame['p_pred'], errors='coerce')
    if phi.isna().any() or p.isna().any():
        raise temporal_ProtocolError('Temporal phi or p_pred contains missing values.')
    return (p * phi, (1.0 - p) * phi)

def temporal_load_script04_temporal_predictions(path: Path, expected_folds: int, requested_models: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = temporal_TRANSITION_KEYS + ['model_name', 'n_S', 'n_total', 'prop_S', 'p_pred', 'fold']
    temporal_require_columns(frame, required, 'temporal_predictions')
    frame = temporal_clean_text(frame, ['Country', 'Species', 'Family', 'model_name'])
    if 'target_observed' in frame.columns:
        frame['target_observed'] = temporal_boolean_series(frame['target_observed'], 'target_observed')
        frame = frame.loc[frame['target_observed']].copy()
    frame = frame.loc[frame['model_name'].isin(requested_models)].copy()
    missing_models = sorted(set(requested_models) - set(frame['model_name'].unique()))
    if missing_models:
        raise temporal_ProtocolError(f"Temporal prediction table is missing models: {missing_models}. Available models: {sorted(pd.read_csv(path, usecols=['model_name'])['model_name'].astype(str).unique().tolist())}")
    for column in ['input_year', 'target_year', 'n_S', 'n_total', 'prop_S', 'p_pred', 'fold']:
        frame[column] = temporal_numeric_series(frame, column, 'temporal_predictions')
    frame['input_year'] = frame['input_year'].astype(int)
    frame['target_year'] = frame['target_year'].astype(int)
    frame['fold'] = temporal_normalize_fold_values(frame['fold'], expected_folds, 'temporal_predictions')
    nonconsecutive = frame['target_year'].ne(frame['input_year'] + 1)
    if nonconsecutive.any():
        examples = frame.loc[nonconsecutive, temporal_TRANSITION_KEYS].head(20)
        raise temporal_ProtocolError('Script 04 predictions contain nonconsecutive transitions:\n' + examples.to_string(index=False))
    if (frame['n_total'] <= 0).any():
        raise temporal_ProtocolError('Temporal prediction n_total must be positive.')
    if (frame['n_S'] < 0).any() or (frame['n_S'] > frame['n_total']).any():
        raise temporal_ProtocolError('Temporal prediction n_S must lie within n_total.')
    if not frame['prop_S'].between(0, 1).all():
        raise temporal_ProtocolError('Temporal observed proportions must lie in zero to one.')
    if not frame['p_pred'].between(0, 1).all():
        raise temporal_ProtocolError('Temporal predictions must lie in zero to one.')
    frame['observed_prop_S'] = temporal_canonical_observed_proportion(frame, source_column='prop_S', table_name='temporal_predictions')
    frame['alpha_pred'], frame['beta_pred'] = temporal_temporal_alpha_beta(frame)
    if (frame['alpha_pred'] <= 0).any() or (frame['beta_pred'] <= 0).any():
        raise temporal_ProtocolError('Temporal alpha and beta must be positive.')
    temporal_assert_unique(frame, temporal_TRANSITION_KEYS + ['model_name'], 'temporal_predictions')
    frame['model_display_name'] = frame['model_name'].map(temporal_DISPLAY_NAMES)
    frame['model_display_name'] = frame['model_display_name'].fillna(frame['model_name'])
    frame['source_table'] = str(path)
    return frame[temporal_TRANSITION_KEYS + ['fold', 'model_name', 'model_display_name', 'n_S', 'n_total', 'observed_prop_S', 'p_pred', 'alpha_pred', 'beta_pred', 'source_table']].copy()

def temporal_load_fold_assignments(path: Path, expected_folds: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == '.json':
        payload = json.loads(path.read_text(encoding='utf8'))
        rows: list[dict[str, object]] = []
        if not isinstance(payload, dict):
            raise temporal_ProtocolError('Fold JSON must contain a dictionary.')
        if all((isinstance(value, list) for value in payload.values())):
            for raw_fold, countries in payload.items():
                for country in countries:
                    rows.append({'Country': str(country).strip(), 'raw_fold': raw_fold})
        elif all((not isinstance(value, (dict, list)) for value in payload.values())):
            for country, raw_fold in payload.items():
                rows.append({'Country': str(country).strip(), 'raw_fold': raw_fold})
        else:
            raise temporal_ProtocolError('Fold JSON must map folds to country lists or countries to folds.')
        frame = pd.DataFrame(rows)
    else:
        source = pd.read_csv(path)
        temporal_require_columns(source, ['Country'], 'country_folds')
        fold_column = next((column for column in ['fold_model', 'colleague_fold', 'external_fold', 'fold'] if column in source.columns), None)
        if fold_column is None:
            raise temporal_ProtocolError('Country fold CSV needs fold_model, colleague_fold, external_fold, or fold.')
        frame = source[['Country', fold_column]].rename(columns={fold_column: 'raw_fold'})
        frame['Country'] = frame['Country'].astype(str).str.strip()
    frame = frame.drop_duplicates().copy()
    duplicated = frame.duplicated('Country', keep=False)
    if duplicated.any():
        raise temporal_ProtocolError('Country fold file assigns a country more than once.')
    frame['fold'] = temporal_normalize_fold_values(frame['raw_fold'], expected_folds, 'country_folds')
    return frame[['Country', 'fold']].copy()

def temporal_verify_fold_assignments(predictions: pd.DataFrame, assignments: pd.DataFrame, table_name: str) -> None:
    observed = predictions[['Country', 'fold']].drop_duplicates()
    if observed['Country'].duplicated().any():
        raise temporal_ProtocolError(f'{table_name} assigns at least one country to multiple folds.')
    comparison = observed.merge(assignments.rename(columns={'fold': 'expected_fold'}), on='Country', how='left', validate='one_to_one')
    if comparison['expected_fold'].isna().any():
        countries = comparison.loc[comparison['expected_fold'].isna(), 'Country'].head(20).tolist()
        raise temporal_ProtocolError(f'{table_name} contains countries absent from the fold file: {countries}')
    bad = comparison['fold'].ne(comparison['expected_fold'].astype(int))
    if bad.any():
        raise temporal_ProtocolError(f'{table_name} disagrees with the supplied fold file:\n' + comparison.loc[bad].head(20).to_string(index=False))

def temporal_key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[temporal_TRANSITION_KEYS])

def temporal_create_common_transition_table(gnn: pd.DataFrame, temporal: pd.DataFrame, requested_models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_frames: dict[str, pd.DataFrame] = {'GNN': gnn}
    for model_name in requested_models:
        source_frames[model_name] = temporal.loc[temporal['model_name'].eq(model_name)].copy()
    common_keys: pd.MultiIndex | None = None
    coverage_rows: list[dict[str, object]] = []
    for model_name, frame in source_frames.items():
        model_keys = temporal_key_index(frame)
        common_keys = model_keys if common_keys is None else common_keys.intersection(model_keys)
        coverage_rows.append({'model_name': model_name, 'model_display_name': temporal_DISPLAY_NAMES.get(model_name, model_name), 'n_source_transition_cells': int(len(frame)), 'n_source_countries': int(frame['Country'].nunique()), 'source_table': str(frame['source_table'].iloc[0])})
    if common_keys is None or len(common_keys) == 0:
        raise temporal_ProtocolError('The GNN and temporal models have no common transition cells.')
    common_key_frame = common_keys.to_frame(index=False)
    canonical = common_key_frame.merge(gnn, on=temporal_TRANSITION_KEYS, how='left', validate='one_to_one').rename(columns={'fold': 'fold_canonical', 'n_S': 'n_S_canonical', 'n_total': 'n_total_canonical', 'observed_prop_S': 'observed_prop_S_canonical'})
    standardized_parts: list[pd.DataFrame] = []
    for model_name, frame in source_frames.items():
        selected = common_key_frame.merge(frame, on=temporal_TRANSITION_KEYS, how='left', validate='one_to_one')
        comparison = selected.merge(canonical[temporal_TRANSITION_KEYS + ['fold_canonical', 'n_S_canonical', 'n_total_canonical', 'observed_prop_S_canonical']], on=temporal_TRANSITION_KEYS, how='left', validate='one_to_one')
        if comparison['fold'].ne(comparison['fold_canonical']).any():
            examples = comparison.loc[comparison['fold'].ne(comparison['fold_canonical']), temporal_TRANSITION_KEYS + ['fold', 'fold_canonical']].head(20)
            raise temporal_ProtocolError(f'Fold assignment disagrees for model {model_name}:\n' + examples.to_string(index=False))
        for outcome_column in ['n_S', 'n_total']:
            canonical_column = f'{outcome_column}_canonical'
            close = np.isclose(comparison[outcome_column].to_numpy(dtype=float), comparison[canonical_column].to_numpy(dtype=float), rtol=0.0, atol=temporal_COUNT_TOLERANCE)
            if not close.all():
                examples = comparison.loc[~close, temporal_TRANSITION_KEYS + [outcome_column, canonical_column]].head(20)
                raise temporal_ProtocolError(f'Observed count {outcome_column} disagrees for model {model_name}:\n' + examples.to_string(index=False))
        comparison['observed_prop_S_canonical'] = comparison['n_S_canonical'].to_numpy(dtype=float) / comparison['n_total_canonical'].to_numpy(dtype=float)
        standardized_parts.append(comparison[temporal_TRANSITION_KEYS + ['fold_canonical', 'n_S_canonical', 'n_total_canonical', 'observed_prop_S_canonical', 'model_name', 'model_display_name', 'p_pred', 'alpha_pred', 'beta_pred', 'source_table']].rename(columns={'fold_canonical': 'fold', 'n_S_canonical': 'n_S', 'n_total_canonical': 'n_total', 'observed_prop_S_canonical': 'observed_prop_S'}))
    standardized = pd.concat(standardized_parts, ignore_index=True, sort=False)
    expected_models = ['GNN'] + requested_models
    expected_rows = len(common_keys) * len(expected_models)
    if len(standardized) != expected_rows:
        raise temporal_ProtocolError(f'Expected {expected_rows} standardized rows, found {len(standardized)}.')
    for row in coverage_rows:
        row['n_common_transition_cells'] = int(len(common_keys))
        row['fraction_retained'] = float(len(common_keys) / row['n_source_transition_cells']) if row['n_source_transition_cells'] > 0 else np.nan
    return (standardized, pd.DataFrame(coverage_rows))

def temporal_beta_binomial_nll(n_s: np.ndarray, n_total: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    k = np.asarray(n_s, dtype=float)
    n = np.asarray(n_total, dtype=float)
    a = np.maximum(np.asarray(alpha, dtype=float), temporal_EPS)
    b = np.maximum(np.asarray(beta, dtype=float), temporal_EPS)
    return -(gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0) + betaln(k + a, n - k + b) - betaln(a, b))

def temporal_compute_metrics(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        raise temporal_ProtocolError('Cannot calculate metrics on an empty table.')
    observed = frame['observed_prop_S'].to_numpy(dtype=float)
    predicted = np.clip(frame['p_pred'].to_numpy(dtype=float), temporal_EPS, 1.0 - temporal_EPS)
    n_s = frame['n_S'].to_numpy(dtype=float)
    n_total = frame['n_total'].to_numpy(dtype=float)
    alpha = frame['alpha_pred'].to_numpy(dtype=float)
    beta = frame['beta_pred'].to_numpy(dtype=float)
    error = observed - predicted
    absolute_error = np.abs(error)
    squared_error = error ** 2
    weights = np.sqrt(n_total)
    nll = temporal_beta_binomial_nll(n_s, n_total, alpha, beta)
    return {'n_cells': int(len(frame)), 'n_tests': int(round(float(n_total.sum()))), 'n_countries': int(frame['Country'].nunique()), 'weighted_mae': float(np.average(absolute_error, weights=weights)), 'weighted_rmse': float(math.sqrt(np.average(squared_error, weights=weights))), 'beta_binomial_nll_per_test': float(np.sum(nll) / np.sum(n_total)), 'unweighted_mae': float(np.mean(absolute_error)), 'unweighted_rmse': float(math.sqrt(np.mean(squared_error))), 'mean_signed_error_observed_minus_predicted': float(np.mean(error))}

def temporal_assign_evaluation_task(frame: pd.DataFrame, historical_max_year: int, vault_years: list[int]) -> pd.DataFrame:
    out = frame.copy()
    out['evaluation_task'] = np.select([out['target_year'].le(historical_max_year), out['target_year'].isin(vault_years)], [temporal_HISTORICAL_TASK, temporal_VAULT_TASK], default='outside_configured_tasks')
    outside = out['evaluation_task'].eq('outside_configured_tasks')
    if outside.any():
        years = sorted(out.loc[outside, 'target_year'].unique().tolist())
        print('Ignoring transitions outside historical and vault tasks. Years: ' + str(years))
        out = out.loc[~outside].copy()
    expected = {temporal_HISTORICAL_TASK, temporal_VAULT_TASK}
    missing = sorted(expected - set(out['evaluation_task'].unique()))
    if missing:
        raise temporal_ProtocolError(f'No common temporal rows are available for tasks: {missing}')
    return out

def temporal_calculate_fold_metrics(common: pd.DataFrame, expected_folds: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = common.groupby(['evaluation_task', 'model_name', 'model_display_name', 'fold'], sort=True)
    for (evaluation_task, model_name, display_name, fold), group in grouped:
        row: dict[str, object] = {'evaluation_family': 'temporal_projection', 'evaluation_task': evaluation_task, 'model_name': model_name, 'model_display_name': display_name, 'fold': int(fold), 'year_min': int(group['target_year'].min()), 'year_max': int(group['target_year'].max()), 'n_years': int(group['target_year'].nunique())}
        row.update(temporal_compute_metrics(group))
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(['evaluation_task', 'model_name', 'fold']).reset_index(drop=True)
    counts = result.groupby(['evaluation_task', 'model_name'])['fold'].nunique()
    bad = counts.loc[counts.ne(expected_folds)]
    if not bad.empty:
        raise temporal_ProtocolError(f'Every temporal model and task must contain all configured folds. Observed: {bad.to_dict()}')
    support = result.groupby(['evaluation_task', 'fold']).agg(n_cells_values=('n_cells', lambda values: values.nunique()), n_tests_values=('n_tests', lambda values: values.nunique()), n_countries_values=('n_countries', lambda values: values.nunique()))
    if support.ne(1).any(axis=None):
        raise temporal_ProtocolError('Temporal models do not have identical support inside every task and fold.')
    return result

def temporal_summarize_folds(fold_metrics: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = fold_metrics.groupby(['evaluation_family', 'evaluation_task', 'model_name', 'model_display_name'], sort=True)
    for keys, group in grouped:
        evaluation_family, evaluation_task, model_name, display_name = keys
        row: dict[str, object] = {'evaluation_family': evaluation_family, 'evaluation_task': evaluation_task, 'model_name': model_name, 'model_display_name': display_name, 'n_folds': int(group['fold'].nunique()), 'n_cells_total': int(group['n_cells'].sum()), 'n_tests_total': int(group['n_tests'].sum()), 'n_countries_total': int(group['n_countries'].sum()), 'year_min': int(group['year_min'].min()), 'year_max': int(group['year_max'].max()), 'error_weighting': 'sqrt_n_total', 'uncertainty_definition': 'sample standard deviation across fold metrics', 'standard_deviation_ddof': 1, 'transition_set_rule': 'exact intersection across GNN and all requested models'}
        for metric in temporal_PRIMARY_METRICS:
            values = pd.to_numeric(group[metric], errors='coerce').dropna()
            mean = float(values.mean()) if not values.empty else np.nan
            sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f'{metric}_mean'] = mean
            row[f'{metric}_sd'] = sd
            row[f'{metric}_mean_plus_minus_sd'] = f'{mean:.6f} ± {sd:.6f}' if np.isfinite(mean) and np.isfinite(sd) else ''
        rows.append(row)
    order = {name: index for index, name in enumerate(model_order)}
    result = pd.DataFrame(rows)
    result['model_order'] = result['model_name'].map(order)
    result = result.sort_values(['evaluation_task', 'model_order']).drop(columns='model_order')
    return result.reset_index(drop=True)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create completion and temporal model comparison tables using "
            "exactly common cells and transitions."
        )
    )
    parser.add_argument(
        "--gnn_reconstruction_path",
        type=Path,
        required=True,
        help="GNN reconstruction_observed_loo CSV.",
    )
    parser.add_argument(
        "--completion_predictions_path",
        type=Path,
        required=True,
        help=(
            "country_generalization_leave_one_out_predictions.csv produced "
            "by script 02."
        ),
    )
    parser.add_argument(
        "--gnn_projection_path",
        type=Path,
        required=True,
        help="GNN projection_next_year CSV.",
    )
    parser.add_argument(
        "--temporal_predictions_path",
        type=Path,
        required=True,
        help=(
            "temporal_observed_predictions_all_external_tests.csv produced "
            "by script 04."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--temporal_models",
        nargs="+",
        default=[
            "species_family_mean",
            "rolling_mean_k",
            "temporal_residual_encoder",
        ],
        help=(
            "Temporal model names retained from the script 04 table. By "
            "default all four temporal baselines and the temporal residual "
            "encoder are compared with the GNN on the exact same transitions."
        ),
    )
    parser.add_argument(
        "--country_folds_path",
        type=Path,
        default=None,
        help="Optional JSON or CSV country fold assignment for verification.",
    )
    parser.add_argument("--historical_max_year", type=int, default=2022)
    parser.add_argument(
        "--vault_years",
        nargs="+",
        type=int,
        default=[2023, 2024],
    )
    parser.add_argument("--expected_folds", type=int, default=5)
    parser.add_argument(
        "--report_scope",
        choices=["vault", "all"],
        default="vault",
        help=(
            "vault writes only the 2023 and 2024 comparison. all also writes "
            "the historical task through 2022."
        ),
    )
    parser.add_argument(
        "--save_common_predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    args.temporal_models = list(dict.fromkeys(args.temporal_models))
    args.vault_years = sorted(set(args.vault_years))
    if not args.temporal_models:
        raise ValueError("At least one temporal model is required.")
    if not args.vault_years:
        raise ValueError("At least one vault year is required.")
    if min(args.vault_years) <= args.historical_max_year:
        raise ValueError("Every vault year must follow historical_max_year.")
    if args.expected_folds < 2:
        raise ValueError("expected_folds must be at least two.")
    return args


def filter_report_scope(
    frame: pd.DataFrame,
    report_scope: str,
    vault_task: str,
) -> pd.DataFrame:
    if report_scope == "all":
        return frame.reset_index(drop=True)
    if "evaluation_task" not in frame.columns:
        return frame.reset_index(drop=True)
    return frame.loc[frame["evaluation_task"].eq(vault_task)].reset_index(
        drop=True
    )


def run_completion(args: argparse.Namespace) -> dict[str, Path]:
    gnn = completion_load_gnn_reconstruction(
        args.gnn_reconstruction_path,
        args.expected_folds,
    )
    completion = completion_load_completion_predictions(
        args.completion_predictions_path,
        args.expected_folds,
    )

    if args.country_folds_path is not None:
        assignments = completion_load_fold_assignments(
            args.country_folds_path,
            args.expected_folds,
        )
        completion_verify_fold_assignments(
            gnn,
            assignments,
            "gnn_reconstruction",
        )
        completion_verify_fold_assignments(
            completion,
            assignments,
            "completion_predictions",
        )

    common, coverage = completion_create_common_cell_table(gnn, completion)
    common = completion_assign_evaluation_task(
        common,
        historical_max_year=args.historical_max_year,
        vault_years=args.vault_years,
    )
    fold_metrics = completion_calculate_fold_metrics(
        common,
        args.expected_folds,
    )
    summary = completion_summarize_folds(fold_metrics)

    common = filter_report_scope(
        common,
        args.report_scope,
        completion_VAULT_TASK,
    )
    fold_metrics = filter_report_scope(
        fold_metrics,
        args.report_scope,
        completion_VAULT_TASK,
    )
    summary = filter_report_scope(
        summary,
        args.report_scope,
        completion_VAULT_TASK,
    )

    summary_path = args.output_dir / "completion_common_cells_metrics.csv"
    fold_path = args.output_dir / "completion_common_cells_metrics_by_fold.csv"
    coverage_path = args.output_dir / "completion_common_cells_coverage.csv"
    prediction_path = args.output_dir / "completion_common_cells_predictions.csv"

    summary.to_csv(summary_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    if args.save_common_predictions:
        common.to_csv(prediction_path, index=False)

    print("\nCompletion comparison")
    print("=====================")
    print(
        summary[
            [
                "evaluation_task",
                "model_display_name",
                "n_cells_total",
                "weighted_mae_mean_plus_minus_sd",
                "weighted_rmse_mean_plus_minus_sd",
                "beta_binomial_nll_per_test_mean_plus_minus_sd",
            ]
        ].to_string(index=False)
    )

    return {
        "summary": summary_path,
        "fold": fold_path,
        "coverage": coverage_path,
        "predictions": prediction_path,
    }


def run_temporal(args: argparse.Namespace) -> dict[str, Path]:
    targets = temporal_load_gnn_reconstruction_targets(
        args.gnn_reconstruction_path,
        args.expected_folds,
    )
    gnn, matching = temporal_load_gnn_temporal_predictions(
        args.gnn_projection_path,
        targets,
        args.expected_folds,
    )
    temporal_predictions = temporal_load_script04_temporal_predictions(
        args.temporal_predictions_path,
        args.expected_folds,
        args.temporal_models,
    )

    if args.country_folds_path is not None:
        assignments = temporal_load_fold_assignments(
            args.country_folds_path,
            args.expected_folds,
        )
        temporal_verify_fold_assignments(
            gnn,
            assignments,
            "gnn_temporal",
        )
        temporal_verify_fold_assignments(
            temporal_predictions,
            assignments,
            "script04_temporal",
        )

    common, coverage = temporal_create_common_transition_table(
        gnn,
        temporal_predictions,
        args.temporal_models,
    )
    common = temporal_assign_evaluation_task(
        common,
        historical_max_year=args.historical_max_year,
        vault_years=args.vault_years,
    )
    fold_metrics = temporal_calculate_fold_metrics(
        common,
        args.expected_folds,
    )
    model_order = ["GNN"] + args.temporal_models
    summary = temporal_summarize_folds(fold_metrics, model_order)

    common = filter_report_scope(
        common,
        args.report_scope,
        temporal_VAULT_TASK,
    )
    fold_metrics = filter_report_scope(
        fold_metrics,
        args.report_scope,
        temporal_VAULT_TASK,
    )
    summary = filter_report_scope(
        summary,
        args.report_scope,
        temporal_VAULT_TASK,
    )

    summary_path = args.output_dir / "temporal_common_transitions_metrics.csv"
    fold_path = args.output_dir / "temporal_common_transitions_metrics_by_fold.csv"
    coverage_path = args.output_dir / "temporal_common_transitions_coverage.csv"
    prediction_path = args.output_dir / "temporal_common_transitions_predictions.csv"
    matching_path = args.output_dir / "gnn_temporal_target_matching.csv"

    summary.to_csv(summary_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    matching.to_csv(matching_path, index=False)
    if args.save_common_predictions:
        common.to_csv(prediction_path, index=False)

    print("\nTemporal comparison")
    print("===================")
    print(
        summary[
            [
                "evaluation_task",
                "model_display_name",
                "n_cells_total",
                "weighted_mae_mean_plus_minus_sd",
                "weighted_rmse_mean_plus_minus_sd",
                "beta_binomial_nll_per_test_mean_plus_minus_sd",
            ]
        ].to_string(index=False)
    )

    return {
        "summary": summary_path,
        "fold": fold_path,
        "coverage": coverage_path,
        "predictions": prediction_path,
        "matching": matching_path,
    }


def main() -> None:
    args = parse_args()
    print(f"Script version: {UNIFIED_SCRIPT_VERSION}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completion_paths = run_completion(args)
    temporal_paths = run_temporal(args)

    completion_summary = pd.read_csv(completion_paths["summary"])
    temporal_summary = pd.read_csv(temporal_paths["summary"])
    combined = pd.concat(
        [completion_summary, temporal_summary],
        ignore_index=True,
        sort=False,
    )
    combined_path = args.output_dir / "all_common_protocol_metrics.csv"
    combined.to_csv(combined_path, index=False)

    metadata = {
        "script_version": UNIFIED_SCRIPT_VERSION,
        "report_scope": args.report_scope,
        "historical_max_year": int(args.historical_max_year),
        "vault_years": args.vault_years,
        "expected_folds": int(args.expected_folds),
        "completion_cell_rule": "exact intersection across all completion models",
        "temporal_transition_rule": (
            "exact intersection across GNN and all requested temporal models"
        ),
        "canonical_outcome_rule": "n_S divided by n_total",
        "error_weighting": "sqrt_n_total",
        "standard_deviation_ddof": 1,
        "prospective_2025_evaluated": False,
    }
    metadata_path = args.output_dir / "common_protocol_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf8")

    print("\nMain tables")
    print("===========")
    print(completion_paths["summary"])
    print(temporal_paths["summary"])
    print(combined_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
