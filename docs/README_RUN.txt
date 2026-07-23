AMR dashboard with GNN integration

Dashboard pages

1. Heterogeneity
   Uses the calibrated GNN observed reconstruction. The page reads
   graph_model/sampling_returns_cells.csv, derived from
   reconstruction_observed_loo.csv.

2. Data completion
   Uses landscape_prediction/landscape_predictions.csv from the residual
   completion encoder and overlays the GNN outputs copied to
   graph_model/reconstruction_observed_loo.csv and
   graph_model/reconstruction_imputed.csv.

3. Jump analysis
   Uses temporal_prediction/temporal_jump_candidate_rankings.csv. Script 08
   first creates the temporal residual jump rows, then appends GNN down_prob
   and up_prob rows from observed_direction_forecast.csv. Target years 2023,
   2024 and 2025 are retained.

Required GNN notebook exports

reconstruction_observed_loo.csv
reconstruction_imputed.csv
observed_direction_forecast.csv

Optional GNN notebook exports copied when present

projection_next_year.csv
README.json
threshold_diagnostic.csv
gated_alerts_summary.csv
precision_at_k_movers_only.csv
uncertainty_decile_table.csv
uncertainty_ordering.csv
uncertainty_ordering_by_n_stratum.csv
sampling_effort_summary.csv
alert_coverage_summary.csv

Run

Place 08_data_for_dashboard.py beside
02_country_generalization_completion_encoders.py and run:

python scripts/08_data_for_dashboard.py \
  --input_path data/amr_disaggregated.csv \
  --completion_output_dir results/reconstruction/country_generalization \
  --jump_output_dir results/temporal_prediction/jumps_analysis \
  --gnn_export_dir path/to/gnn_reconstruction_export \
  --output_dir dashboard

The old argument name --graph_export_dir remains accepted as an alias.

The completion output directory must contain

country_generalization_leave_one_out_predictions.csv
country_generalization_fold_assignment.csv
country_generalization_historical_phi_train.csv
fold_01 through fold_05, each with the saved residual encoder checkpoint

The jump output directory must contain

temporal_residual_jump_predictions_all_external_tests.csv
temporal_residual_future_jump_predictions.csv

Copy index.html, styles.css, app.js, gnn_sampling.js and
temporal_dashboard_patch.js into the dashboard root beside the generated data
folders.

Serve the directory through a local server:

cd dashboard
python -m http.server 8000

Then open http://localhost:8000 in a browser.

Verified uploaded GNN run

Observed reconstruction rows: 32252
Imputed reconstruction rows: 37304
Countries: 55
Species: 16
Drug families: 17
External folds: 5
Direction rows: 32252
GNN direction rows retained for dashboard years after conversion to long form:
17152
Target years: 2023, 2024, 2025
