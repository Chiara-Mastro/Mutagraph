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

