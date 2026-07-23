# Mutagraph

Mutagraph is an open source research pipeline and interactive dashboard for antimicrobial resistance surveillance.

The project was developed for the 2026 Vivli AMR Surveillance Data Challenge. It treats each country and year as a partially observed antimicrobial susceptibility landscape, then asks three connected questions:

1. When do additional isolates still improve precision, and when does biological heterogeneity dominate further sampling?

2. Can the missing pathogen species and antimicrobial class combinations be reconstructed from the cells that were measured in the same surveillance snapshot?

3. Which pathogen species and antimicrobial class combinations should be prioritized because their susceptibility may change in the following year?

Mutagraph answers these questions with two complementary model families. A residual snapshot encoder learns a compact representation of each country year susceptibility matrix. A graph neural network represents the same snapshot as a bipartite pathogen and antimicrobial graph. The dashboard brings their outputs together without assuming that the two models use the same internal scale.

## What the dashboard provides

The dashboard can be found here: https://chiara-mastro.github.io/Mutagraph/ .


It provides three views.

### Heterogeneity and sampling returns

The first view asks when more isolates are still expected to help.

The graph model predicts a Beta Binomial concentration parameter for every observed cell. This parameter describes residual isolate level heterogeneity after accounting for the predicted mean susceptibility. It is converted into an irreducible latent rate floor and compared with the current isolate count.

Cells are grouped into three practical regimes:

1. Sampling sensitive, where additional isolates may still reduce uncertainty materially

2. Diminishing returns, where more isolates help but the expected gain is becoming smaller

3. Heterogeneity dominated, where further sampling is unlikely to reduce uncertainty substantially because variation between isolates, hospitals, populations, periods, and clonal backgrounds dominates the finite sample component

These categories are decision support summaries, not universal adequacy thresholds.

### Data completion

The second view reconstructs a selected country year susceptibility matrix.

Observed cells are shown beside model estimates for both measured and unmeasured pathogen species and antimicrobial class combinations. Intrinsic resistance combinations remain explicitly separated from ordinary missingness.

For observed cells, performance is evaluated through cellwise leave one out reconstruction. The target cell is removed from the snapshot before prediction, so its own outcome cannot be used as context.

For genuinely unobserved cells, the models use all available observed cells in the selected country year snapshot and predict the missing combination with frozen parameters from the external country fold.

### Directional change prioritization

The third view ranks combinations that may show a downward or upward change in susceptibility in the following year.

Downward change corresponds to possible resistance emergence. Upward change corresponds to possible susceptibility recovery.

The temporal residual model and the graph model are displayed as separate prioritization lists. Their raw sigmoid outputs are not treated as directly comparable epidemiological probabilities. The dashboard therefore emphasizes rank, overlap between the two top lists, current susceptibility, isolate support, and each model specific alert rule.

### Country and year generalization

The same external countries are evaluated in 2023 and 2024. These years are excluded from model fitting, so this setting tests simultaneous transfer to unseen countries and unseen future years.

Prospective outputs for 2025 are generated only after model selection. They are not used as observed evaluation targets.

## Model architectures

### Transferable Pathogen Species and Antimicrobial Class prior

A smoothed empirical prior estimates susceptibility from patterns learned across training countries, falling back from the exact Pathogen Species and Antimicrobial Class combination to broader class, species, and global averages when support is limited.

### Residual snapshot encoder

The residual snapshot encoder represents each Country Year as a pooled latent state built from observed susceptibility cells and learns how that local surveillance landscape deviates from the transferable prior, while accounting for isolate support and Beta Binomial overdispersion.

### Graph neural network

The graph neural network represents each Country Year as a bipartite graph linking Pathogen Species and Antimicrobial Classes, uses separate susceptible and resistant pathogen channels with country conditioned message passing, and predicts susceptibility, heterogeneity, missing cells, and directional change rankings.

## Installation

Create and activate a Python environment, then install the packages listed in `requirements.txt`.

The dashboard loads Plotly and Papa Parse in the browser. Serve the `dashboard` directory through a local web server, then open the local address shown by that server.

## Data access and governance

The Vivli source datasets are not included in this repository.

Access to Vivli data is governed by the applicable Vivli Data Use Agreement. The open source license for this repository applies only to the original software. It does not grant permission to redistribute Vivli datasets, restricted source tables, trained artifacts whose release is not permitted, or third party content.

Before publishing checkpoints or row level prediction files, confirm that their release is allowed by the applicable data use terms and challenge guidance.

## Interpretation notes

1. Completion estimates are model based reconstructions and do not replace microbiological testing.

2. Heterogeneity estimates describe variation that may remain even with large isolate counts.

3. Sampling regimes summarize expected marginal returns under the calibrated graph model. They are not universal adequacy standards.

4. Jump scores support prioritization. They should not be interpreted as calibrated event probabilities unless calibration is established separately.

5. Prospective rankings require local epidemiological review before operational use.

6. Dashboard outputs support surveillance planning and do not constitute clinical guidance.

## License

Original source code in this repository is licensed under the Apache License, Version 2.0.

The license does not cover:

1. Vivli source data

2. Third party datasets

3. Third party software and assets

4. Files whose redistribution is restricted by contract or data use terms

Each dependency and external resource remains subject to its own license and terms.

## Acknowledgements

This project was developed using data accessed through the Vivli AMR Register for the 2026 Vivli AMR Surveillance Data Challenge.

## Disclaimer

Mutagraph is a research prototype. It is not a clinical decision system. Its outputs require validation, domain review, and interpretation in the context of local surveillance practices.
