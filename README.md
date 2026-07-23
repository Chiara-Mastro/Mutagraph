# Mutagraph

Mutagraph is an open source research pipeline and interactive dashboard for antimicrobial resistance surveillance.

The project was developed for the 2026 Vivli AMR Surveillance Data Challenge. It treats each country and year as a partially observed antimicrobial susceptibility landscape, then asks three connected questions:

1. When do additional isolates still improve precision, and when does biological heterogeneity dominate further sampling?

2. Can the missing pathogen species and antimicrobial class combinations be reconstructed from the cells that were measured in the same surveillance snapshot?

3. Which pathogen species and antimicrobial class combinations should be prioritized because their susceptibility may change in the following year?

Mutagraph answers these questions with two complementary model families. A residual snapshot encoder learns a compact representation of each country year susceptibility matrix. A graph neural network represents the same snapshot as a bipartite pathogen and antimicrobial graph. The dashboard brings their outputs together without assuming that the two models use the same internal scale.

## What the dashboard provides

The dashboard can be found here:
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

### Transferable pathogen species and antimicrobial class prior

The simplest model is a transferable empirical prior.

For each pathogen species and antimicrobial class combination, the prior pools susceptible and tested isolate counts across the training countries. A weak Beta one one contribution smooths the estimate away from exact zero and one when support is limited.

The model uses no information from the target country or target year. It therefore measures how much can be predicted from broad transferable regularities alone.

For a target cell, the prediction hierarchy is:

1. The matching pathogen species and antimicrobial class combination
2. The antimicrobial class estimate when the exact combination is unsupported
3. The pathogen species estimate when the class estimate is also unavailable
4. The overall training susceptibility mean

The prior plays two roles. First, it is a transparent benchmark for external country completion. Second, it provides the baseline logit used by the residual snapshot encoder. The neural model does not need to relearn the entire global susceptibility landscape. It only needs to learn how the current country year differs from this transferable reference.

### Residual snapshot encoder

The family of residual snapshot encoder models each country year as a partially observed collection of susceptibility cells.

Each observed context cell contributes:

1. A learned pathogen species embedding

2. A learned antimicrobial class embedding

3. The transferable baseline prediction for that combination

4. The observed deviation from the baseline

5. The number of tested isolates

These inputs are processed by a feed forward cell encoder. The resulting cell representations are pooled within the country year snapshot, with isolate support contributing to the pooling weights. A second network transforms the pooled summary into a low dimensional latent country year state.

This latent state is intended to capture the shared local structure of the snapshot. For example, it can represent whether susceptibility in a particular country and year is systematically above or below the transferable prior, and whether that departure differs across parts of the pathogen species and antimicrobial class landscape.

This residual formulation is important. The model is not asked to predict susceptibility from nothing. It begins from a stable transferable prior and learns only the country year specific departure that can be inferred from the observed cells.

The model is trained with a Beta Binomial likelihood. In addition to the predicted mean susceptibility, it learns a fold level concentration parameter that accounts for residual overdispersion in the observed isolate counts.

### Graph neural network

The graph model represents every country year snapshot as a bipartite graph connecting pathogen species and antimicrobial classes.

Observed susceptibility cells are graph edges. Missing combinations are absent edges rather than negative examples. This distinction is essential because an unobserved combination means that the pair was not tested, not that susceptibility or resistance is zero.

The architecture uses dual channel message passing.

Each pathogen has two coupled latent states:

1. A wild type state

2. A mutant or resistant state derived as a learned displacement from the wild type state

Each antimicrobial class has its own latent state.

Susceptible isolate counts and non susceptible isolate counts route information through the two pathogen channels separately. Incoming messages are aggregated by their mean rather than their sum, which reduces sensitivity to the total volume of testing in a country. Node states are updated across repeated message passing steps with gated recurrent units.

The graph also includes country conditioning. A hypernetwork transforms a country level health system embedding into the matrices used for message transformation. Because the conditioning is based on a country description rather than a country identifier, the model can generate country specific transformations for a country that was absent from training.

The graph output head is structured around the geometry of the learned representation.

The predicted susceptibility mean is anchored on the alignment between:

1. The learned displacement from the pathogen wild type state to the resistant state

2. The antimicrobial class embedding

A neural residual corrects this geometric signal when the simple alignment is insufficient.

The model also predicts a Beta Binomial concentration parameter. This parameter is calibrated after fitting and is interpreted as isolate level heterogeneity within a surveillance cell. It supports both uncertainty intervals and the sampling return analysis shown in the dashboard.

For completion, an observed target edge is removed before reconstruction. A genuinely missing edge is predicted from the full observed graph of the country year.

For directional change, the graph model produces separate downward and upward scores. These scores support ranking and model specific alerts. They are not assumed to be calibrated probabilities.


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
