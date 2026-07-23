const LANDSCAPE_CSV_PATH = "landscape_prediction/landscape_predictions.csv";
const GRAPH_MODEL_PATHS = {
  observed: "graph_model/reconstruction_observed_loo.csv",
  imputed: "graph_model/reconstruction_imputed.csv"
};

const PAL_PINK = ["#590d22", "#c9184a", "#ff4d6d", "#ff8fa3", "#ffb3c1", "#ffccd5", "#ffe0e6"];
const PAL_BLUE = ["#013a63", "#2a6f97", "#468faf", "#61a5c2", "#89c2d9", "#a9d6e5"];

const MODEL_DEFINITIONS = [
  { key: "p_residual_encoder", label: "Snapshot completion model", intervalLabel: "Exported interval" },
  { key: "p_graph_completion", label: "Graph completion model", intervalLabel: "95% latent rate interval" },
  { key: "p_species_family_prior", label: "Pathogen Species and Antimicrobial Class prior", intervalLabel: "Exported interval" },
  { key: "p_snapshot_encoder", label: "Snapshot encoder", intervalLabel: "Exported interval" },
  { key: "p_country_aware_reference", label: "Country aware reference", intervalLabel: "Exported interval" }
];

const completionState = {
  rows: [],
  allSpecies: [],
  allFamilies: [],
  countries: [],
  selectedSpecies: null,
  selectedFamily: null
};

const graphState = {
  loaded: false
};

function byId(id) {
  return document.getElementById(id);
}

function asNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function asText(value) {
  return String(value ?? "").trim();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normaliseStatus(value) {
  const text = asText(value).toLowerCase().replaceAll(" ", "_");
  if (["observed", "measured"].includes(text)) return "observed";
  if (["to_impute", "impute", "missing", "unobserved"].includes(text)) return "to_impute";
  if (["intrinsic_resistance", "do_not_impute_intrinsic", "intrinsic"].includes(text)) return "intrinsic_resistance";
  return text || "to_impute";
}

function uniqueSorted(values, numeric = false) {
  const clean = Array.from(new Set(values.filter((value) => value !== null && value !== undefined && value !== "")));
  return clean.sort(numeric ? (a, b) => Number(a) - Number(b) : (a, b) => String(a).localeCompare(String(b)));
}

function median(values) {
  const clean = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!clean.length) return null;
  const middle = Math.floor(clean.length / 2);
  return clean.length % 2 ? clean[middle] : (clean[middle - 1] + clean[middle]) / 2;
}

function quantile(values, probability) {
  const clean = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!clean.length) return null;
  const position = (clean.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return clean[lower];
  return clean[lower] + (clean[upper] - clean[lower]) * (position - lower);
}

function formatProbability(value, digits = 3) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
}

function formatSigned(value, digits = 3) {
  if (!Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${Number(value).toFixed(digits)}`;
}

function formatPercent(value, digits = 1) {
  return Number.isFinite(value) ? `${(100 * value).toFixed(digits)}%` : "—";
}

function firstNumber(raw, names) {
  for (const name of names) {
    const value = asNumber(raw[name]);
    if (value !== null) return value;
  }
  return null;
}

function modelKeyFromName(value) {
  const text = asText(value).toLowerCase();
  if (text.includes("graph") && text.includes("completion")) return "p_graph_completion";
  if (text.includes("residual")) return "p_residual_encoder";
  if (text.includes("snapshot")) return "p_snapshot_encoder";
  if (text.includes("country") && text.includes("reference")) return "p_country_aware_reference";
  if (text.includes("species") && text.includes("family")) return "p_species_family_prior";
  if (text === "prior") return "p_species_family_prior";
  return null;
}

function cleanLandscapeRows(rawRows) {
  const merged = new Map();

  rawRows.forEach((raw) => {
    const country = asText(raw.Country ?? raw.country);
    const year = firstNumber(raw, ["Year", "year"]);
    const species = asText(raw.Species ?? raw.species);
    const family = asText(raw.Family ?? raw.family);
    if (!country || year === null || !species || !family) return;

    const key = [country, year, species, family].join("||");
    const existing = merged.get(key) ?? {
      country,
      year: Number(year),
      species,
      family,
      status: normaliseStatus(raw.status ?? raw.Status),
      nS: firstNumber(raw, ["n_S", "n_s"]),
      nTotal: firstNumber(raw, ["n_total", "n_tests"]),
      observed: firstNumber(raw, ["prop_S", "observed_prop_S", "pS_observed"]),
      fold: firstNumber(raw, ["fold", "external_fold"]),
      predictionContext: asText(raw.prediction_context),
      baselineSource: asText(raw.baseline_source),
      completionAvailable: asText(raw.completion_prediction_available).toLowerCase() === "true" || Number(raw.completion_prediction_available) === 1,
      exclusionReason: asText(raw.completion_exclusion_reason),
      contextCells: firstNumber(raw, ["context_n_cells"]),
      contextTests: firstNumber(raw, ["context_n_tests"]),
      p_species_family_prior: null,
      p_species_family_prior_q05: null,
      p_species_family_prior_q95: null,
      p_snapshot_encoder: null,
      p_snapshot_encoder_q05: null,
      p_snapshot_encoder_q95: null,
      p_residual_encoder: null,
      p_residual_encoder_q05: null,
      p_residual_encoder_q95: null,
      p_country_aware_reference: null,
      p_country_aware_reference_q05: null,
      p_country_aware_reference_q95: null,
      p_graph_completion: null,
      p_graph_completion_q025: null,
      p_graph_completion_q975: null,
      graphNuCal: null,
      graphFloorSd: null,
      graphEffortStatus: "",
      graphNStop2Nu: null,
      graphNStop5Nu: null,
      graphNStop10Nu: null,
      graphNOverNu: null,
      graphContextEdges: null,
      graphFoldModel: null,
      graphLambda: null,
      deltaLogitResidual: firstNumber(raw, ["delta_logit_residual_encoder"])
    };

    existing.status = normaliseStatus(raw.status ?? raw.Status ?? existing.status);
    existing.nS = firstNumber(raw, ["n_S", "n_s"]) ?? existing.nS;
    existing.nTotal = firstNumber(raw, ["n_total", "n_tests"]) ?? existing.nTotal;
    existing.observed = firstNumber(raw, ["prop_S", "observed_prop_S", "pS_observed"]) ?? existing.observed;

    MODEL_DEFINITIONS.forEach((model) => {
      existing[model.key] = firstNumber(raw, [model.key]) ?? existing[model.key];
      existing[`${model.key}_q05`] = firstNumber(raw, [`${model.key}_q05`, `${model.key}_lower`]) ?? existing[`${model.key}_q05`];
      existing[`${model.key}_q95`] = firstNumber(raw, [`${model.key}_q95`, `${model.key}_upper`]) ?? existing[`${model.key}_q95`];
    });

    const longModelKey = modelKeyFromName(raw.model_name ?? raw.method ?? raw.model);
    const longPrediction = firstNumber(raw, ["p_pred", "pred_p", "prediction", "pred_prob"]);
    if (longModelKey && longPrediction !== null) {
      existing[longModelKey] = longPrediction;
      existing[`${longModelKey}_q05`] = firstNumber(raw, ["predictive_prop_q05", "beta_latent_q05", "p_q05"]) ?? existing[`${longModelKey}_q05`];
      existing[`${longModelKey}_q95`] = firstNumber(raw, ["predictive_prop_q95", "beta_latent_q95", "p_q95"]) ?? existing[`${longModelKey}_q95`];
    }

    merged.set(key, existing);
  });

  return Array.from(merged.values());
}

async function fetchTextOptional(path) {
  try {
    const response = await fetch(path);
    if (!response.ok) return null;
    return await response.text();
  } catch (_error) {
    return null;
  }
}

async function fetchCsvOptional(path) {
  const text = await fetchTextOptional(path);
  if (text === null) return [];
  const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
  return parsed.data ?? [];
}

async function fetchJsonOptional(path) {
  const text = await fetchTextOptional(path);
  if (text === null) return null;
  try {
    return JSON.parse(text);
  } catch (_error) {
    return null;
  }
}

function mergeGraphCompletionRows(rawRows, status) {
  const lookup = new Map(
    completionState.rows.map((row) => [
      [row.country, row.year, row.species, row.family].join("||"),
      row
    ])
  );

  let matched = 0;
  rawRows.forEach((raw) => {
    const country = asText(raw.Country ?? raw.country);
    const year = firstNumber(raw, ["Year", "year"]);
    const species = asText(raw.Species ?? raw.species);
    const family = asText(raw.Family ?? raw.family);
    if (!country || year === null || !species || !family) return;

    const row = lookup.get([country, Number(year), species, family].join("||"));
    if (!row) return;

    row.p_graph_completion = firstNumber(raw, ["prop_S_pred"]);
    row.p_graph_completion_q025 = firstNumber(raw, ["ci_lo_cal"]);
    row.p_graph_completion_q975 = firstNumber(raw, ["ci_hi_cal"]);
    row.graphNuCal = firstNumber(raw, ["nu_cal"]);
    row.graphFloorSd = firstNumber(raw, ["floor_sd_cal"]);
    row.graphEffortStatus = asText(raw.effort_status);
    row.graphNStop2Nu = firstNumber(raw, ["n_stop_2nu"]);
    row.graphNStop5Nu = firstNumber(raw, ["n_stop_5nu"]);
    row.graphNStop10Nu = firstNumber(raw, ["n_stop_10nu"]);
    row.graphNOverNu = firstNumber(raw, ["n_total_over_nu_cal"]);
    row.graphContextEdges = firstNumber(raw, ["n_context_edges"]);
    row.graphFoldModel = firstNumber(raw, ["fold_model"]);
    row.graphLambda = firstNumber(raw, ["lambda_applied"]);

    if (status === "observed") {
      row.observed = firstNumber(raw, ["prop_S_observed"]) ?? row.observed;
      row.nS = firstNumber(raw, ["n_S"]) ?? row.nS;
      row.nTotal = firstNumber(raw, ["n_total"]) ?? row.nTotal;
    }
    matched += 1;
  });
  return matched;
}

async function loadGraphModelData() {
  const [observed, imputed] = await Promise.all([
    fetchCsvOptional(GRAPH_MODEL_PATHS.observed),
    fetchCsvOptional(GRAPH_MODEL_PATHS.imputed)
  ]);

  if (!observed.length && !imputed.length) {
    updateStatus("graphStatusBadge", "GNN completion exports not loaded", true);
    return false;
  }

  const matchedObserved = mergeGraphCompletionRows(observed, "observed");
  const matchedImputed = mergeGraphCompletionRows(imputed, "to_impute");
  graphState.loaded = true;

  const completionCount = matchedObserved + matchedImputed;
  updateStatus(
    "graphStatusBadge",
    `${completionCount.toLocaleString()} GNN completion cells loaded`
  );
  return true;
}

function updateStatus(id, message, isError = false) {
  const element = byId(id);
  if (!element) return;
  element.classList.toggle("isError", isError);
  element.innerHTML = `<span class="statusDot"></span>${escapeHtml(message)}`;
}

function fillSelect(id, values, preferred = null, labelFunction = null) {
  const select = byId(id);
  if (!select) return;
  const current = select.value;
  select.innerHTML = values.map((value) => {
    const label = labelFunction ? labelFunction(value) : value;
    return `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`;
  }).join("");
  if (values.map(String).includes(String(current))) select.value = current;
  else if (preferred !== null && values.map(String).includes(String(preferred))) select.value = preferred;
  else if (values.length) select.value = values[0];
}

function currentCountry() {
  return byId("completionCountrySelect")?.value ?? "";
}

function currentYear() {
  return Number(byId("completionYearSelect")?.value);
}

function currentModelKey() {
  return byId("completionModelSelect")?.value ?? "p_residual_encoder";
}

function currentModelDefinition() {
  return MODEL_DEFINITIONS.find((model) => model.key === currentModelKey()) ?? MODEL_DEFINITIONS[0];
}

function availableCompletionModels() {
  return MODEL_DEFINITIONS.filter((model) => completionState.rows.some((row) => Number.isFinite(row[model.key])));
}

function snapshotRows() {
  const country = currentCountry();
  const year = currentYear();
  return completionState.rows.filter((row) => row.country === country && row.year === year);
}

function snapshotMap(rows) {
  return new Map(rows.map((row) => [[row.species, row.family].join("||"), row]));
}

function cellFromMap(map, species, family) {
  return map.get([species, family].join("||")) ?? {
    country: currentCountry(),
    year: currentYear(),
    species,
    family,
    status: "to_impute",
    observed: null,
    nTotal: null,
    nS: null,
    p_residual_encoder: null,
    p_species_family_prior: null,
    predictionContext: "not_exported",
    exclusionReason: "cell_missing_from_export"
  };
}

function ensureSelectedCell(rows) {
  const map = snapshotMap(rows);
  const current = completionState.selectedSpecies && completionState.selectedFamily
    ? cellFromMap(map, completionState.selectedSpecies, completionState.selectedFamily)
    : null;
  if (current && (map.has([current.species, current.family].join("||")) || completionState.allSpecies.includes(current.species))) return;

  const preferred = rows.find((row) => row.status === "to_impute" && Number.isFinite(row[currentModelKey()]))
    ?? rows.find((row) => row.status === "observed")
    ?? rows[0];
  completionState.selectedSpecies = preferred?.species ?? completionState.allSpecies[0] ?? null;
  completionState.selectedFamily = preferred?.family ?? completionState.allFamilies[0] ?? null;
}

function selectedCell(rows) {
  const map = snapshotMap(rows);
  return cellFromMap(map, completionState.selectedSpecies, completionState.selectedFamily);
}

function syncCompletionCellSelectors() {
  const speciesSelect = byId("completionSpeciesSelect");
  const familySelect = byId("completionFamilySelect");
  if (speciesSelect && completionState.selectedSpecies !== null) {
    speciesSelect.value = completionState.selectedSpecies;
  }
  if (familySelect && completionState.selectedFamily !== null) {
    familySelect.value = completionState.selectedFamily;
  }
}

function selectCompletionCell(species, family) {
  completionState.selectedSpecies = species;
  completionState.selectedFamily = family;
  syncCompletionCellSelectors();
  redrawCompletion();
}

function matrix(rows, valueFunction) {
  const map = snapshotMap(rows);
  return completionState.allSpecies.map((species) => completionState.allFamilies.map((family) => valueFunction(cellFromMap(map, species, family))));
}

function matrixText(rows, textFunction) {
  const map = snapshotMap(rows);
  return completionState.allSpecies.map((species) => completionState.allFamilies.map((family) => textFunction(cellFromMap(map, species, family))));
}

function plotHeight(minimum = 440, maximum = 900) {
  return Math.max(minimum, Math.min(maximum, 150 + completionState.allSpecies.length * 24));
}

function abbreviatedPathogenSpecies(value) {
  const text = asText(value);
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length >= 2) {
    const remainder = words.slice(1).join(" ");
    const shortened = `${words[0].charAt(0)}. ${remainder}`;
    return shortened.length <= 15 ? shortened : `${shortened.slice(0, 14)}.`;
  }
  return text.length <= 14 ? text : `${text.slice(0, 13)}.`;
}

function abbreviatedAntimicrobialClass(value) {
  const text = asText(value);
  if (text.length <= 13) return text;
  return `${text.slice(0, 12)}.`;
}

function heatmapLayout(height, compact = false) {
  const xaxis = {
    side: "bottom",
    tickangle: compact ? -35 : -48,
    automargin: !compact,
    fixedrange: true
  };
  const yaxis = {
    autorange: "reversed",
    automargin: !compact,
    fixedrange: true
  };

  if (compact) {
    xaxis.tickmode = "array";
    xaxis.tickvals = completionState.allFamilies;
    xaxis.ticktext = completionState.allFamilies.map(abbreviatedAntimicrobialClass);
    xaxis.tickfont = { size: 8 };
    yaxis.tickmode = "array";
    yaxis.tickvals = completionState.allSpecies;
    yaxis.ticktext = completionState.allSpecies.map(abbreviatedPathogenSpecies);
    yaxis.tickfont = { size: 8 };
  }

  return {
    height,
    autosize: true,
    margin: compact ? { l: 92, r: 14, t: 10, b: 82 } : { l: 190, r: 68, t: 20, b: 130 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#ffffff",
    font: { family: "Inter, system-ui, sans-serif", color: "#102d44", size: compact ? 8 : 11 },
    xaxis,
    yaxis,
    hoverlabel: { bgcolor: "#ffffff", bordercolor: "#bdd2de", font: { color: "#102d44" } }
  };
}

function observedMatrixHeight() {
  if (window.matchMedia("(max-width: 760px)").matches) return 500;
  if (window.matchMedia("(max-width: 1180px)").matches) return 450;
  return window.matchMedia("(min-width: 1500px)").matches ? 430 : 410;
}

function completionMatrixHeight() {
  if (window.matchMedia("(max-width: 760px)").matches) return 580;
  if (window.matchMedia("(max-width: 1120px)").matches) return 560;

  const side = document.querySelector(".completionSideCompact");
  const card = document.querySelector(".completionMainWide");
  const plot = byId("completedSnapshotPlot");
  if (!side || !card || !plot) return 540;

  const sideHeight = side.getBoundingClientRect().height;
  const cardTop = card.getBoundingClientRect().top;
  const plotTop = plot.getBoundingClientRect().top;
  const contentBeforePlot = Math.max(135, plotTop - cardTop);
  return Math.max(500, Math.min(680, Math.round(sideHeight - contentBeforePlot - 18)));
}

function susceptibilityScale() {
  return [
    [0, PAL_BLUE[5]],
    [0.2, PAL_BLUE[4]],
    [0.4, PAL_BLUE[3]],
    [0.6, PAL_BLUE[2]],
    [0.8, PAL_BLUE[1]],
    [1, PAL_BLUE[0]]
  ];
}

function attachMatrixClick(plotId) {
  const plot = byId(plotId);
  if (!plot || typeof plot.on !== "function") return;
  plot.removeAllListeners?.("plotly_click");
  plot.on("plotly_click", (event) => {
    const point = event.points?.[0];
    if (!point) return;
    const species = point.y;
    const family = point.x;
    if (species && family) selectCompletionCell(species, family);
  });
}

function intrinsicOverlay(rows, compact = false) {
  const intrinsic = rows.filter((row) => row.status === "intrinsic_resistance");
  return {
    type: "scatter",
    mode: "markers",
    x: intrinsic.map((row) => row.family),
    y: intrinsic.map((row) => row.species),
    marker: { symbol: "x", size: compact ? 7 : 10, color: "#590d22", line: { width: 2 } },
    hovertemplate: "Intrinsic resistance, excluded<extra></extra>",
    showlegend: false
  };
}

function observedOutlineOverlay(rows, compact = false) {
  const observed = rows.filter((row) => row.status === "observed");
  return {
    type: "scatter",
    mode: "markers",
    x: observed.map((row) => row.family),
    y: observed.map((row) => row.species),
    marker: { symbol: "square-open", size: compact ? 6 : 9, color: "#102d44", line: { width: compact ? 1 : 1.6 } },
    hoverinfo: "skip",
    showlegend: false
  };
}

function selectedOverlay(compact = false) {
  if (!completionState.selectedSpecies || !completionState.selectedFamily) return null;
  return {
    type: "scatter",
    mode: "markers",
    x: [completionState.selectedFamily],
    y: [completionState.selectedSpecies],
    marker: { symbol: "square-open", size: compact ? 12 : 17, color: "#ff4d6d", line: { width: 3 } },
    hoverinfo: "skip",
    showlegend: false
  };
}

function drawObservedPlot(rows) {
  const z = matrix(rows, (cell) => cell.status === "observed" ? cell.observed : null);
  const text = matrixText(rows, (cell) => cell.status === "observed"
    ? `<b>${escapeHtml(cell.species)} × ${escapeHtml(cell.family)}</b><br>Measured pS: ${formatProbability(cell.observed)}<br>Tests: ${cell.nTotal ?? "—"}`
    : `<b>${escapeHtml(cell.species)} × ${escapeHtml(cell.family)}</b><br>${cell.status === "intrinsic_resistance" ? "Intrinsic resistance" : "Not measured"}`);
  const traces = [{
    type: "heatmap",
    x: completionState.allFamilies,
    y: completionState.allSpecies,
    z,
    text,
    hovertemplate: "%{text}<extra></extra>",
    zmin: 0,
    zmax: 1,
    colorscale: susceptibilityScale(),
    showscale: false,
    xgap: 1,
    ygap: 1,
    hoverongaps: true
  }, selectedOverlay(true)].filter(Boolean);
  const observedHeight = observedMatrixHeight();
  Plotly.react("observedSnapshotPlot", traces, heatmapLayout(observedHeight, true), { responsive: true, displayModeBar: false });
  attachMatrixClick("observedSnapshotPlot");
}

function drawErrorPlot(rows, targetId = "completedSnapshotPlot") {
  const model = currentModelDefinition();
  const modelKey = model.key;
  const errors = rows
    .filter((row) => row.status === "observed" && Number.isFinite(row.observed) && Number.isFinite(row[modelKey]))
    .map((row) => row.observed - row[modelKey]);
  const bound = Math.min(0.50, Math.max(0.10, quantile(errors.map(Math.abs), 0.95) ?? 0.10));
  const z = matrix(rows, (cell) => cell.status === "observed" && Number.isFinite(cell.observed) && Number.isFinite(cell[modelKey]) ? cell.observed - cell[modelKey] : null);
  const text = matrixText(rows, (cell) => {
    const error = cell.status === "observed" && Number.isFinite(cell.observed) && Number.isFinite(cell[modelKey]) ? cell.observed - cell[modelKey] : null;
    return `<b>${escapeHtml(cell.species)} × ${escapeHtml(cell.family)}</b><br>Measured pS: ${formatProbability(cell.observed)}<br>${escapeHtml(model.label)}: ${formatProbability(cell[modelKey])}<br>Measured minus predicted: ${formatSigned(error)}<br>Absolute error: ${formatProbability(Number.isFinite(error) ? Math.abs(error) : null)}`;
  });

  byId("completionPlotTitle").textContent = `${model.label}: measured minus predicted susceptibility`;
  byId("completionPlotNote").textContent = "Only observed cells can be evaluated. Purple indicates that predicted susceptibility was too high, orange that it was too low, and green close agreement.";
  const legend = byId("completionPredictionLegend");
  if (legend) legend.hidden = true;

  const traces = [{
    type: "heatmap",
    x: completionState.allFamilies,
    y: completionState.allSpecies,
    z,
    text,
    hovertemplate: "%{text}<extra></extra>",
    zmin: -bound,
    zmax: bound,
    zmid: 0,
    colorscale: [
      [0.00, "#54278f"],
      [0.22, "#9e77c6"],
      [0.42, "#e4d5ef"],
      [0.50, "#d9f0d3"],
      [0.58, "#fee8c8"],
      [0.78, "#fdbb84"],
      [1.00, "#b35806"]
    ],
    showscale: true,
    colorbar: {
      title: { text: "Measured minus<br>predicted pS", side: "right", font: { size: 11 } },
      tickmode: "array",
      tickvals: [-bound, 0, bound],
      ticktext: [
        `−${bound.toFixed(2)}<br>prediction too high`,
        "0<br>close agreement",
        `+${bound.toFixed(2)}<br>prediction too low`
      ],
      thickness: 15,
      len: 0.78,
      x: 1.02,
      tickfont: { size: 9 }
    },
    xgap: 1,
    ygap: 1,
    hoverongaps: true
  }, selectedOverlay(false)].filter(Boolean);
  const layout = heatmapLayout(completionMatrixHeight(), false);
  layout.margin.r = 150;
  Plotly.react(targetId, traces, layout, { responsive: true, displayModeBar: false });
  attachMatrixClick(targetId);
}

function drawCompletedPlot(rows) {
  const view = byId("completionViewSelect")?.value ?? "prediction";
  if (view === "error") {
    drawErrorPlot(rows, "completedSnapshotPlot");
    return;
  }

  const model = currentModelDefinition();
  const z = matrix(rows, (cell) => cell.status !== "intrinsic_resistance" ? cell[model.key] : null);
  const text = matrixText(rows, (cell) => {
    const status = cell.status === "observed" ? "Measured cell, predicted leave one out" : cell.status === "to_impute" ? "Completion target" : "Intrinsic resistance";
    return `<b>${escapeHtml(cell.species)} × ${escapeHtml(cell.family)}</b><br>${status}<br>${escapeHtml(model.label)}: ${formatProbability(cell[model.key])}<br>Measured pS: ${formatProbability(cell.observed)}`;
  });

  byId("completionPlotTitle").textContent = `${model.label}: predicted susceptibility`;
  byId("completionPlotNote").textContent = `All ${completionState.allSpecies.length} Pathogen Species and all ${completionState.allFamilies.length} Antimicrobial Classes are shown. Empty predictions remain visible as unfilled completion targets.`;
  const legend = byId("completionPredictionLegend");
  if (legend) legend.hidden = false;

  const traces = [{
    type: "heatmap",
    x: completionState.allFamilies,
    y: completionState.allSpecies,
    z,
    text,
    hovertemplate: "%{text}<extra></extra>",
    zmin: 0,
    zmax: 1,
    colorscale: susceptibilityScale(),
    colorbar: { title: "Predicted pS", thickness: 14, len: 0.72 },
    xgap: 1,
    ygap: 1,
    hoverongaps: true
  }, observedOutlineOverlay(rows), intrinsicOverlay(rows), selectedOverlay(false)].filter(Boolean);

  Plotly.react("completedSnapshotPlot", traces, heatmapLayout(completionMatrixHeight(), false), { responsive: true, displayModeBar: false });
  attachMatrixClick("completedSnapshotPlot");
}

function drawCompletionMetrics(rows) {
  const modelKey = currentModelKey();
  const observed = rows.filter((row) => row.status === "observed");
  const toImpute = rows.filter((row) => row.status === "to_impute");
  const predictedToImpute = toImpute.filter((row) => Number.isFinite(row[modelKey]));
  const comparable = observed.filter((row) => Number.isFinite(row.observed) && Number.isFinite(row[modelKey]));
  const weightSum = comparable.reduce((sum, row) => sum + (row.nTotal ?? 0), 0);
  const weightedMae = weightSum > 0
    ? comparable.reduce((sum, row) => sum + Math.abs(row.observed - row[modelKey]) * (row.nTotal ?? 0), 0) / weightSum
    : null;

  byId("completionObservedCount").textContent = observed.length.toLocaleString();
  byId("completionImputeCount").textContent = toImpute.length.toLocaleString();
  byId("completionPredictedCount").textContent = predictedToImpute.length.toLocaleString();
  byId("completionWeightedMae").textContent = formatProbability(weightedMae);
}

function intervalBounds(cell, key) {
  const low95 = cell[`${key}_q025`];
  const high95 = cell[`${key}_q975`];
  if (Number.isFinite(low95) && Number.isFinite(high95)) {
    return { low: low95, high: high95, level: "95%" };
  }
  const low90 = cell[`${key}_q05`];
  const high90 = cell[`${key}_q95`];
  if (Number.isFinite(low90) && Number.isFinite(high90)) {
    return { low: low90, high: high90, level: "90%" };
  }
  return null;
}

function intervalFor(cell, key) {
  const bounds = intervalBounds(cell, key);
  return bounds ? `${bounds.level}: ${formatProbability(bounds.low)} to ${formatProbability(bounds.high)}` : "—";
}

function detailItem(label, value, className = "") {
  return `<div class="compactDetailRow ${className}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}


function drawCellDetails(rows) {
  const cell = selectedCell(rows);
  const model = currentModelDefinition();
  const error = Number.isFinite(cell.observed) && Number.isFinite(cell[model.key]) ? cell.observed - cell[model.key] : null;
  const statusLabel = cell.status === "observed" ? "Observed" : cell.status === "to_impute" ? "To impute" : "Intrinsic resistance";
  const statusClass = cell.status === "observed" ? "statusObserved" : cell.status === "to_impute" ? "statusImpute" : "statusIntrinsic";

  byId("completionControlNote").textContent = `${cell.species} × ${cell.family}`;
  byId("completionCellDetails").innerHTML = `
    <div class="cellHeading">
      <div><h4>${escapeHtml(cell.species)}</h4><p>${escapeHtml(cell.family)} · ${escapeHtml(cell.country)} · ${cell.year}</p></div>
      <span class="statusTag ${statusClass}">${statusLabel}</span>
    </div>
    <div class="compactDetailList">
      ${detailItem("Measured pS", formatProbability(cell.observed))}
      ${detailItem(model.label, formatProbability(cell[model.key]))}
      ${detailItem("Measured minus predicted", formatSigned(error), Number.isFinite(error) && error < 0 ? "pinkValue" : "orangeValue")}
      ${detailItem("Selected interval", intervalFor(cell, model.key))}
      ${detailItem("Tests", cell.nTotal === null ? "—" : Math.round(cell.nTotal).toLocaleString())}
      ${detailItem("Context cells", cell.contextCells ?? cell.graphContextEdges ?? "—")}
    </div>
  `;

  const prior = cell.p_species_family_prior;
  byId("completionModelTableBody").innerHTML = availableCompletionModels().map((definition) => {
    const value = cell[definition.key];
    const difference = Number.isFinite(value) && Number.isFinite(prior) ? value - prior : null;
    return `<tr>
      <td>${escapeHtml(definition.label)}</td>
      <td>${formatProbability(value)}</td>
      <td>${formatSigned(difference)}</td>
      <td>${intervalFor(cell, definition.key)}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="4">No model estimates are available for this cell.</td></tr>`;

}


function updateCompletionYearOptions() {
  const country = currentCountry() || completionState.countries[0];
  const years = uniqueSorted(completionState.rows.filter((row) => row.country === country).map((row) => row.year), true);
  const preferred = years.includes(2024) ? 2024 : years[years.length - 1];
  fillSelect("completionYearSelect", years.map(String), String(preferred));
}

function redrawCompletion() {
  const rows = snapshotRows();
  if (!rows.length) return;
  ensureSelectedCell(rows);
  syncCompletionCellSelectors();
  drawCompletionMetrics(rows);
  drawObservedPlot(rows);
  drawCellDetails(rows);
  drawCompletedPlot(rows);
}

function setupTaskChooser() {
  document.querySelectorAll(".taskButton").forEach((button) => {
    button.addEventListener("click", () => {
      const task = button.dataset.task;
      document.querySelectorAll(".taskButton").forEach((candidate) => {
        const active = candidate === button;
        candidate.classList.toggle("isActive", active);
        candidate.setAttribute("aria-selected", active ? "true" : "false");
      });
      document.querySelectorAll(".taskPanel").forEach((panel) => {
        const active = panel.dataset.panel === task;
        panel.classList.toggle("isActive", active);
        panel.hidden = !active;
      });
      window.dispatchEvent(new CustomEvent("amrTaskChange", { detail: { task } }));
      if (task === "completion" && completionState.rows.length) {
        requestAnimationFrame(redrawCompletion);
      }
    });
  });
}

function setupCompletionListeners() {
  byId("completionCountrySelect").addEventListener("change", () => {
    updateCompletionYearOptions();
    redrawCompletion();
  });
  byId("completionYearSelect").addEventListener("change", redrawCompletion);
  byId("completionSpeciesSelect").addEventListener("change", () => {
    completionState.selectedSpecies = byId("completionSpeciesSelect").value;
    redrawCompletion();
  });
  byId("completionFamilySelect").addEventListener("change", () => {
    completionState.selectedFamily = byId("completionFamilySelect").value;
    redrawCompletion();
  });
  byId("completionModelSelect").addEventListener("change", redrawCompletion);
  byId("completionViewSelect")?.addEventListener("change", redrawCompletion);
}

async function loadLandscapeData() {
  try {
    const response = await fetch(LANDSCAPE_CSV_PATH);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
    const rows = cleanLandscapeRows(parsed.data);
    if (!rows.length) throw new Error("No valid completion rows were found");

    completionState.rows = rows;
    completionState.allSpecies = uniqueSorted(rows.map((row) => row.species));
    completionState.allFamilies = uniqueSorted(rows.map((row) => row.family));
    completionState.countries = uniqueSorted(rows.map((row) => row.country));

    await loadGraphModelData();

    fillSelect("completionCountrySelect", completionState.countries, completionState.countries.includes("Italy") ? "Italy" : completionState.countries[0]);
    updateCompletionYearOptions();
    fillSelect("completionSpeciesSelect", completionState.allSpecies);
    fillSelect("completionFamilySelect", completionState.allFamilies);
    const models = availableCompletionModels();
    fillSelect("completionModelSelect", models.map((model) => model.key), models.some((model) => model.key === "p_residual_encoder") ? "p_residual_encoder" : models[0]?.key, (key) => MODEL_DEFINITIONS.find((model) => model.key === key)?.label ?? key);

    updateStatus("dataStatusBadge", `${rows.length.toLocaleString()} completion cells loaded`);
    redrawCompletion();
    window.dispatchEvent(new CustomEvent("amrLandscapeReady", { detail: { species: completionState.allSpecies, families: completionState.allFamilies, countries: completionState.countries } }));
  } catch (error) {
    console.error(error);
    updateStatus("dataStatusBadge", "Completion CSV not loaded", true);
    ["observedSnapshotPlot", "completedSnapshotPlot"].forEach((id) => {
      const element = byId(id);
      if (element) element.innerHTML = `<div class="emptyState">Could not load <code>${LANDSCAPE_CSV_PATH}</code><br>${escapeHtml(error.message || error)}</div>`;
    });
  }
}

window.AMRDashboard = {
  byId,
  asNumber,
  asText,
  escapeHtml,
  uniqueSorted,
  median,
  quantile,
  formatProbability,
  formatSigned,
  formatPercent,
  updateStatus,
  getUniverse: () => ({
    species: [...completionState.allSpecies],
    families: [...completionState.allFamilies],
    countries: [...completionState.countries]
  }),
  getLandscapeRows: () => completionState.rows.map((row) => ({ ...row }))
};


document.addEventListener("DOMContentLoaded", () => {
  setupTaskChooser();
  setupCompletionListeners();
  loadLandscapeData();

  let completionResizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(completionResizeTimer);
    completionResizeTimer = setTimeout(() => {
      const panel = byId("taskCompletion");
      if (completionState.rows.length && panel && !panel.hidden) redrawCompletion();
    }, 140);
  });
});
