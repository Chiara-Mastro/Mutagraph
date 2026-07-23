const GNN_SAMPLING_PATHS = {
  cells: "graph_model/sampling_returns_cells.csv",
  legacyCells: "graph_model/sampling_sufficiency_cells.csv",
  excluded: "graph_model/sampling_returns_excluded.csv",
  legacyExcluded: "graph_model/sampling_sufficiency_excluded.csv",
  fallbackObserved: "graph_model/reconstruction_observed_loo.csv"
};

const SAMPLING_COLORS = {
  sampling_sensitive: "#c9184a",
  diminishing_returns: "#61a5c2",
  heterogeneity_dominated: "#013a63"
};

const SAMPLING_LABELS = {
  sampling_sensitive: "Sampling sensitive",
  diminishing_returns: "Diminishing returns",
  heterogeneity_dominated: "Heterogeneity dominated"
};

const SAMPLING_EXPLANATIONS = {
  sampling_sensitive: "More isolates may reduce sampling uncertainty.",
  diminishing_returns: "Expected gains become smaller.",
  heterogeneity_dominated: "Latent heterogeneity contributes more than sample size."
};

const SAMPLING_ALIASES = {
  under_sampled: "sampling_sensitive",
  sampling_sensitive: "sampling_sensitive",
  at_limit: "diminishing_returns",
  diminishing_returns: "diminishing_returns",
  over_sampled: "heterogeneity_dominated",
  heterogeneity_dominated: "heterogeneity_dominated"
};

const samplingState = {
  rows: [],
  excludedRows: [],
  loaded: false,
  listenersReady: false
};

function sById(id) {
  return document.getElementById(id);
}

function sText(value) {
  return String(value ?? "").trim();
}

function sNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function sEscape(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function sUnique(values, numeric = false) {
  const clean = Array.from(new Set(values.filter((value) => value !== null && value !== undefined && value !== "")));
  return clean.sort(numeric ? (a, b) => Number(a) - Number(b) : (a, b) => String(a).localeCompare(String(b)));
}

function sFillSelect(id, values, allLabel, preferred = "") {
  const select = sById(id);
  if (!select) return;
  const current = select.value;
  const options = [{ value: "", label: allLabel }].concat(values.map((value) => ({ value: String(value), label: String(value) })));
  select.innerHTML = options.map((option) => `<option value="${sEscape(option.value)}">${sEscape(option.label)}</option>`).join("");
  if (options.some((option) => option.value === current)) select.value = current;
  else if (options.some((option) => option.value === String(preferred))) select.value = String(preferred);
  else select.value = "";
}

function sUpdateStatus(message, isError = false) {
  const badge = sById("samplingStatusBadge");
  if (!badge) return;
  badge.classList.toggle("isError", isError);
  badge.innerHTML = `<span class="statusDot"></span>${sEscape(message)}`;
}

async function sFetchCsv(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${path}`);
  const text = await response.text();
  return Papa.parse(text, { header: true, skipEmptyLines: true }).data ?? [];
}

async function sFetchCsvOptional(path) {
  try {
    return await sFetchCsv(path);
  } catch (_error) {
    return [];
  }
}

function deriveSamplingCategory(nTotal, nuCal) {
  if (!Number.isFinite(nTotal) || nTotal <= 0 || !Number.isFinite(nuCal) || nuCal <= 0) return null;
  if (nTotal < 2 * nuCal) return "sampling_sensitive";
  if (nTotal <= 10 * nuCal) return "diminishing_returns";
  return "heterogeneity_dominated";
}

function normaliseSamplingCategory(value) {
  return SAMPLING_ALIASES[sText(value).toLowerCase()] ?? null;
}

function cleanSamplingRows(rawRows, allowDerivation = false) {
  const rows = [];
  const excluded = [];
  const keys = new Set();

  rawRows.forEach((raw) => {
    const country = sText(raw.Country ?? raw.country);
    const year = sNumber(raw.Year ?? raw.year);
    const species = sText(raw.Species ?? raw.species);
    const family = sText(raw.Family ?? raw.family);
    const nTotal = sNumber(raw.n_total ?? raw.n_tests);
    const nuCal = sNumber(raw.nu_cal);
    let category = normaliseSamplingCategory(raw.sampling_category ?? raw.sampling_regime);
    if (!category && allowDerivation) category = deriveSamplingCategory(nTotal, nuCal);

    if (!country || year === null || !species || !family) return;
    const key = [country, year, species, family].join("||");
    if (keys.has(key)) throw new Error(`Duplicate GNN sampling cell: ${key}`);
    keys.add(key);

    if (!Number.isFinite(nTotal) || nTotal <= 0 || !Number.isFinite(nuCal) || nuCal <= 0 || !category) {
      excluded.push({ country, year, species, family });
      return;
    }

    rows.push({
      country,
      year: Number(year),
      species,
      family,
      nTotal,
      nuCal,
      category
    });
  });

  return { rows, excluded };
}

function cleanExcludedRows(rawRows) {
  return rawRows.map((raw) => ({
    country: sText(raw.Country ?? raw.country),
    year: sNumber(raw.Year ?? raw.year),
    species: sText(raw.Species ?? raw.species),
    family: sText(raw.Family ?? raw.family)
  })).filter((row) => row.country && row.year !== null && row.species && row.family);
}

function currentSamplingFilters() {
  return {
    year: sNumber(sById("samplingYearSelect")?.value),
    country: sText(sById("samplingCountrySelect")?.value),
    species: sText(sById("samplingSpeciesSelect")?.value),
    family: sText(sById("samplingFamilySelect")?.value),
    category: sText(sById("samplingCategorySelect")?.value)
  };
}

function samplingRowMatches(row, includeCategory = true) {
  const filters = currentSamplingFilters();
  return (!Number.isFinite(filters.year) || row.year === filters.year)
    && (!filters.country || row.country === filters.country)
    && (!filters.species || row.species === filters.species)
    && (!filters.family || row.family === filters.family)
    && (!includeCategory || !filters.category || row.category === filters.category);
}

function filteredSamplingRows() {
  return samplingState.rows.filter((row) => samplingRowMatches(row, true));
}

function filteredExcludedRows() {
  return samplingState.excludedRows.filter((row) => samplingRowMatches(row, false));
}

function aggregateCountries(rows) {
  const groups = new Map();
  rows.forEach((row) => {
    if (!groups.has(row.country)) groups.set(row.country, []);
    groups.get(row.country).push(row);
  });

  return Array.from(groups, ([country, cells]) => {
    const counts = {
      sampling_sensitive: cells.filter((row) => row.category === "sampling_sensitive").length,
      diminishing_returns: cells.filter((row) => row.category === "diminishing_returns").length,
      heterogeneity_dominated: cells.filter((row) => row.category === "heterogeneity_dominated").length
    };
    const nCells = cells.length;
    return {
      country,
      nCells,
      nTests: cells.reduce((total, row) => total + row.nTotal, 0),
      shares: {
        sampling_sensitive: counts.sampling_sensitive / nCells,
        diminishing_returns: counts.diminishing_returns / nCells,
        heterogeneity_dominated: counts.heterogeneity_dominated / nCells
      }
    };
  });
}

function sortCountryRows(rows) {
  const mode = sById("samplingCountrySortSelect")?.value ?? "sensitive";
  const value = (row) => {
    if (mode === "heterogeneity") return row.shares.heterogeneity_dominated;
    if (mode === "cells") return row.nCells;
    return row.shares.sampling_sensitive;
  };
  return [...rows].sort((a, b) => value(b) - value(a) || a.country.localeCompare(b.country));
}

function drawSamplingCountryBars(aggregates) {
  const element = sById("samplingCountryBarPlot");
  if (!element) return;
  if (!aggregates.length) {
    element.innerHTML = '<div class="samplingEmpty">No eligible cells match the current filters.</div>';
    return;
  }

  const ordered = sortCountryRows(aggregates);
  const categories = ["sampling_sensitive", "diminishing_returns", "heterogeneity_dominated"];
  const traces = categories.map((category) => ({
    type: "bar",
    orientation: "h",
    name: SAMPLING_LABELS[category],
    x: ordered.map((row) => row.shares[category]),
    y: ordered.map((row) => row.country),
    marker: { color: SAMPLING_COLORS[category] },
    customdata: ordered.map((row) => [
      row.country,
      row.nCells,
      row.nTests,
      row.shares.sampling_sensitive,
      row.shares.diminishing_returns,
      row.shares.heterogeneity_dominated
    ]),
    hovertemplate:
      "<b>%{customdata[0]}</b><br>" +
      "Observed cells: %{customdata[1]:,}<br>" +
      "Isolate tests: %{customdata[2]:,.0f}<br>" +
      "Sampling sensitive: %{customdata[3]:.1%}<br>" +
      "Diminishing returns: %{customdata[4]:.1%}<br>" +
      "Heterogeneity dominated: %{customdata[5]:.1%}<br><br>" +
      `<b>${SAMPLING_LABELS[category]}</b><br>${SAMPLING_EXPLANATIONS[category]}` +
      "<extra></extra>"
  }));

  Plotly.react(element, traces, {
    barmode: "stack",
    height: Math.max(470, Math.min(1300, 145 + 27 * ordered.length)),
    margin: {
      l: Math.min(250, Math.max(125, 9 * Math.max(...ordered.map((row) => row.country.length)))),
      r: 25,
      t: 18,
      b: 70
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#ffffff",
    font: { family: "Inter, system-ui, sans-serif", color: "#102d44" },
    xaxis: {
      title: "Share of eligible observed GNN cells",
      range: [0, 1],
      tickformat: ".0%",
      gridcolor: "#e9f0f4"
    },
    yaxis: { autorange: "reversed", automargin: true },
    legend: { orientation: "h", y: -0.12 }
  }, { responsive: true, displayModeBar: false });
}

function updateSamplingMetrics(rows, aggregates) {
  const excluded = filteredExcludedRows();
  sById("samplingIncludedCount").textContent = rows.length.toLocaleString();
  sById("samplingExcludedCount").textContent = excluded.length.toLocaleString();
  sById("samplingCountryCount").textContent = aggregates.length.toLocaleString();
  sById("samplingTestCount").textContent = Math.round(rows.reduce((total, row) => total + row.nTotal, 0)).toLocaleString();

  const filters = currentSamplingFilters();
  const summary = [
    filters.year ? `Year ${filters.year}` : null,
    filters.country || null,
    filters.species || null,
    filters.family || null,
    filters.category ? SAMPLING_LABELS[filters.category] : null
  ].filter(Boolean);
  sById("samplingFilterSummary").textContent = summary.length ? summary.join(" · ") : "All eligible cells";
}

function redrawSamplingSection() {
  if (!samplingState.loaded) return;
  const rows = filteredSamplingRows();
  const aggregates = aggregateCountries(rows);
  updateSamplingMetrics(rows, aggregates);
  drawSamplingCountryBars(aggregates);
}

function initialiseSamplingControls() {
  const years = sUnique(samplingState.rows.map((row) => row.year), true).map(String);
  sFillSelect("samplingYearSelect", years, "All years", years.includes("2024") ? "2024" : years.at(-1));
  sFillSelect("samplingCountrySelect", sUnique(samplingState.rows.map((row) => row.country)), "All countries");
  sFillSelect("samplingSpeciesSelect", sUnique(samplingState.rows.map((row) => row.species)), "All pathogen species");
  sFillSelect("samplingFamilySelect", sUnique(samplingState.rows.map((row) => row.family)), "All antimicrobial classes");
  sFillSelect("samplingCategorySelect", ["sampling_sensitive", "diminishing_returns", "heterogeneity_dominated"], "All regimes");
  Array.from(sById("samplingCategorySelect")?.options ?? []).forEach((option) => {
    if (SAMPLING_LABELS[option.value]) option.textContent = SAMPLING_LABELS[option.value];
  });
}

function setupSamplingListeners() {
  if (samplingState.listenersReady) return;
  samplingState.listenersReady = true;
  [
    "samplingYearSelect",
    "samplingCountrySelect",
    "samplingSpeciesSelect",
    "samplingFamilySelect",
    "samplingCategorySelect",
    "samplingCountrySortSelect"
  ].forEach((id) => sById(id)?.addEventListener("change", redrawSamplingSection));

  const resizePlot = () => {
    if (!samplingState.loaded) return;
    const plot = sById("samplingCountryBarPlot");
    if (plot) requestAnimationFrame(() => Plotly.Plots.resize(plot));
  };
  window.addEventListener("resize", resizePlot);
  window.addEventListener("amrTaskChange", (event) => {
    if (event.detail?.task === "noise") {
      redrawSamplingSection();
      setTimeout(resizePlot, 0);
    }
  });
}

async function loadGnnSamplingData() {
  try {
    let rawRows = await sFetchCsvOptional(GNN_SAMPLING_PATHS.cells);
    let allowDerivation = false;
    if (!rawRows.length) rawRows = await sFetchCsvOptional(GNN_SAMPLING_PATHS.legacyCells);
    if (!rawRows.length) {
      rawRows = await sFetchCsvOptional(GNN_SAMPLING_PATHS.fallbackObserved);
      allowDerivation = true;
    }
    if (!rawRows.length) throw new Error("No GNN sampling rows were found.");

    const cleaned = cleanSamplingRows(rawRows, allowDerivation);
    let excludedRaw = await sFetchCsvOptional(GNN_SAMPLING_PATHS.excluded);
    if (!excludedRaw.length) excludedRaw = await sFetchCsvOptional(GNN_SAMPLING_PATHS.legacyExcluded);

    samplingState.rows = cleaned.rows;
    samplingState.excludedRows = excludedRaw.length ? cleanExcludedRows(excludedRaw) : cleaned.excluded;
    samplingState.loaded = true;

    initialiseSamplingControls();
    setupSamplingListeners();
    redrawSamplingSection();
    sUpdateStatus(`${samplingState.rows.length.toLocaleString()} sampling cells loaded`);
  } catch (error) {
    console.error(error);
    sUpdateStatus("Sampling data not loaded", true);
    const element = sById("samplingCountryBarPlot");
    if (element) element.innerHTML = `<div class="samplingEmpty">Could not load GNN sampling data.<br>${sEscape(error.message || error)}</div>`;
  }
}

document.addEventListener("DOMContentLoaded", loadGnnSamplingData);
