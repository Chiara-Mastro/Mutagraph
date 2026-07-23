const TEMPORAL_JUMP_PATHS = [
  "temporal_prediction/temporal_jump_candidate_rankings.csv",
  "temporal_jump_candidate_rankings.csv"
];

const PROSPECTIVE_YEAR = 2025;
const PRIORITY_TOP_COUNT = 10;

const JUMP_MODEL_NAMES = {
  residual: [
    "frozen_temporal_residual_jump_heads",
    "frozen_temporal_residual_jump_heads_future",
    "frozen_residual_jump_heads",
    "frozen_residual_jump_heads_future"
  ],
  gnn: ["gnn", "gnn_future"]
};

const jumpState = {
  rows: [],
  sourcePath: "",
  listenersReady: false
};

function jById(id) {
  return document.getElementById(id);
}

function jText(value) {
  return String(value ?? "").trim();
}

function jNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function jFirstNumber(raw, names) {
  for (const name of names) {
    const value = jNumber(raw[name]);
    if (value !== null) return value;
  }
  return null;
}

function jBoolean(value) {
  if (typeof value === "boolean") return value;
  const text = jText(value).toLowerCase();
  return ["true", "1", "yes", "y"].includes(text);
}

function jEscape(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function jUnique(values, numeric = false) {
  const clean = Array.from(new Set(values.filter((value) => value !== null && value !== undefined && value !== "")));
  return clean.sort(numeric ? (a, b) => Number(a) - Number(b) : (a, b) => String(a).localeCompare(String(b)));
}

function jFormatSusceptibility(value) {
  return Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : "—";
}

function jFormatCount(value) {
  return Number.isFinite(value) ? Math.round(value).toLocaleString() : "—";
}

function jUpdateStatus(message, isError = false) {
  const badge = jById("jumpStatusBadge");
  if (!badge) return;
  badge.classList.toggle("isError", isError);
  badge.innerHTML = `<span class="statusDot"></span>${jEscape(message)}`;
}

function jFillSelect(id, values, includeAll = false, preferred = null) {
  const select = jById(id);
  if (!select) return;
  const current = select.value;
  const options = includeAll ? ["All", ...values] : values;
  select.innerHTML = options.map((value) => `<option value="${jEscape(value)}">${jEscape(value)}</option>`).join("");
  if (options.map(String).includes(String(current))) select.value = current;
  else if (preferred !== null && options.map(String).includes(String(preferred))) select.value = preferred;
  else if (options.length) select.value = options[0];
}

function normaliseJumpDirection(value) {
  const direction = jText(value).toLowerCase();
  if (["down", "resistance", "emergence", "resistance_increase"].includes(direction)) return "down";
  if (["up", "susceptibility", "recovery", "susceptibility_increase"].includes(direction)) return "up";
  return "";
}

function jumpModelRole(modelName) {
  const name = jText(modelName).toLowerCase();
  if (JUMP_MODEL_NAMES.residual.includes(name)) return "residual";
  if (JUMP_MODEL_NAMES.gnn.includes(name)) return "gnn";
  if (name.includes("residual") && name.includes("jump") && name.includes("head")) return "residual";
  if (name === "gnn" || name.startsWith("gnn_")) return "gnn";
  return "";
}

function candidateIdentity(row) {
  return [row.country, row.inputYear, row.targetYear, row.species, row.family, row.direction].join("||");
}

function cleanJumpRows(rawRows) {
  const rows = new Map();
  let rejected = 0;

  rawRows.forEach((raw) => {
    const country = jText(raw.Country ?? raw.country);
    const species = jText(raw.Species ?? raw.species);
    const family = jText(raw.Family ?? raw.family);
    const inputYear = jFirstNumber(raw, ["input_year", "year_from"]);
    const targetYear = jFirstNumber(raw, ["target_year", "year_to", "Year", "year"]);
    const modelName = jText(raw.model_name ?? raw.model).toLowerCase();
    const direction = normaliseJumpDirection(raw.direction);
    const score = jFirstNumber(raw, ["score", "jump_probability", "direction_score"]);
    const role = jumpModelRole(modelName);

    if (!country || !species || !family || targetYear === null || !modelName || !direction || score === null || !role) {
      rejected += 1;
      return;
    }

    const resolvedInputYear = inputYear === null ? Number(targetYear) - 1 : Number(inputYear);
    const key = [modelName, direction, targetYear, country, species, family].join("||");
    if (rows.has(key)) throw new Error(`Duplicate temporal priority row: ${key}`);

    rows.set(key, {
      country,
      species,
      family,
      inputYear: resolvedInputYear,
      targetYear: Number(targetYear),
      modelName,
      role,
      direction,
      score,
      rankCountry: jFirstNumber(raw, [
        "rank_country",
        "rank_within_country_year_direction",
        "notebook_alert_rank"
      ]),
      predictedAlert: jBoolean(raw.predicted_alert ?? raw.model_alert),
      prospective: jBoolean(raw.prospective) || Number(targetYear) === PROSPECTIVE_YEAR,
      currentSusceptibility: jFirstNumber(raw, [
        "prop_S_prev",
        "p_current",
        "prop_S_current",
        "source_p_current"
      ]),
      currentIsolates: jFirstNumber(raw, [
        "n_current",
        "current_n_total",
        "n_total_current",
        "source_n_total",
        "input_n_total",
        "n_total"
      ])
    });
  });

  if (rejected) console.warn(`${rejected} temporal rows were ignored because required ranking fields were unavailable.`);
  return Array.from(rows.values());
}

function alertDirection() {
  return jById("alertDirectionSelect")?.value ?? "down";
}

function alertDirectionTitle(direction = alertDirection()) {
  return direction === "up"
    ? "Susceptibility recovery priority list"
    : "Resistance emergence priority list";
}

function baseYearDirectionRows() {
  const year = Number(jById("alertYearSelect")?.value);
  const direction = alertDirection();
  return jumpState.rows.filter((row) => row.targetYear === year && row.direction === direction);
}

function currentAlertCountry() {
  return jById("alertCountrySelect")?.value ?? "";
}

function alertFilters() {
  return {
    year: Number(jById("alertYearSelect")?.value),
    direction: alertDirection(),
    country: currentAlertCountry(),
    species: jById("alertSpeciesSelect")?.value ?? "All",
    family: jById("alertFamilySelect")?.value ?? "All"
  };
}

function rowMatchesAlertFilters(row, filters) {
  return row.targetYear === filters.year
    && row.direction === filters.direction
    && row.country === filters.country
    && (filters.species === "All" || row.species === filters.species)
    && (filters.family === "All" || row.family === filters.family);
}

function selectedModelName(role, year) {
  const available = jUnique(
    jumpState.rows
      .filter((row) => row.role === role && row.targetYear === year)
      .map((row) => row.modelName)
  );
  if (!available.length) return null;

  const preferred = year === PROSPECTIVE_YEAR
    ? available.find((name) => name.includes("future"))
    : available.find((name) => !name.includes("future"));

  return preferred ?? available[0];
}

function alertRowsForRole(role) {
  const filters = alertFilters();
  const modelName = selectedModelName(role, filters.year);
  if (!modelName || !filters.country) return [];

  return jumpState.rows
    .filter((row) => row.modelName === modelName && rowMatchesAlertFilters(row, filters))
    .sort((a, b) => {
      const aRank = Number.isFinite(a.rankCountry) && a.rankCountry > 0 ? a.rankCountry : Infinity;
      const bRank = Number.isFinite(b.rankCountry) && b.rankCountry > 0 ? b.rankCountry : Infinity;
      return aRank - bRank || b.score - a.score || a.species.localeCompare(b.species) || a.family.localeCompare(b.family);
    });
}

function updateAlertCountries() {
  const countries = jUnique(baseYearDirectionRows().map((row) => row.country));
  jFillSelect("alertCountrySelect", countries, false, countries[0] ?? null);
}

function countryRows() {
  const country = currentAlertCountry();
  return baseYearDirectionRows().filter((row) => row.country === country);
}

function updateAlertFamilies(preferred = "All") {
  const species = jById("alertSpeciesSelect")?.value ?? "All";
  const families = jUnique(
    countryRows()
      .filter((row) => species === "All" || row.species === species)
      .map((row) => row.family)
  );
  jFillSelect("alertFamilySelect", families, true, preferred);
}

function updateAlertSpeciesAndFamilies() {
  const species = jUnique(countryRows().map((row) => row.species));
  jFillSelect("alertSpeciesSelect", species, true, "All");
  updateAlertFamilies("All");
}

function updateAlertMetrics(prefix, rows) {
  const top = rows.slice(0, PRIORITY_TOP_COUNT);
  jById(`${prefix}AlertCandidateCount`).textContent = rows.length.toLocaleString();
  jById(`${prefix}AlertFlaggedCount`).textContent = rows.filter((row) => row.predictedAlert).length.toLocaleString();
  jById(`${prefix}AlertTopCount`).textContent = top.length.toLocaleString();
}

function rowRank(row, index) {
  return Number.isFinite(row.rankCountry) && row.rankCountry > 0
    ? Math.round(row.rankCountry)
    : index + 1;
}

function drawPriorityTable(prefix, role, rows, sharedKeys) {
  const body = jById(`${prefix}AlertTableBody`);
  const top = rows.slice(0, PRIORITY_TOP_COUNT);
  const label = role === "residual" ? "Temporal residual jump heads" : "GNN direction detector";

  if (!top.length) {
    body.innerHTML = `<tr><td colspan="7">No ${jEscape(label)} candidates match the selected filters.</td></tr>`;
    return;
  }

  body.innerHTML = top.map((row, index) => {
    const shared = sharedKeys.has(candidateIdentity(row));
    return `<tr class="${row.predictedAlert ? "isFlagged" : ""} ${shared ? "isSharedPriority" : ""}">
      <td><span class="priorityRank">${rowRank(row, index)}</span>${shared ? '<span class="sharedPriorityTag">Shared</span>' : ""}</td>
      <td>${jEscape(row.country)}</td>
      <td>${jEscape(row.species)}</td>
      <td>${jEscape(row.family)}</td>
      <td>${jFormatSusceptibility(row.currentSusceptibility)}</td>
      <td>${jFormatCount(row.currentIsolates)}</td>
      <td><span class="statusTag ${row.predictedAlert ? "statusAlert" : "statusBelow"}">${row.predictedAlert ? "Alert" : "No alert"}</span></td>
    </tr>`;
  }).join("");
}

function updateOverlapSummary(residualRows, gnnRows) {
  const residualTop = residualRows.slice(0, PRIORITY_TOP_COUNT);
  const gnnTop = gnnRows.slice(0, PRIORITY_TOP_COUNT);
  const residualMap = new Map(residualTop.map((row) => [candidateIdentity(row), row]));
  const gnnMap = new Map(gnnTop.map((row) => [candidateIdentity(row), row]));
  const sharedKeys = new Set([...residualMap.keys()].filter((key) => gnnMap.has(key)));
  const sharedFlagged = [...sharedKeys].filter((key) => residualMap.get(key)?.predictedAlert && gnnMap.get(key)?.predictedAlert).length;

  jById("prioritySharedCount").textContent = sharedKeys.size.toLocaleString();
  jById("priorityResidualOnlyCount").textContent = (residualTop.length - sharedKeys.size).toLocaleString();
  jById("priorityGnnOnlyCount").textContent = (gnnTop.length - sharedKeys.size).toLocaleString();
  jById("prioritySharedFlaggedCount").textContent = sharedFlagged.toLocaleString();

  const filters = alertFilters();
  const directionLabel = filters.direction === "up" ? "recovery" : "emergence";
  const summary = jById("priorityFilterSummary");
  if (summary) summary.textContent = `${filters.country} · ${filters.year} · ${directionLabel}`;

  return sharedKeys;
}

function redrawAlerts() {
  if (!jumpState.rows.length) return;

  const residualRows = alertRowsForRole("residual");
  const gnnRows = alertRowsForRole("gnn");
  const sharedKeys = updateOverlapSummary(residualRows, gnnRows);

  jById("snapshotAlertPlotTitle").textContent = alertDirectionTitle();
  jById("gnnAlertPlotTitle").textContent = alertDirectionTitle();

  updateAlertMetrics("snapshot", residualRows);
  updateAlertMetrics("gnn", gnnRows);
  drawPriorityTable("snapshot", "residual", residualRows, sharedKeys);
  drawPriorityTable("gnn", "gnn", gnnRows, sharedKeys);
}

function initialiseAlertControls() {
  const years = jUnique(jumpState.rows.map((row) => row.targetYear), true).map(String);
  jFillSelect("alertYearSelect", years, false, years.includes(String(PROSPECTIVE_YEAR)) ? String(PROSPECTIVE_YEAR) : years.at(-1));
  const directionSelect = jById("alertDirectionSelect");
  if (directionSelect) directionSelect.value = "down";
  updateAlertCountries();
  updateAlertSpeciesAndFamilies();
  redrawAlerts();
}

function setupJumpListeners() {
  if (jumpState.listenersReady) return;
  jumpState.listenersReady = true;

  ["alertYearSelect", "alertDirectionSelect"].forEach((id) => {
    jById(id)?.addEventListener("change", () => {
      updateAlertCountries();
      updateAlertSpeciesAndFamilies();
      redrawAlerts();
    });
  });

  jById("alertCountrySelect")?.addEventListener("change", () => {
    updateAlertSpeciesAndFamilies();
    redrawAlerts();
  });

  jById("alertSpeciesSelect")?.addEventListener("change", () => {
    updateAlertFamilies("All");
    redrawAlerts();
  });

  jById("alertFamilySelect")?.addEventListener("change", redrawAlerts);
  window.addEventListener("amrTaskChange", (event) => {
    if (event.detail?.task === "alerts" && jumpState.rows.length) redrawAlerts();
  });
}

async function fetchFirstJumpCsv() {
  let lastError = null;
  for (const path of TEMPORAL_JUMP_PATHS) {
    try {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return { path, text: await response.text() };
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(`Temporal priority data could not be loaded. ${lastError?.message ?? "No matching CSV was found."}`);
}

async function loadJumpData() {
  try {
    const loaded = await fetchFirstJumpCsv();
    const parsed = Papa.parse(loaded.text, { header: true, skipEmptyLines: true });
    const rows = cleanJumpRows(parsed.data);
    if (!rows.length) throw new Error("No valid temporal priority rows were found.");

    jumpState.rows = rows;
    jumpState.sourcePath = loaded.path;
    initialiseAlertControls();
    jUpdateStatus(`${rows.length.toLocaleString()} priority rows loaded`);
  } catch (error) {
    console.error(error);
    jUpdateStatus("Jump priority data not loaded", true);
    ["snapshotAlertTableBody", "gnnAlertTableBody"].forEach((id) => {
      const body = jById(id);
      if (body) body.innerHTML = `<tr><td colspan="7">Could not load priority data.<br>${jEscape(error.message || error)}</td></tr>`;
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setupJumpListeners();
  loadJumpData();
});
