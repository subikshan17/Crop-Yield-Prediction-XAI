/**
 * CropYield AI — Frontend Application Logic
 * Dedicated to real-time machine learning prediction directly from user inputs.
 */

const API_BASE = '';

// ============================================================
// Utility Helpers
// ============================================================
function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function formatNumber(num, decimals = 0) {
  if (num == null || isNaN(num)) return '—';
  return new Intl.NumberFormat('en-IN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(num);
}

function clamp(val, min, max) {
  return Math.min(max, Math.max(min, val));
}

// Toast notification
function showToast(message, type = 'error', duration = 4000) {
  let container = $('#toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.style.cssText = 'position:fixed;top:80px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:12px;pointer-events:none;';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  const colors = {
    error:   { bg: '#c95a5a', icon: '✕' },
    success: { bg: '#4a7c59', icon: '✓' },
    warning: { bg: '#e0aa62', icon: '⚠' },
    info:    { bg: '#6ba89b', icon: 'ℹ' },
  };
  const c = colors[type] || colors.info;
  toast.style.cssText = `
    display:flex;align-items:center;gap:10px;padding:14px 20px;
    background:${c.bg};color:#fff;border-radius:10px;
    font-family:var(--font-body,sans-serif);font-size:14px;font-weight:500;
    box-shadow:0 8px 30px rgba(0,0,0,0.2);pointer-events:auto;
    opacity:0;transform:translateX(40px);transition:all 0.3s ease;max-width:400px;
  `;
  toast.innerHTML = `<span style="font-size:18px;flex-shrink:0;">${c.icon}</span><span>${message}</span>`;
  container.appendChild(toast);

  requestAnimationFrame(() => {
    toast.style.opacity = '1';
    toast.style.transform = 'translateX(0)';
  });

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(40px)';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// Globals
const SHAP_COLORS = [
  '#4a7c59', '#6ba89b', '#e0aa62', '#d4856a',
  '#529e66', '#7b9eb6', '#b58db6', '#36573f',
];

let metadataCache = null;
let globalShapCache = null;

// ============================================================
// Load Metadata
// ============================================================
async function loadMetadata() {
  try {
    const res = await fetch(`${API_BASE}/api/metadata`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    metadataCache = data;

    populateSelect('#crop', data.crops, 'Select a crop');
    populateSelect('#district', data.districts, 'Select district');
    populateSelect('#season', data.seasons, 'Select season');

    if (data.available_models && $('#model-select')) {
      populateSelect('#model-select', data.available_models, null, data.model_name);
    }

  } catch (err) {
    console.error('Failed to load metadata:', err);
    showToast('Could not load dropdown options from server. Is Flask running?', 'error');
  }
}

function populateSelect(selector, items, placeholder, defaultVal) {
  const el = $(selector);
  if (!el) return;
  el.innerHTML = '';
  if (placeholder) {
    const opt = document.createElement('option');
    opt.value = ''; opt.textContent = placeholder; opt.disabled = true; opt.selected = true;
    el.appendChild(opt);
  }
  (items || []).forEach(item => {
    const opt = document.createElement('option');
    opt.value = item; opt.textContent = item;
    if (defaultVal && item === defaultVal) opt.selected = true;
    el.appendChild(opt);
  });
}

// ============================================================
// Load Dataset Stats
// ============================================================
async function loadDatasetStats() {
  try {
    const res = await fetch(`${API_BASE}/api/dataset-stats`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    setStatValue('stat-records', formatNumber(data.total_records));
    setStatValue('stat-crops', data.num_crops);
    setStatValue('stat-districts', data.num_districts);
    if (data.year_range && data.year_range.min) {
      setStatValue('stat-years', `${data.year_range.min}–${data.year_range.max}`);
    }
    setStatValue('stat-models', data.models_count);
    setStatValue('stat-primary', data.primary_model || '—');

  } catch (err) {
    console.error('Failed to load dataset stats:', err);
  }
}

function setStatValue(id, value) {
  const el = document.getElementById(id);
  if (el) {
    el.textContent = value;
    el.classList.remove('loading-pulse');
  }
}

// ============================================================
// Load Metrics & SHAP
// ============================================================
async function loadMetrics() {
  try {
    const res = await fetch(`${API_BASE}/api/metrics`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    renderMetricsTable(data.model_comparison || []);
    renderShapChart(data.climate_shap || []);

  } catch (err) {
    console.error('Failed to load metrics:', err);
    const tbody = $('#metrics-tbody');
    if (tbody) tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--on-dark-soft);">Failed to load metrics</td></tr>';
  }
}

function renderMetricsTable(models) {
  const tbody = $('#metrics-tbody');
  if (!tbody || !models.length) return;

  let bestR2 = -Infinity;
  models.forEach(m => {
    const r2 = parseFloat(m.R2 || m['R² Score'] || m['R2_Score'] || 0);
    if (r2 > bestR2) bestR2 = r2;
  });

  tbody.innerHTML = '';
  models.forEach((m, idx) => {
    const name = m.Model || m.model || `Model ${idx + 1}`;
    const r2 = parseFloat(m.R2 || m['R² Score'] || m['R2_Score'] || 0);
    const mae = parseFloat(m.MAE || m['MAE (kg/ha)'] || 0);
    const rmse = parseFloat(m.RMSE || m['RMSE (kg/ha)'] || 0);
    const time = parseFloat(m.Training_Time || m['Training Time (s)'] || 0);
    const isBest = Math.abs(r2 - bestR2) < 0.0001;

    const tr = document.createElement('tr');
    if (isBest) tr.classList.add('best');

    const r2Color = r2 > 0.95 ? '#5db872' : r2 > 0.8 ? '#e8a55a' : '#c64545';

    tr.innerHTML = `
      <td class="model-name">${isBest ? '<span style="margin-right:4px;">🏆</span>' : ''}${name}</td>
      <td class="r2-cell" style="color: ${r2Color}">${r2.toFixed(4)}</td>
      <td>${formatNumber(mae, 1)}</td>
      <td>${formatNumber(rmse, 1)}</td>
      <td>${time.toFixed(2)}s</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderShapChart(shapData) {
  const container = $('#shap-chart');
  if (!container) return;

  if (!shapData.length) {
    container.innerHTML = '<p style="color: var(--on-dark-soft);">No SHAP data available.</p>';
    return;
  }

  shapData.sort((a, b) => {
    const aVal = parseFloat(a.Mean_SHAP || a['Mean |SHAP|'] || a.mean_abs_shap || 0);
    const bVal = parseFloat(b.Mean_SHAP || b['Mean |SHAP|'] || b.mean_abs_shap || 0);
    return bVal - aVal;
  });

  const maxVal = parseFloat(
    shapData[0].Mean_SHAP || shapData[0]['Mean |SHAP|'] || shapData[0].mean_abs_shap || 1
  );

  container.innerHTML = '';
  shapData.forEach((item, i) => {
    const name = (item.Feature || item.feature || item.Climate_Factor || 'Unknown')
      .replace(/_/g, ' ');
    const val = parseFloat(item.Mean_SHAP || item['Mean |SHAP|'] || item.mean_abs_shap || 0);
    const pct = (val / maxVal) * 100;
    const color = SHAP_COLORS[i % SHAP_COLORS.length];

    const row = document.createElement('div');
    row.classList.add('shap-bar');
    row.innerHTML = `
      <div class="shap-bar__label">${name}</div>
      <div class="shap-bar__track">
        <div class="shap-bar__fill" style="width: 0%; background: ${color};"></div>
      </div>
      <div class="shap-bar__value">${val.toFixed(1)}</div>
    `;
    container.appendChild(row);

    setTimeout(() => {
      row.querySelector('.shap-bar__fill').style.width = `${pct}%`;
    }, 100 + i * 80);
  });

  globalShapCache = shapData;
  renderShapInsights(shapData);
}

function renderShapInsights(shapData) {
  const container = $('#shap-insights');
  if (!container || !shapData || !shapData.length) return;

  const topFeature = (shapData[0].Feature || shapData[0].feature || shapData[0].Climate_Factor || '')
    .replace(/_/g, ' ');

  const insights = [
    {
      icon: '🎯', title: `${topFeature} is the strongest factor`,
      text: `Across the dataset, ${topFeature.toLowerCase()} exerts the largest overall influence on crop yield outcome predictions.`,
    },
    {
      icon: '🌡️', title: 'Thermal metrics shape yield capacity',
      text: 'Average, minimum, and maximum temperatures collectively determine baseline thermal growth conditions.',
    },
    {
      icon: '💧', title: 'Rainfall & soil moisture interaction',
      text: 'Non-linear coupling between soil moisture and precipitation governs stress levels during critical growth phases.',
    },
    {
      icon: '⚡', title: 'Live Input-Specific Impact',
      text: 'When you submit custom values above, your prediction card and this insights panel dynamically update with your specific scenario analysis.',
    },
  ];

  container.innerHTML = insights.map((insight, i) => `
    <div class="card card--dark fade-in" style="padding:16px;border:1px solid rgba(255,255,255,0.06);">
      <div style="display:flex;gap:12px;align-items:flex-start;">
        <div style="width:36px;height:36px;border-radius:8px;background:${SHAP_COLORS[i % SHAP_COLORS.length]}22;color:${SHAP_COLORS[i % SHAP_COLORS.length]};display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;">${insight.icon}</div>
        <div>
          <div class="title-sm" style="color:var(--on-dark);margin-bottom:4px;">${insight.title}</div>
          <div class="body-sm" style="color:var(--on-dark-soft);">${insight.text}</div>
        </div>
      </div>
    </div>
  `).join('');
}

function updateDynamicShapInsights(data, payload) {
  const container = $('#shap-insights');
  if (!container || !data.feature_contributions) return;

  const contribs = data.feature_contributions;
  const posContribs = contribs.filter(c => c.direction === 'positive');
  const negContribs = contribs.filter(c => c.direction === 'negative');

  const topPos = posContribs.length > 0 ? posContribs[0] : null;
  const topNeg = negContribs.length > 0 ? negContribs[0] : null;

  const crop = payload.crop;
  const district = payload.district;
  const yieldVal = formatNumber(data.predicted_yield_kg_per_ha);

  const insights = [];

  // 1. Top Positive Driver
  if (topPos) {
    const featName = topPos.feature.replace(/_/g, ' ');
    insights.push({
      icon: '🌿',
      title: `Top Driver: ${featName} (+${formatNumber(topPos.impact_kg_ha)} kg/ha)`,
      text: `For ${crop} in ${district}, your input value (${topPos.user_value}) vs avg (${topPos.mean_value}) provides the strongest positive yield boost.`,
    });
  }

  // 2. Main Stress Factor
  if (topNeg) {
    const featName = topNeg.feature.replace(/_/g, ' ');
    insights.push({
      icon: '⚠️',
      title: `Main Stress Factor: ${featName} (${formatNumber(topNeg.impact_kg_ha)} kg/ha)`,
      text: `Your ${featName.toLowerCase()} value of ${topNeg.user_value} (vs avg ${topNeg.mean_value}) acts as the primary limiting factor reducing yield.`,
    });
  } else {
    insights.push({
      icon: '✨',
      title: `Optimal Climate Conditions`,
      text: `All input climate parameters are within highly favorable ranges for ${crop} in ${district}.`,
    });
  }

  // 3. Historical Benchmark Comparison
  if (data.crop_historical_stats) {
    const stats = data.crop_historical_stats;
    const diff = data.predicted_yield_kg_per_ha - stats.mean;
    const pctDiff = ((diff / stats.mean) * 100).toFixed(1);
    const isAbove = diff >= 0;

    insights.push({
      icon: '📊',
      title: `Yield vs Benchmark: ${isAbove ? '+' : ''}${pctDiff}%`,
      text: `Predicted yield (${yieldVal} kg/ha) is ${Math.abs(pctDiff)}% ${isAbove ? 'above' : 'below'} the historical mean of ${formatNumber(stats.mean)} kg/ha for ${crop}.`,
    });
  }

  // 4. Model & SHAP Summary
  insights.push({
    icon: '⚡',
    title: `Live Scenario Analysis (${data.model_used})`,
    text: `SHAP analysis dynamically calculated feature contributions across all 8 micro-climate variables for your ${crop} scenario in ${district}.`,
  });

  container.innerHTML = insights.map((insight, i) => `
    <div class="card card--dark fade-in" style="padding:16px;border:1px solid rgba(255,255,255,0.08);background:var(--surface-dark-elevated);">
      <div style="display:flex;gap:12px;align-items:flex-start;">
        <div style="width:36px;height:36px;border-radius:8px;background:${SHAP_COLORS[i % SHAP_COLORS.length]}22;color:${SHAP_COLORS[i % SHAP_COLORS.length]};display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;">${insight.icon}</div>
        <div>
          <div class="title-sm" style="color:var(--on-dark);margin-bottom:4px;">${insight.title}</div>
          <div class="body-sm" style="color:var(--on-dark-soft);">${insight.text}</div>
        </div>
      </div>
    </div>
  `).join('');
}

// ============================================================
// Form Validation
// ============================================================
function validateForm() {
  const errors = [];

  const crop = $('#crop');
  if (!crop || !crop.value) errors.push('Please select a Crop');

  const district = $('#district');
  if (!district || !district.value) errors.push('Please select a District');

  const season = $('#season');
  if (!season || !season.value) errors.push('Please select a Season');

  const tMin = parseFloat($('#temperature_min')?.value);
  const tMax = parseFloat($('#temperature_max')?.value);
  const tAvg = parseFloat($('#temperature_avg')?.value);

  if (isNaN(tMin) || isNaN(tMax) || isNaN(tAvg)) {
    errors.push('Please enter valid temperature values');
  } else {
    if (tMin > tMax) errors.push('Min temperature cannot be higher than max temperature');
  }

  const rainfall = parseFloat($('#rainfall')?.value);
  if (isNaN(rainfall) || rainfall < 0) errors.push('Rainfall must be 0 or greater');

  const humidity = parseFloat($('#humidity')?.value);
  if (isNaN(humidity) || humidity < 0 || humidity > 100) errors.push('Humidity must be between 0% and 100%');

  const soil = parseFloat($('#soil_moisture')?.value);
  if (isNaN(soil) || soil < 0 || soil > 100) errors.push('Soil moisture must be between 0% and 100%');

  return errors;
}

// ============================================================
// Prediction Form Handler
// ============================================================
async function handlePredict(e) {
  e.preventDefault();

  const errors = validateForm();
  if (errors.length > 0) {
    errors.forEach(err => showToast(err, 'warning', 4000));
    return;
  }

  const btn = $('#predict-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Calculating…';

  const compareModels = $('#compare-models-check')?.checked || false;
  const modelSelect = $('#model-select');
  const selectedModel = modelSelect ? modelSelect.value : '';

  const featureA = $('#feature-a-select')?.value || 'Soil_Moisture';
  const featureB = $('#feature-b-select')?.value || 'Rainfall_Precipitation';

  const payload = {
    crop: $('#crop').value,
    district: $('#district').value,
    season: $('#season').value,
    area: parseFloat($('#area').value) || 5000,
    temperature_avg: parseFloat($('#temperature_avg').value),
    temperature_min: parseFloat($('#temperature_min').value),
    temperature_max: parseFloat($('#temperature_max').value),
    rainfall: parseFloat($('#rainfall').value),
    humidity: parseFloat($('#humidity').value),
    solar_radiation: parseFloat($('#solar_radiation').value),
    wind_speed: parseFloat($('#wind_speed').value),
    soil_moisture: parseFloat($('#soil_moisture').value),
    feature_a: featureA,
    feature_b: featureB,
  };

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (data.success) {
      displayResult(data, payload);
      showToast('Yield & feature comparison calculated directly from your inputs!', 'success', 3000);
    } else {
      showToast('Prediction failed: ' + (data.error || 'Unknown error'), 'error');
    }

  } catch (err) {
    console.error('Prediction failed:', err);
    showToast('Could not connect to the API server. Check if app.py is running.', 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '⚡ Calculate & Compare Features';
  }
}

function displayResult(data, payload) {
  const placeholder = $('#result-placeholder');
  const output = $('#result-output');
  if (placeholder) placeholder.classList.add('hidden');
  if (!output) return;
  output.classList.remove('hidden');

  // Yield display
  const yieldEl = $('#result-yield');
  if (yieldEl) {
    yieldEl.innerHTML = `
      ${formatNumber(data.predicted_yield_kg_per_ha)} <span class="unit">kg/ha</span>
    `;
  }

  const badge = $('#result-model-badge');
  if (badge) badge.textContent = data.model_used;

  // Meta items
  const meta = $('#result-meta');
  if (meta) {
    meta.innerHTML = `
      <div class="result-card__meta-item">
        <div class="label">Crop</div>
        <div class="value">${payload.crop}</div>
      </div>
      <div class="result-card__meta-item">
        <div class="label">District</div>
        <div class="value">${payload.district}</div>
      </div>
      <div class="result-card__meta-item">
        <div class="label">Season</div>
        <div class="value">${payload.season}</div>
      </div>
      <div class="result-card__meta-item">
        <div class="label">Tonnes/Ha</div>
        <div class="value">${data.predicted_yield_tonnes_per_ha}</div>
      </div>
    `;
  }

  // Historical context bar
  const ctxEl = $('#result-context');
  if (data.crop_historical_stats && ctxEl) {
    const ctx = data.crop_historical_stats;
    const range = ctx.max - ctx.min;
    const predicted = data.predicted_yield_kg_per_ha;
    const pct = range > 0 ? clamp((predicted - ctx.min) / range * 100, 0, 100) : 50;
    const meanPct = range > 0 ? clamp((ctx.mean - ctx.min) / range * 100, 0, 100) : 50;

    ctxEl.style.display = 'block';

    const minEl = $('#context-min');
    const maxEl = $('#context-max');
    if (minEl) minEl.textContent = formatNumber(ctx.min) + ' kg/ha';
    if (maxEl) maxEl.textContent = formatNumber(ctx.max) + ' kg/ha';

    setTimeout(() => {
      const fill = $('#context-fill');
      const marker = $('#context-marker');
      if (fill) fill.style.width = `${pct}%`;
      if (marker) {
        marker.style.left = `${meanPct}%`;
        marker.title = `Historical Mean: ${formatNumber(ctx.mean)} kg/ha`;
      }
    }, 100);
  }

  // Feature breakdown for user input
  const breakdownContainer = $('#input-breakdown-container');
  const breakdownList = $('#input-breakdown-list');
  if (breakdownContainer && breakdownList && data.feature_contributions) {
    breakdownContainer.style.display = 'block';
    breakdownList.innerHTML = data.feature_contributions.map(item => {
      const isPos = item.direction === 'positive';
      const color = isPos ? '#5db872' : '#c64545';
      const sign = isPos ? '+' : '';
      const featName = item.feature.replace(/_/g, ' ');

      return `
        <div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:13px;">
          <span style="color:var(--on-dark-soft);">
            ${featName}: <strong style="color:var(--on-dark);">${item.user_value}</strong>
            <span style="font-size:11px;color:var(--muted-soft);">(avg: ${item.mean_value})</span>
          </span>
          <span style="font-family:var(--font-code);color:${color};font-weight:500;">
            ${sign}${formatNumber(item.impact_kg_ha)} kg/ha
          </span>
        </div>
      `;
    }).join('');
  }

  // Feature Comparison rendering
  if (data.feature_comparison) {
    renderFeatureComparisonDashboard(data.feature_comparison);
  }

  // Dynamically update the SHAP Analytical Insights section for this specific prediction
  updateDynamicShapInsights(data, payload);

  output.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function renderFeatureComparisonDashboard(fc) {
  const container = $('#feature-comparison-content');
  const placeholder = $('#feature-comparison-placeholder');
  if (!container || !fc) return;

  if (placeholder) placeholder.style.display = 'none';
  container.classList.remove('hidden');

  const colorA = fc.feature_a_impact >= 0 ? '#5db872' : '#c95a5a';
  const colorB = fc.feature_b_impact >= 0 ? '#5db872' : '#c95a5a';
  const colorC = fc.coupled_impact >= 0 ? '#6ba89b' : '#d4856a';
  const signA = fc.feature_a_impact >= 0 ? '+' : '';
  const signB = fc.feature_b_impact >= 0 ? '+' : '';
  const signC = fc.coupled_impact >= 0 ? '+' : '';

  container.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:20px;">
      <!-- Feature A Card -->
      <div class="card card--dark" style="padding:20px;border:1px solid rgba(255,255,255,0.08);background:var(--surface-dark-elevated);">
        <div class="caption-uppercase" style="color:var(--accent-teal);margin-bottom:6px;">Factor A: ${fc.feature_a_label}</div>
        <div class="title-md" style="color:var(--on-dark);margin-bottom:12px;">Input: ${fc.feature_a_val} <span style="font-size:13px;color:var(--on-dark-soft);">(avg: ${fc.feature_a_mean})</span></div>
        <div style="font-family:var(--font-code);font-size:16px;color:${colorA};font-weight:600;">
          Net Impact: ${signA}${formatNumber(fc.feature_a_impact)} kg/ha
        </div>
      </div>

      <!-- Feature B Card -->
      <div class="card card--dark" style="padding:20px;border:1px solid rgba(255,255,255,0.08);background:var(--surface-dark-elevated);">
        <div class="caption-uppercase" style="color:var(--accent-amber);margin-bottom:6px;">Factor B: ${fc.feature_b_label}</div>
        <div class="title-md" style="color:var(--on-dark);margin-bottom:12px;">Input: ${fc.feature_b_val} <span style="font-size:13px;color:var(--on-dark-soft);">(avg: ${fc.feature_b_mean})</span></div>
        <div style="font-family:var(--font-code);font-size:16px;color:${colorB};font-weight:600;">
          Net Impact: ${signB}${formatNumber(fc.feature_b_impact)} kg/ha
        </div>
      </div>

      <!-- Combined Coupled Interaction Card -->
      <div class="card card--dark" style="padding:20px;border:1px solid rgba(255,255,255,0.08);background:var(--surface-dark-soft);">
        <div class="caption-uppercase" style="color:var(--on-dark-soft);margin-bottom:6px;">Coupled Pair Interaction</div>
        <div class="title-sm" style="color:var(--on-dark);margin-bottom:8px;">${fc.feature_a_label} × ${fc.feature_b_label}</div>
        <div style="font-family:var(--font-code);font-size:18px;color:${colorC};font-weight:700;margin-bottom:6px;">
          Synergy Impact: ${signC}${formatNumber(fc.coupled_impact)} kg/ha
        </div>
        <div style="font-size:12px;color:var(--on-dark-soft);">
          Dominant Stress Leverage: <strong style="color:var(--on-dark);">${fc.dominant_feature}</strong>
        </div>
        <span class="badge badge--teal" style="margin-top:12px;display:inline-block;">${fc.stress_type}</span>
      </div>
    </div>
  `;
}

// ============================================================
// Reset Form
// ============================================================
function handleReset() {
  const form = $('#prediction-form');
  if (form) form.reset();

  setInputValue('#area', '5000');
  setInputValue('#temperature_avg', '28.0');
  setInputValue('#temperature_min', '22.0');
  setInputValue('#temperature_max', '34.0');
  setInputValue('#rainfall', '400');
  setInputValue('#humidity', '72');
  setInputValue('#solar_radiation', '19.5');
  setInputValue('#wind_speed', '12');
  setInputValue('#soil_moisture', '26');

  const crop = $('#crop');
  if (crop) crop.selectedIndex = 0;
  const district = $('#district');
  if (district) district.selectedIndex = 0;
  const season = $('#season');
  if (season) season.selectedIndex = 0;

  const output = $('#result-output');
  const placeholder = $('#result-placeholder');
  if (output) output.classList.add('hidden');
  if (placeholder) placeholder.classList.remove('hidden');

  const fill = $('#context-fill');
  if (fill) fill.style.width = '0%';
  const marker = $('#context-marker');
  if (marker) marker.style.left = '0%';

  const multi = $('#multi-model-results');
  if (multi) multi.classList.add('hidden');

  const breakdownContainer = $('#input-breakdown-container');
  if (breakdownContainer) breakdownContainer.style.display = 'none';

  // Restore global baseline SHAP insights on reset
  if (globalShapCache) {
    renderShapInsights(globalShapCache);
  }
  if (breakdownContainer) breakdownContainer.style.display = 'none';

  showToast('Form cleared', 'info', 2000);
}

// ============================================================
// Mobile Menu
// ============================================================
function setupMobileMenu() {
  const btn = $('#mobile-menu-btn');
  const menu = $('#mobile-menu');
  if (!btn || !menu) return;

  btn.addEventListener('click', () => {
    const isOpen = !menu.classList.contains('hidden');
    if (isOpen) {
      menu.classList.add('hidden');
      btn.setAttribute('aria-expanded', 'false');
    } else {
      menu.classList.remove('hidden');
      btn.setAttribute('aria-expanded', 'true');
    }
  });

  menu.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      menu.classList.add('hidden');
      btn.setAttribute('aria-expanded', 'false');
    });
  });
}

// ============================================================
// Smooth Scroll & Scroll Spy
// ============================================================
function setupSmoothScroll() {
  $$('.top-nav__links a, .hero__actions a, .footer__col a, #mobile-menu a').forEach(link => {
    link.addEventListener('click', e => {
      const href = link.getAttribute('href');
      if (href && href.startsWith('#')) {
        e.preventDefault();
        const target = $(href);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
    });
  });
}

function setupScrollSpy() {
  const sections = ['hero', 'predict', 'metrics', 'explainability'];
  const links = $$('.top-nav__links a');

  let ticking = false;
  window.addEventListener('scroll', () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      const scrollY = window.scrollY + 120;
      let current = '';

      sections.forEach(id => {
        const el = $(`#${id}`);
        if (el && el.offsetTop <= scrollY) {
          current = id;
        }
      });

      links.forEach(link => {
        link.classList.remove('active');
        const href = link.getAttribute('href');
        if (href === '#' + current || (current === 'hero' && href === '#predict')) {
          link.classList.add('active');
        }
      });
      ticking = false;
    });
  });
}

// ============================================================
// Init
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  loadMetadata();
  loadDatasetStats();
  loadMetrics();

  setupSmoothScroll();
  setupScrollSpy();
  setupMobileMenu();

  const form = $('#prediction-form');
  if (form) form.addEventListener('submit', handlePredict);

  const resetBtn = $('#reset-btn');
  if (resetBtn) resetBtn.addEventListener('click', handleReset);
});
