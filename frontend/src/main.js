import {
  renderForestPlot,
  renderSensitivityChart,
  renderOverlapDiagnostics,
  renderBudgetAllocation,
  renderExperimentTable,
  renderProvenanceGrid
} from './dashboard.js';

/**
 * Main Application Orchestrator for ResearchResult Dashboard.
 * Loads results directly from /research_result.json, /ga4_research_result.json, or /criteo_research_result.json.
 * Completely free of hardcoded mock data.
 */
class ResearchDashboardApp {
  constructor() {
    this.currentDataset = 'synthetic'; // 'synthetic', 'ga4', 'criteo'
    this.currentGamma = 1.0;
    this.selectedOverlapChannel = null;
    this.data = null;

    this.datasetFiles = {
      synthetic: '/research_result.json',
      ga4: '/ga4_research_result.json',
      criteo: '/criteo_research_result.json'
    };

    this.datasetTitles = {
      synthetic: {
        title: 'Synthetic Multi-Touch Benchmark',
        subtitle: 'Simulated customer journeys with known unobserved intent confounding U'
      },
      ga4: {
        title: 'Google Analytics 4 Observational Cohort',
        subtitle: '168k real visitor sessions from Google Merchandise Store with observational traffic sources'
      },
      criteo: {
        title: 'Criteo Randomized Uplift Benchmark',
        subtitle: '13.98M impressions randomized controlled trial benchmark with empirical causal ground truth'
      }
    };

    this.init();
  }

  async init() {
    try {
      this.bindControls();
      await this.loadDataset(this.currentDataset);
    } catch (err) {
      console.error('Initialization error:', err);
      this.showError(err);
    }
  }

  bindControls() {
    // Dataset switch buttons
    const buttons = document.querySelectorAll('.dataset-btn');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        const ds = btn.getAttribute('data-dataset');
        buttons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.loadDataset(ds);
      });
    });

    // Gamma slider
    const gammaSlider = document.getElementById('gamma-slider');
    const gammaDisplay = document.getElementById('gamma-display');
    if (gammaSlider) {
      gammaSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        this.currentGamma = val;
        if (gammaDisplay) gammaDisplay.textContent = val.toFixed(1);
        if (this.data && this.data.sensitivity_analyses) {
          renderSensitivityChart('sensitivity-chart-container', this.data.sensitivity_analyses, val);
        }
      });
    }

    // Overlap channel dropdown
    const overlapSelect = document.getElementById('overlap-channel-select');
    if (overlapSelect) {
      overlapSelect.addEventListener('change', (e) => {
        this.selectedOverlapChannel = e.target.value;
        if (this.data && this.data.overlap_diagnostics) {
          renderOverlapDiagnostics('covariate-balance-container', 'diagnostics-metrics-row', this.data.overlap_diagnostics, this.selectedOverlapChannel);
        }
      });
    }
  }

  async loadDataset(datasetKey) {
    this.currentDataset = datasetKey;
    const filePath = this.datasetFiles[datasetKey] || '/research_result.json';

    const resp = await fetch(filePath);
    if (!resp.ok) {
      throw new Error(`Failed to load dataset from ${filePath}: ${resp.statusText}`);
    }

    this.data = await resp.json();
    this.renderAll();
  }

  renderAll() {
    if (!this.data) return;

    // 1. Header Titles
    const meta = this.datasetTitles[this.currentDataset] || { title: 'Causal Attribution Result', subtitle: '' };
    document.getElementById('dashboard-title').textContent = meta.title;
    document.getElementById('dashboard-subtitle').textContent = meta.subtitle;

    // 2. Stat Ribbon
    this.renderStatRibbon();

    // 3. Warnings Banner
    this.renderWarnings();

    // 4. Forest Plot
    renderForestPlot('forest-plot-container', this.data.effect_estimates || []);

    // 5. Sensitivity Analysis Chart
    renderSensitivityChart('sensitivity-chart-container', this.data.sensitivity_analyses || [], this.currentGamma);

    // 6. Overlap Diagnostics & Channel Dropdown
    this.populateOverlapDropdown();
    renderOverlapDiagnostics('covariate-balance-container', 'diagnostics-metrics-row', this.data.overlap_diagnostics || [], this.selectedOverlapChannel);

    // 7. Decision-Stable Budget Allocation
    renderBudgetAllocation('budget-chart-container', 'budget-meta-summary', this.data.budget_decisions || []);

    // 8. Experiment Candidates Table
    renderExperimentTable('experiments-tbody', this.data.experiment_candidates || []);

    // 9. Provenance Grid
    renderProvenanceGrid('provenance-grid', this.data);
  }

  renderStatRibbon() {
    const res = this.data;
    document.getElementById('stat-run-id').textContent = res.run_id || '-';
    document.getElementById('stat-dataset-id').textContent = res.dataset_id || '-';

    const count = (res.effect_estimates || []).length;
    document.getElementById('stat-treatments-count').textContent = `${count} Treatment${count === 1 ? '' : 's'}`;

    // Find min breakdown gamma across sensitivity points
    let minGamma = 3.0;
    (res.sensitivity_analyses || []).forEach(sa => {
      const breakdownPt = sa.points.find(p => p.includes_null);
      if (breakdownPt && breakdownPt.gamma < minGamma) {
        minGamma = breakdownPt.gamma;
      }
    });

    const gammaEl = document.getElementById('stat-min-gamma');
    if (gammaEl) {
      gammaEl.textContent = `Γ* = ${minGamma.toFixed(1)}`;
      gammaEl.className = `stat-value ${minGamma >= 2.0 ? 'text-emerald' : minGamma >= 1.4 ? 'text-amber' : 'text-rose'}`;
    }

    const budget = (res.budget_decisions && res.budget_decisions[0]) ? res.budget_decisions[0].total_budget : 0;
    document.getElementById('stat-total-budget').textContent = `$${budget.toLocaleString()}`;

    const worstRegret = (res.budget_decisions && res.budget_decisions[1] && res.budget_decisions[1].worst_case_regret) 
      ? res.budget_decisions[1].worst_case_regret.value.toFixed(4)
      : (res.budget_decisions && res.budget_decisions[0] && res.budget_decisions[0].worst_case_regret)
        ? res.budget_decisions[0].worst_case_regret.value.toFixed(4)
        : '0.0000';
    document.getElementById('stat-worst-case-regret').textContent = `${worstRegret} units`;
  }

  renderWarnings() {
    const banner = document.getElementById('warnings-banner');
    const content = document.getElementById('warning-content');
    const warnings = this.data.warnings || [];

    if (warnings.length > 0) {
      banner.classList.remove('hidden');
      content.innerHTML = warnings.map(w => `<div>${w}</div>`).join('');
    } else {
      banner.classList.add('hidden');
      content.innerHTML = '';
    }
  }

  populateOverlapDropdown() {
    const select = document.getElementById('overlap-channel-select');
    if (!select) return;

    const list = this.data.overlap_diagnostics || [];
    select.innerHTML = list.map(d => `<option value="${d.treatment_id}">${d.treatment_id}</option>`).join('');

    if (list.length > 0) {
      if (!this.selectedOverlapChannel || !list.some(d => d.treatment_id === this.selectedOverlapChannel)) {
        this.selectedOverlapChannel = list[0].treatment_id;
      }
      select.value = this.selectedOverlapChannel;
    }
  }

  showError(err) {
    const grid = document.querySelector('.dashboard-grid');
    if (grid) {
      grid.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 2rem; background: rgba(244, 63, 94, 0.1); border: 1px solid #f43f5e; border-radius: 8px; color: #f43f5e;">
          <h2>Failed to load research result</h2>
          <p>${err.message}</p>
        </div>
      `;
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.app = new ResearchDashboardApp();
});
