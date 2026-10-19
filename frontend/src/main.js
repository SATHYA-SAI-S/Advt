import { TouchpointConstellation } from './constellation.js';
import { ModelComparisonManager } from './model-comparison.js';
import { ShapExplainer } from './shap-explainer.js';
import { JourneyExplorer } from './journey-explorer.js';

/**
 * Causal Cosmos • Main Application Orchestrator
 * Coordinates state, multi-universe data loading, sensitivity Γ updates,
 * and cosmic HUD interactions.
 */
class AttributionApp {
  constructor() {
    this.data = null;
    this.activeUniverse = 'criteo'; // 'criteo', 'ga4', 'simulated'
    this.activeModel = 'Causal AIPW'; // Default to Causal AIPW!
    this.activeCohort = 'all';
    this.currentGamma = 1.0;
    this.selectedChannel = null;
    this.selectedCompareChannel = null;

    this.universeFiles = {
      'criteo': '/cosmos_criteo.json',
      'ga4': '/cosmos_ga4.json',
      'simulated': '/cosmos_simulated.json'
    };

    this.init();
  }

  async init() {
    try {
      await this.loadUniverse(this.activeUniverse);
      this.bindUIEventListeners();
      this.setupKeyboardShortcuts();
    } catch (err) {
      console.error('Initialization error:', err);
      this.showErrorState(err);
    }
  }

  async loadUniverse(universeId) {
    const fileUrl = this.universeFiles[universeId] || '/cosmos_criteo.json';
    const response = await fetch(fileUrl);
    if (!response.ok) {
      throw new Error(`Failed to load ${universeId} universe: ${response.statusText}`);
    }
    this.data = await response.json();
    this.activeUniverse = universeId;

    this.updateHeaderAndProvenance();
    this.updateExecutiveStatRibbon();

    if (!this.constellation) {
      this.initModules();
    } else {
      this.updateModulesData();
    }
  }

  showErrorState(err) {
    const section = document.getElementById('constellation-section');
    if (section) {
      section.innerHTML = `
        <div style="padding: 3rem; text-align: center; color: #f43f5e; background: rgba(244, 63, 94, 0.08); border-radius: 12px; margin: 2rem;">
          <h2>Unable to Load Celestial Universe</h2>
          <p style="margin-top: 0.5rem; color: #cbd5e1;">${err.message}</p>
        </div>
      `;
    }
  }

  updateHeaderAndProvenance() {
    const titleEl = document.getElementById('universe-title-display');
    const badgeEl = document.getElementById('universe-status-badge');
    const narrativeEl = document.getElementById('universe-narrative-banner');

    if (titleEl) titleEl.textContent = this.data.universe_title || 'Causal Cosmos';
    if (badgeEl) {
      const isRct = this.data.observational_status === 'randomized_rct';
      const isSim = this.data.observational_status === 'synthetic_ground_truth';
      badgeEl.className = isRct ? 'badge badge-emerald' : isSim ? 'badge badge-purple' : 'badge badge-amber';
      badgeEl.textContent = isRct ? '✓ Randomized RCT Clarity' : isSim ? '🔬 Ground Truth Calibrated' : '⚠️ Purely Observational Fog';
    }
    if (narrativeEl) {
      narrativeEl.textContent = this.data.narrative || '';
    }
  }

  updateExecutiveStatRibbon() {
    const stats = this.data.stats || {};
    const totalEl = document.getElementById('stat-total-journeys');
    const convEl = document.getElementById('stat-conversions');
    const chEl = document.getElementById('stat-channels-count');
    const modelBadge = document.getElementById('stat-active-model');
    const topChEl = document.getElementById('stat-top-channel');
    const disagEl = document.getElementById('stat-disagreement');

    if (totalEl) totalEl.textContent = (stats.total_entities || 5000).toLocaleString();
    if (convEl) convEl.innerHTML = `${(stats.conversions || 495).toLocaleString()} <span class="stat-rate">(${stats.conversion_rate || '9.9%'})</span>`;
    if (chEl) chEl.textContent = `${(this.data.channels || []).length} Channels`;
    if (modelBadge) modelBadge.textContent = this.activeModel;
    if (topChEl) topChEl.textContent = stats.top_channel || 'Paid Search';
    if (disagEl) disagEl.textContent = stats.most_fragile || 'Direct & Display';
  }

  initModules() {
    // 1. Constellation Engine
    this.constellation = new TouchpointConstellation('constellation-viewport', {
      onChannelSelect: (selection) => this.handleChannelSelect(selection),
      onArcSelect: (arc) => this.handleArcSelect(arc),
      onModelChange: (model) => this.handleModelChange(model),
      onAsteroidSelect: (astInfo) => this.showAsteroidHUD(astInfo),
      onCometSelect: (cometData) => this.showCometHUD(cometData)
    });
    this.constellation.setData(this.data);
    this.constellation.setModel(this.activeModel);

    // 2. Model Comparison Manager
    this.modelComparison = new ModelComparisonManager('comparison-tbody', 'model-disagreement-narrative');
    this.modelComparison.setData(this.data);
    this.modelComparison.setActiveModel(this.activeModel);

    // 3. SHAP Explainer (if SHAP data present)
    if (document.getElementById('shap-global-bars')) {
      this.shapExplainer = new ShapExplainer('shap-global-bars', 'shap-impact-breakdown');
      this.shapExplainer.setData(this.data);
    }

    // 4. Customer Journey Explorer
    if (document.getElementById('journeys-stream')) {
      this.journeyExplorer = new JourneyExplorer('journeys-stream', (touchpoints) => {
        this.constellation.highlightJourneyPath(touchpoints);
      });
      this.journeyExplorer.setData(this.data);
      this.journeyExplorer.setActiveModel(this.activeModel);
    }
  }

  updateModulesData() {
    this.constellation.setData(this.data);
    this.constellation.setModel(this.activeModel);
    this.modelComparison.setData(this.data);
    this.modelComparison.setActiveModel(this.activeModel);
    if (this.shapExplainer) this.shapExplainer.setData(this.data);
    if (this.journeyExplorer) this.journeyExplorer.setData(this.data);
  }

  bindUIEventListeners() {
    // Universe Switcher Dropdown / Buttons
    const universeBtns = document.querySelectorAll('.universe-btn');
    universeBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const uid = btn.getAttribute('data-universe');
        universeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.loadUniverse(uid);
      });
    });

    // Model Selector Buttons (including Causal AIPW!)
    const modelBtns = document.querySelectorAll('#model-selector .segment-btn');
    modelBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const model = btn.getAttribute('data-model');
        modelBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.switchModel(model);
      });
    });

    // Γ Sensitivity Slider
    const gammaSlider = document.getElementById('gamma-slider');
    const gammaValueDisplay = document.getElementById('gamma-value-display');
    const gammaBreakdownAlert = document.getElementById('gamma-breakdown-alert');

    if (gammaSlider) {
      gammaSlider.addEventListener('input', (e) => {
        const gVal = parseFloat(e.target.value);
        this.currentGamma = gVal;
        if (gammaValueDisplay) gammaValueDisplay.textContent = gVal.toFixed(1);

        this.constellation.setGamma(gVal);

        // Check if any channel broke down at this gamma
        const fragileChannels = (this.data.nodes || [])
          .filter(n => !n.is_endpoint && gVal >= (n.breakdown_gamma || 3.0))
          .map(n => n.name);

        if (gammaBreakdownAlert) {
          if (fragileChannels.length > 0) {
            gammaBreakdownAlert.innerHTML = `⚠️ <strong>Causal Breakdown:</strong> ${fragileChannels.join(', ')} reversed!`;
            gammaBreakdownAlert.classList.remove('hidden');
          } else {
            gammaBreakdownAlert.classList.add('hidden');
          }
        }
      });
    }

    // Cohort Outcome Buttons
    const cohortBtns = document.querySelectorAll('#cohort-selector .segment-btn');
    cohortBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const filter = btn.getAttribute('data-filter');
        cohortBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.activeCohort = filter;
        this.constellation.setFilter(filter);
      });
    });

    // Cosmic Toggles
    const bindToggle = (id, fn) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('change', (e) => fn(e.target.checked));
    };

    bindToggle('toggle-labels', (v) => this.constellation.toggleLabels(v));
    bindToggle('toggle-arcs', (v) => this.constellation.toggleArcs(v));
    bindToggle('toggle-particles', (v) => this.constellation.toggleParticles(v));
    bindToggle('toggle-halos', (v) => this.constellation.toggleHalos(v));
    bindToggle('toggle-comets', (v) => this.constellation.toggleComets(v));
    bindToggle('toggle-wells', (v) => this.constellation.toggleGravityWells(v));
    bindToggle('toggle-heartbeat', (v) => this.constellation.toggleHeartbeat(v));
    bindToggle('toggle-asteroids', (v) => this.constellation.toggleAsteroids(v));
    bindToggle('toggle-nebulae', (v) => this.constellation.toggleNebulae(v));

    // Reset Zoom
    const btnResetZoom = document.getElementById('btn-reset-zoom');
    if (btnResetZoom) {
      btnResetZoom.addEventListener('click', () => this.constellation.resetZoom());
    }

    // Compare Mode
    const btnCompareMode = document.getElementById('btn-compare-mode');
    if (btnCompareMode) {
      btnCompareMode.addEventListener('click', () => {
        const isNowActive = !btnCompareMode.classList.contains('active');
        btnCompareMode.classList.toggle('active', isNowActive);
        this.constellation.setCompareMode(isNowActive);
      });
    }

    // Close HUD Modals
    const closeHudBtns = document.querySelectorAll('.hud-close-btn');
    closeHudBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.cosmic-hud-modal').forEach(m => m.classList.add('hidden'));
      });
    });
  }

  switchModel(modelName) {
    this.activeModel = modelName;
    const modelBadge = document.getElementById('stat-active-model');
    if (modelBadge) modelBadge.textContent = modelName;

    this.constellation.setModel(modelName);
    this.modelComparison.setActiveModel(modelName);
    if (this.journeyExplorer) this.journeyExplorer.setActiveModel(modelName);
  }

  handleChannelSelect(channel) {
    this.selectedChannel = channel;
    this.renderChannelDetails();
  }

  handleArcSelect(arc) {
    if (this.journeyExplorer) {
      this.journeyExplorer.filterByArc(arc);
    }
  }

  renderChannelDetails() {
    if (!this.selectedChannel) return;
    const ch = this.selectedChannel;

    const nameEl = document.getElementById('detail-channel-name');
    const roleEl = document.getElementById('detail-channel-role');
    const shareEl = document.getElementById('det-credit-share');
    const ciEl = document.getElementById('det-ci-range');
    const assistEl = document.getElementById('det-assist-ratio');
    const spreadEl = document.getElementById('det-spread-badge');

    if (nameEl) nameEl.textContent = ch.name;
    if (roleEl) roleEl.textContent = ch.role || ch.description;

    const modelData = (ch.models && ch.models[this.activeModel]) || (ch.models && ch.models['Causal AIPW']) || {};
    const share = ((modelData.credit_share || 0) * 100).toFixed(1);
    if (shareEl) shareEl.textContent = `${share}%`;

    if (ciEl) {
      if (ch.ci) ciEl.textContent = `[${(ch.ci[0]*100).toFixed(1)}% to ${(ch.ci[1]*100).toFixed(1)}%]`;
      else ciEl.textContent = '±1.4%';
    }

    if (assistEl) {
      assistEl.textContent = ch.causal_ate ? `${(ch.causal_ate * 100).toFixed(2)}% ATE` : 'N/A';
    }

    if (spreadEl) {
      spreadEl.textContent = `Γ* = ${ch.breakdown_gamma} (${(ch.phenomenon_type || ch.stability_label).toUpperCase()})`;
    }

    const emptyHint = document.getElementById('detail-empty-state');
    const singleView = document.getElementById('single-channel-view');
    if (emptyHint) emptyHint.classList.add('hidden');
    if (singleView) singleView.classList.remove('hidden');
  }

  showAsteroidHUD(astInfo) {
    const modal = document.getElementById('asteroid-hud-modal');
    if (!modal) return;

    const ast = astInfo.asteroid;
    const node = astInfo.node;

    document.getElementById('hud-ast-name').textContent = ast.name;
    document.getElementById('hud-ast-channel').textContent = `Target: ${node.name} (${node.stability_label.toUpperCase()})`;
    document.getElementById('hud-ast-sample').textContent = `${ast.sample_size.toLocaleString()} users`;
    document.getElementById('hud-ast-cost').textContent = `$${ast.cost.toLocaleString()}`;
    document.getElementById('hud-ast-regret').textContent = `${ast.regret_reduction} units`;
    document.getElementById('hud-ast-voi').textContent = `$${ast.voi}`;

    modal.classList.remove('hidden');
  }

  showCometHUD(comet) {
    const modal = document.getElementById('comet-hud-modal');
    if (!modal) return;

    document.getElementById('hud-comet-id').textContent = comet.id;
    document.getElementById('hud-comet-status').textContent = comet.converted === 1 ? '🎯 Converted Nexus' : '🌑 Dark Void Drop-off';
    document.getElementById('hud-comet-status').style.color = comet.converted === 1 ? '#10b981' : '#f43f5e';
    document.getElementById('hud-comet-duration').textContent = `${comet.duration_hours} hours`;
    document.getElementById('hud-comet-touches').textContent = `${comet.touchpoints.length} touchpoints`;
    document.getElementById('hud-comet-path').textContent = comet.touchpoints.join(' ➔ ');

    modal.classList.remove('hidden');
  }

  setupKeyboardShortcuts() {
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        document.querySelectorAll('.cosmic-hud-modal').forEach(m => m.classList.add('hidden'));
      }
    });
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.causalApp = new AttributionApp();
});
