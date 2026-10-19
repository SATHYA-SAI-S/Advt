/**
 * Customer Journey Explorer & Comet Dispatcher Module
 * Inspects real sequential customer paths, timestamps, conversion outcomes,
 * and allows launching individual journey comets into the constellation.
 */
export class JourneyExplorer {
  constructor(streamContainerId, onJourneyClick) {
    this.container = document.getElementById(streamContainerId);
    this.onJourneyClick = onJourneyClick || (() => {});
    this.data = null;
    this.activeModel = 'Causal AIPW';

    // Filters
    this.filterChannel = 'all';
    this.filterStatus = 'all';
    this.filterLength = 'all';
    this.selectedArc = null;
    this.highlightedJourneyId = null;

    this.initFilterListeners();
  }

  setData(data) {
    this.data = data;
    this.populateChannelDropdown();
    this.render();
  }

  setActiveModel(modelName) {
    this.activeModel = modelName;
    this.render();
  }

  initFilterListeners() {
    const chSelect = document.getElementById('journey-channel-filter');
    const stSelect = document.getElementById('journey-status-filter');
    const lnSelect = document.getElementById('journey-length-filter');

    if (chSelect) {
      chSelect.addEventListener('change', (e) => {
        this.filterChannel = e.target.value;
        this.selectedArc = null;
        this.render();
      });
    }

    if (stSelect) {
      stSelect.addEventListener('change', (e) => {
        this.filterStatus = e.target.value;
        this.selectedArc = null;
        this.render();
      });
    }

    if (lnSelect) {
      lnSelect.addEventListener('change', (e) => {
        this.filterLength = e.target.value;
        this.selectedArc = null;
        this.render();
      });
    }
  }

  populateChannelDropdown() {
    const chSelect = document.getElementById('journey-channel-filter');
    if (!chSelect || !this.data) return;

    const channels = this.data.channels || [];
    const journeyCount = (this.data.journeys || this.data.comets || []).length;
    chSelect.innerHTML = `<option value="all">All Channels (${journeyCount} journeys)</option>` +
      channels.map(ch => `<option value="${ch}">${ch}</option>`).join('');
  }

  filterByArc(arc) {
    this.selectedArc = arc;
    this.render();
  }

  render() {
    if (!this.data || !this.container) return;

    let journeys = this.data.journeys || this.data.comets || [];

    // Apply Channel Filter
    if (this.filterChannel !== 'all') {
      journeys = journeys.filter(j => (j.touchpoints || []).includes(this.filterChannel));
    }

    // Apply Status Filter
    if (this.filterStatus === 'converted') {
      journeys = journeys.filter(j => j.converted === 1);
    } else if (this.filterStatus === 'dropoff') {
      journeys = journeys.filter(j => j.converted === 0);
    }

    // Apply Length Filter
    if (this.filterLength === '1') {
      journeys = journeys.filter(j => (j.touchpoints || []).length === 1);
    } else if (this.filterLength === '2') {
      journeys = journeys.filter(j => (j.touchpoints || []).length === 2);
    } else if (this.filterLength === '3') {
      journeys = journeys.filter(j => (j.touchpoints || []).length === 3);
    } else if (this.filterLength === '4+') {
      journeys = journeys.filter(j => (j.touchpoints || []).length >= 4);
    }

    // Apply Arc Filter if set
    if (this.selectedArc) {
      journeys = journeys.filter(j => {
        const tps = j.touchpoints || [];
        for (let i = 0; i < tps.length - 1; i++) {
          if (tps[i] === this.selectedArc.source && tps[i+1] === this.selectedArc.target) {
            return true;
          }
        }
        if (this.selectedArc.target === 'Conversion' && j.converted === 1 && tps[tps.length - 1] === this.selectedArc.source) {
          return true;
        }
        if (this.selectedArc.target === 'Drop-off' && j.converted === 0 && tps[tps.length - 1] === this.selectedArc.source) {
          return true;
        }
        return false;
      });
    }

    if (journeys.length === 0) {
      this.container.innerHTML = `
        <div class="empty-state-hint">
          <p>No customer journey comets match the selected filter criteria.</p>
        </div>
      `;
      return;
    }

    const channelColorMap = new Map(this.data.nodes.map(n => [n.id, n.color || '#38bdf8']));

    this.container.innerHTML = journeys.slice(0, 35).map(j => {
      const isHighlighted = j.id === this.highlightedJourneyId;
      const modelCredits = (j.models && j.models[this.activeModel]) || [];
      const tps = j.touchpoints || [];

      const touchpointsHtml = tps.map((tp, idx) => {
        const color = channelColorMap.get(tp) || '#38bdf8';
        const credit = modelCredits[idx] !== undefined ? (modelCredits[idx] * 100).toFixed(0) : '—';
        const timeOffset = (j.timestamps && j.timestamps[idx]) ? j.timestamps[idx] : `+${idx * 12}h`;

        return `
          <div class="j-node-chip" style="background: ${color}18; color: ${color}; border-color: ${color}45;" title="${tp} (${timeOffset})">
            <span>${tp}</span>
            <span style="font-size: 0.65rem; opacity: 0.75; font-family: var(--font-mono)">${timeOffset}</span>
            ${(j.converted === 1 && credit !== '—') ? `<span style="font-size: 0.65rem; font-weight: 700; background: ${color}35; padding: 1px 4px; border-radius: 4px;">${credit}%</span>` : ''}
          </div>
          ${idx < tps.length - 1 ? `<span class="j-arrow">➔</span>` : ''}
        `;
      }).join('');

      return `
        <div class="journey-row ${isHighlighted ? 'highlighted' : ''}" data-journey-id="${j.id}">
          <div style="display: flex; align-items: center; gap: 8px;">
            <span class="comet-indicator" title="Customer Journey Comet">☄️</span>
            <span class="j-id font-mono">${j.id}</span>
          </div>
          <div class="j-path">${touchpointsHtml}</div>
          <div class="j-meta">
            <span class="j-time">${j.duration_hours || 24}h duration</span>
            <span class="${j.converted === 1 ? 'j-status-converted' : 'j-status-dropoff'}">
              ${j.converted === 1 ? '✓ Nexus' : '✕ Void'}
            </span>
          </div>
        </div>
      `;
    }).join('');

    this.container.querySelectorAll('.journey-row').forEach(row => {
      row.addEventListener('click', () => {
        const jId = row.getAttribute('data-journey-id');
        const journey = (this.data.journeys || this.data.comets || []).find(j => j.id === jId);
        if (journey) {
          this.highlightedJourneyId = (this.highlightedJourneyId === jId) ? null : jId;
          this.render();
          this.onJourneyClick(this.highlightedJourneyId ? journey.touchpoints : null);
        }
      });
    });
  }
}
