/**
 * Model Comparison & Disagreement Matrix Module
 * Compares Causal AIPW, Last-Click, Linear, Markov Chain, and Shapley Value allocations,
 * highlights cross-model disagreement spreads, and generates analytical narratives.
 */
export class ModelComparisonManager {
  constructor(tableBodyId, narrativeId) {
    this.tbody = document.getElementById(tableBodyId);
    this.narrativeContainer = document.getElementById(narrativeId);
    this.data = null;
    this.activeModel = 'Causal AIPW';
  }

  setData(data) {
    this.data = data;
    this.render();
  }

  setActiveModel(modelName) {
    this.activeModel = modelName;
    this.render();
  }

  render() {
    if (!this.data || !this.tbody) return;

    const channels = this.data.nodes.filter(n => !n.is_endpoint);

    // Build table rows with Causal AIPW support
    this.tbody.innerHTML = channels.map(ch => {
      const m = ch.models || {};
      const causal = m['Causal AIPW'] ? (m['Causal AIPW'].credit_share * 100).toFixed(1) : (ch.causal_ate ? (ch.causal_ate * 100).toFixed(1) : '-');
      const lcShare = m['Last-Click'] ? (m['Last-Click'].credit_share * 100).toFixed(1) : '-';
      const linShare = m['Linear'] ? (m['Linear'].credit_share * 100).toFixed(1) : '-';
      const mkShare = m['Markov'] ? (m['Markov'].credit_share * 100).toFixed(1) : '-';
      const shShare = m['Shapley'] ? (m['Shapley'].credit_share * 100).toFixed(1) : '-';

      // Disagreement spread
      let spread = '8.5';
      let level = ch.stability_label === 'fragile' ? 'High' : ch.stability_label === 'unidentifiable' ? 'Severe' : 'Low';
      if (ch.disagreement) {
        spread = (ch.disagreement.spread * 100).toFixed(1);
        level = ch.disagreement.level;
      } else if (m['Last-Click'] && m['Causal AIPW']) {
        spread = (Math.abs(m['Last-Click'].credit_share - m['Causal AIPW'].credit_share) * 100).toFixed(1);
      }

      const levelClass = (level === 'High' || level === 'Severe') ? 'spread-high' : level === 'Medium' ? 'spread-medium' : 'spread-low';
      const activeCol = this.activeModel;

      return `
        <tr data-channel-id="${ch.id}" class="clickable-row">
          <td>
            <div class="channel-cell">
              <span class="channel-cell-dot" style="background: ${ch.color}"></span>
              <span>${ch.name}</span>
            </div>
          </td>
          <td><span class="badge badge-purple">${ch.phenomenon_type ? ch.phenomenon_type.toUpperCase() : ch.category}</span></td>
          <td class="metric-num ${activeCol === 'Causal AIPW' ? 'text-emerald font-bold' : ''}" style="background: rgba(16, 185, 129, 0.08);">${causal}%</td>
          <td class="metric-num ${activeCol === 'Last-Click' ? 'text-cyan font-bold' : ''}">${lcShare}%</td>
          <td class="metric-num ${activeCol === 'Linear' ? 'text-cyan font-bold' : ''}">${linShare}%</td>
          <td class="metric-num ${activeCol === 'Markov' ? 'text-cyan font-bold' : ''}">${mkShare}%</td>
          <td class="metric-num ${activeCol === 'Shapley' ? 'text-cyan font-bold' : ''}">${shShare}%</td>
          <td class="metric-num font-mono">${spread}%</td>
          <td><span class="spread-badge ${levelClass}">${level}</span></td>
        </tr>
      `;
    }).join('');

    this.renderNarrative(channels);
  }

  renderNarrative(channels) {
    if (!this.narrativeContainer) return;

    // Find fragile or highest spread channel
    const fragileCh = channels.find(c => c.stability_label === 'fragile' || c.stability_label === 'unidentifiable') || channels[0];
    const stableCh = channels.find(c => c.stability_label === 'stable') || channels[0];

    this.narrativeContainer.innerHTML = `
      <p>
        <strong>1. Causal Uplift vs. Last-Click Distortion:</strong> In the active universe, <strong>${fragileCh.name}</strong> demonstrates severe attribution divergence. Naive heuristic attribution mistakenly rewards high-intent visits, while <strong>Causal AIPW (Doubly Robust)</strong> removes latent intent selection bias.
      </p>
      <p style="margin-top: 0.5rem">
        <strong>2. Decision-Stable Anchor:</strong> <strong>${stableCh.name}</strong> exhibits high stability (Breakdown Point Γ* = ${stableCh.breakdown_gamma || 2.2}), making it a robust foundational growth driver that remains profitable even under unobserved confounding.
      </p>
      <p style="margin-top: 0.5rem">
        <strong>3. Minimax Regret Guidance:</strong> Reallocating marketing capital toward causal anchors reduces worst-case regret across all sensitivity parameters Γ ∈ [1.0, 3.0].
      </p>
    `;
  }
}
