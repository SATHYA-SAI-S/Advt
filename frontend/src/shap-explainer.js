/**
 * SHAP Explainability Module
 * Visualizes ML-derived SHAP values, decomposing predictive feature importance
 * into early-touch assist impact versus last-touch closing impact.
 */
export class ShapExplainer {
  constructor(globalBarsContainerId, impactContainerId) {
    this.globalContainer = document.getElementById(globalBarsContainerId);
    this.impactContainer = document.getElementById(impactContainerId);
    this.data = null;
    this.selectedChannelId = null;
  }

  setData(data) {
    this.data = data;
    this.render();
  }

  setSelectedChannel(channelId) {
    this.selectedChannelId = channelId;
    this.render();
  }

  render() {
    if (!this.data || !this.globalContainer || !this.impactContainer) return;

    const channels = (this.data.nodes || []).filter(n => !n.is_endpoint);
    const channelsWithShap = channels.filter(n => n.shap && typeof n.shap.global_importance === 'number');

    if (channelsWithShap.length === 0) {
      this.globalContainer.innerHTML = `
        <div style="padding: 1.5rem 0.5rem; color: #64748b; font-size: 0.75rem; text-align: center;">
          SHAP feature attributions not calculated for this universe cohort.
        </div>
      `;
      this.impactContainer.innerHTML = `
        <div style="padding: 1.5rem 0.5rem; color: #64748b; font-size: 0.75rem; text-align: center;">
          Observational / RCT causal estimation active.
        </div>
      `;
      return;
    }

    // Sort channels by global SHAP importance descending
    const sortedGlobal = [...channelsWithShap].sort((a, b) => (b.shap.global_importance || 0) - (a.shap.global_importance || 0));
    const maxImportance = sortedGlobal[0].shap.global_importance || 1.0;

    // Render Global Importance Bars
    this.globalContainer.innerHTML = sortedGlobal.map(ch => {
      const imp = ch.shap.global_importance || 0;
      const widthPct = ((imp / maxImportance) * 100).toFixed(1);
      const isSelected = ch.id === this.selectedChannelId;

      return `
        <div class="shap-bar-row ${isSelected ? 'highlighted' : ''}" style="${isSelected ? 'background: rgba(56, 189, 248, 0.08); border-radius: 4px; padding: 2px 4px;' : ''}">
          <span class="shap-bar-name" style="color: ${ch.color}">${ch.name}</span>
          <div class="shap-bar-track">
            <div class="shap-bar-fill" style="width: ${widthPct}%; background: linear-gradient(90deg, ${ch.color}88, ${ch.color});"></div>
          </div>
          <span class="shap-bar-val font-mono">${imp.toFixed(3)}</span>
        </div>
      `;
    }).join('');

    // Render Assist Impact vs Closing Impact Breakdown
    // Positive assist impact = boosts conversions when appearing early
    // Positive closing impact = boosts conversions when appearing as final step
    const sortedImpact = [...channelsWithShap].sort((a, b) => (b.shap.closing_impact || 0) - (a.shap.closing_impact || 0));

    this.impactContainer.innerHTML = sortedImpact.map(ch => {
      const assist = ch.shap.assist_impact || 0;
      const closing = ch.shap.closing_impact || 0;
      const isSelected = ch.id === this.selectedChannelId;

      return `
        <div class="shap-bar-row ${isSelected ? 'highlighted' : ''}" style="${isSelected ? 'background: rgba(56, 189, 248, 0.08); border-radius: 4px; padding: 2px 4px;' : ''}">
          <span class="shap-bar-name" style="color: ${ch.color}">${ch.name}</span>
          <div style="flex-grow: 1; display: flex; gap: 4px; height: 10px;">
            <div title="Assist Impact: ${assist.toFixed(3)}" style="width: ${Math.max(5, Math.abs(assist) * 60)}%; background: #38bdf8; border-radius: 2px; opacity: 0.85;"></div>
            <div title="Closing Impact: ${closing.toFixed(3)}" style="width: ${Math.max(5, Math.abs(closing) * 60)}%; background: #10b981; border-radius: 2px;"></div>
          </div>
          <span class="shap-bar-val font-mono" style="font-size: 0.72rem">
            <span class="text-cyan">A:${assist.toFixed(2)}</span> / <span class="text-emerald">C:${closing.toFixed(2)}</span>
          </span>
        </div>
      `;
    }).join('');
  }
}
