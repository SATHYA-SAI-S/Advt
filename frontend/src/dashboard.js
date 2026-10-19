import * as d3 from 'd3';

/**
 * Visualizations module for ResearchResult schema data.
 * Purely data-driven with D3.js: Forest Plot, Sensitivity Bounds, Covariate Balance SMD,
 * Decision-Stable Budget Allocation, and Experiment Table.
 */

// Color palette for treatments/channels
const PALETTE = [
  '#38bdf8', '#34d399', '#f472b6', '#fbbf24', '#818cf8',
  '#2dd4bf', '#a78bfa', '#f87171', '#4ade80', '#60a5fa'
];

export function getChannelColor(idx) {
  return PALETTE[idx % PALETTE.length];
}

/**
 * 1. Forest Plot: AIPW ATE with 95% Confidence Interval Error Bars
 */
export function renderForestPlot(containerId, effectEstimates) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';

  if (!effectEstimates || effectEstimates.length === 0) {
    container.innerHTML = '<div class="empty-state">No effect estimates available.</div>';
    return;
  }

  const margin = { top: 20, right: 40, bottom: 40, left: 220 };
  const width = container.clientWidth || 550;
  const rowHeight = 44;
  const height = margin.top + margin.bottom + effectEstimates.length * rowHeight;

  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', '100%')
    .attr('height', height);

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left}, ${margin.top})`);

  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Determine x domain spanning all CIs with zero included
  const allMin = d3.min(effectEstimates, d => d.confidence_interval ? d.confidence_interval[0] : d.estimate);
  const allMax = d3.max(effectEstimates, d => d.confidence_interval ? d.confidence_interval[1] : d.estimate);
  const xMin = Math.min(0, allMin) * 1.15;
  const xMax = Math.max(0.01, allMax) * 1.15;

  const xScale = d3.scaleLinear()
    .domain([xMin, xMax])
    .range([0, innerWidth]);

  const yScale = d3.scaleBand()
    .domain(effectEstimates.map(d => d.treatment_id))
    .range([0, innerHeight])
    .padding(0.4);

  // Zero reference line (null hypothesis)
  g.append('line')
    .attr('x1', xScale(0))
    .attr('x2', xScale(0))
    .attr('y1', 0)
    .attr('y2', innerHeight)
    .attr('stroke', 'rgba(255, 255, 255, 0.25)')
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '4,4');

  // Axes
  const xAxis = d3.axisBottom(xScale).ticks(6).tickFormat(d3.format('+.3f'));
  const yAxis = d3.axisLeft(yScale);

  g.append('g')
    .attr('class', 'axis y-axis')
    .call(yAxis)
    .selectAll('text')
    .style('font-size', '12px')
    .style('font-weight', '500')
    .style('fill', '#e2e8f0');

  g.append('g')
    .attr('class', 'axis x-axis')
    .attr('transform', `translate(0, ${innerHeight})`)
    .call(xAxis)
    .selectAll('text')
    .style('font-size', '11px')
    .style('fill', '#94a3b8');

  // X-axis label
  svg.append('text')
    .attr('x', margin.left + innerWidth / 2)
    .attr('y', height - 6)
    .attr('text-anchor', 'middle')
    .style('font-size', '11px')
    .style('fill', '#94a3b8')
    .text('Average Treatment Effect (ATE)');

  // Tooltip
  const tooltip = d3.select('#chart-tooltip');

  // Draw CI bars and point estimates
  effectEstimates.forEach((item, idx) => {
    const yCenter = yScale(item.treatment_id) + yScale.bandwidth() / 2;
    const color = getChannelColor(idx);
    const hasCI = item.confidence_interval && item.confidence_interval.length === 2;

    if (hasCI) {
      const x1 = xScale(item.confidence_interval[0]);
      const x2 = xScale(item.confidence_interval[1]);

      // Horizontal CI bar
      g.append('line')
        .attr('x1', x1)
        .attr('x2', x2)
        .attr('y1', yCenter)
        .attr('y2', yCenter)
        .attr('stroke', color)
        .attr('stroke-width', 2.5)
        .attr('opacity', 0.85);

      // Tick at CI lower
      g.append('line')
        .attr('x1', x1)
        .attr('x2', x1)
        .attr('y1', yCenter - 6)
        .attr('y2', yCenter + 6)
        .attr('stroke', color)
        .attr('stroke-width', 2);

      // Tick at CI upper
      g.append('line')
        .attr('x1', x2)
        .attr('x2', x2)
        .attr('y1', yCenter - 6)
        .attr('y2', yCenter + 6)
        .attr('stroke', color)
        .attr('stroke-width', 2);
    }

    // Point estimate diamond/circle
    const circle = g.append('circle')
      .attr('cx', xScale(item.estimate))
      .attr('cy', yCenter)
      .attr('r', 6)
      .attr('fill', color)
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 1.5)
      .style('cursor', 'pointer');

    // Value label beside point
    g.append('text')
      .attr('x', xScale(item.estimate) + 10)
      .attr('y', yCenter + 4)
      .style('font-size', '11px')
      .style('font-family', 'var(--font-mono)')
      .style('fill', '#f8fafc')
      .text(`${(item.estimate > 0 ? '+' : '')}${item.estimate.toFixed(4)}`);

    // Hover tooltip
    circle.on('mouseenter', (event) => {
      tooltip.classed('hidden', false)
        .style('left', `${event.pageX + 12}px`)
        .style('top', `${event.pageY - 10}px`)
        .html(`
          <div style="font-weight: 700; color: ${color}; margin-bottom: 4px;">${item.treatment_id}</div>
          <div><strong>AIPW ATE:</strong> ${item.estimate.toFixed(6)}</div>
          <div><strong>Standard Error:</strong> ${item.standard_error ? item.standard_error.toFixed(6) : 'N/A'}</div>
          <div><strong>95% CI:</strong> [${item.confidence_interval ? item.confidence_interval.map(v => v.toFixed(6)).join(', ') : 'N/A'}]</div>
          <div><strong>Method:</strong> ${item.method}</div>
        `);
    }).on('mouseleave', () => {
      tooltip.classed('hidden', true);
    });
  });
}

/**
 * 2. Sensitivity Analysis: Dorn & Guo Marginal Sensitivity Bounds (Gamma vs Bounds)
 */
export function renderSensitivityChart(containerId, sensitivityAnalyses, currentGamma = 1.0) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';

  if (!sensitivityAnalyses || sensitivityAnalyses.length === 0) {
    container.innerHTML = '<div class="empty-state">No sensitivity analysis available.</div>';
    return;
  }

  const margin = { top: 20, right: 180, bottom: 40, left: 60 };
  const width = container.clientWidth || 550;
  const height = 260;

  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', '100%')
    .attr('height', height);

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left}, ${margin.top})`);

  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Collect all points across all sensitivity analyses
  const allPoints = [];
  sensitivityAnalyses.forEach((sa, saIdx) => {
    sa.points.forEach(pt => {
      allPoints.push({ ...pt, treatment_id: sa.treatment_id, color: getChannelColor(saIdx) });
    });
  });

  const gammaExtent = d3.extent(allPoints, d => d.gamma);
  const boundMin = Math.min(0, d3.min(allPoints, d => d.lower_bound));
  const boundMax = d3.max(allPoints, d => d.upper_bound);

  const xScale = d3.scaleLinear()
    .domain([gammaExtent[0] || 1.0, gammaExtent[1] || 3.0])
    .range([0, innerWidth]);

  const yScale = d3.scaleLinear()
    .domain([boundMin * 1.15, boundMax * 1.15])
    .range([innerHeight, 0]);

  // Zero reference line
  g.append('line')
    .attr('x1', 0)
    .attr('x2', innerWidth)
    .attr('y1', yScale(0))
    .attr('y2', yScale(0))
    .attr('stroke', 'rgba(255, 255, 255, 0.25)')
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '4,4');

  // Axes
  const xAxis = d3.axisBottom(xScale).ticks(6).tickFormat(d => `Γ=${d.toFixed(1)}`);
  const yAxis = d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('+.3f'));

  g.append('g')
    .attr('class', 'axis x-axis')
    .attr('transform', `translate(0, ${innerHeight})`)
    .call(xAxis)
    .selectAll('text')
    .style('font-size', '11px')
    .style('fill', '#94a3b8');

  g.append('g')
    .attr('class', 'axis y-axis')
    .call(yAxis)
    .selectAll('text')
    .style('font-size', '11px')
    .style('fill', '#94a3b8');

  // Active Gamma Vertical Line indicator
  const gammaLine = g.append('line')
    .attr('x1', xScale(currentGamma))
    .attr('x2', xScale(currentGamma))
    .attr('y1', 0)
    .attr('y2', innerHeight)
    .attr('stroke', '#fbbf24')
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '3,3');

  g.append('text')
    .attr('x', xScale(currentGamma))
    .attr('y', -6)
    .attr('text-anchor', 'middle')
    .style('font-size', '11px')
    .style('font-weight', '600')
    .style('fill', '#fbbf24')
    .text(`Active Γ=${currentGamma.toFixed(1)}`);

  const tooltip = d3.select('#chart-tooltip');

  // Draw bounds band and lines per treatment
  sensitivityAnalyses.forEach((sa, saIdx) => {
    const color = getChannelColor(saIdx);
    const sortedPoints = [...sa.points].sort((a, b) => a.gamma - b.gamma);

    // Area between lower and upper bounds
    const areaGenerator = d3.area()
      .x(d => xScale(d.gamma))
      .y0(d => yScale(d.lower_bound))
      .y1(d => yScale(d.upper_bound));

    g.append('path')
      .datum(sortedPoints)
      .attr('fill', color)
      .attr('opacity', 0.15)
      .attr('d', areaGenerator);

    // Line for upper bound
    const upperLine = d3.line()
      .x(d => xScale(d.gamma))
      .y(d => yScale(d.upper_bound));

    g.append('path')
      .datum(sortedPoints)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '3,3')
      .attr('d', upperLine);

    // Line for lower bound
    const lowerLine = d3.line()
      .x(d => xScale(d.gamma))
      .y(d => yScale(d.lower_bound));

    g.append('path')
      .datum(sortedPoints)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 2)
      .attr('d', lowerLine);

    // Points at discrete evaluated Gammas
    sortedPoints.forEach(pt => {
      const isBreakdown = pt.includes_null;
      g.append('circle')
        .attr('cx', xScale(pt.gamma))
        .attr('cy', yScale(pt.lower_bound))
        .attr('r', 4)
        .attr('fill', isBreakdown ? '#f43f5e' : color)
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 1)
        .style('cursor', 'pointer')
        .on('mouseenter', (event) => {
          tooltip.classed('hidden', false)
            .style('left', `${event.pageX + 12}px`)
            .style('top', `${event.pageY - 10}px`)
            .html(`
              <div style="font-weight: 700; color: ${color}; margin-bottom: 4px;">${sa.treatment_id} (Γ = ${pt.gamma})</div>
              <div><strong>Lower ATE Bound:</strong> ${pt.lower_bound.toFixed(5)}</div>
              <div><strong>Upper ATE Bound:</strong> ${pt.upper_bound.toFixed(5)}</div>
              <div><strong>Includes Null (Zero):</strong> <span style="color: ${pt.includes_null ? '#f43f5e' : '#10b981'}; font-weight: 600;">${pt.includes_null ? 'Yes (Fragile)' : 'No (Stable)'}</span></div>
            `);
        })
        .on('mouseleave', () => tooltip.classed('hidden', true));
    });

    // Label at end of line in right margin with clean staggering
    const lastPoint = sortedPoints[sortedPoints.length - 1];
    const yPos = yScale(lastPoint.lower_bound) + (saIdx % 2 === 0 ? -2 : 10);
    g.append('text')
      .attr('x', innerWidth + 8)
      .attr('y', yPos)
      .style('font-size', '10.5px')
      .style('font-weight', '600')
      .style('fill', color)
      .text(sa.treatment_id);
  });

  // Footnote with calibration description
  const footnote = document.getElementById('sensitivity-footnote');
  if (footnote && sensitivityAnalyses[0]) {
    footnote.innerHTML = `<strong>Method:</strong> ${sensitivityAnalyses[0].method}. ${sensitivityAnalyses[0].calibration_description}`;
  }
}

/**
 * 3. Overlap & Positivity Diagnostics: Propensity Summary & Covariate Balance SMD
 */
export function renderOverlapDiagnostics(containerId, summaryId, overlapDiagnostics, selectedTreatmentId) {
  const container = document.getElementById(containerId);
  const summaryRow = document.getElementById(summaryId);
  if (!container || !summaryRow) return;

  container.innerHTML = '';
  summaryRow.innerHTML = '';

  if (!overlapDiagnostics || overlapDiagnostics.length === 0) {
    container.innerHTML = '<div class="empty-state">No overlap diagnostics available.</div>';
    return;
  }

  const selected = overlapDiagnostics.find(d => d.treatment_id === selectedTreatmentId) || overlapDiagnostics[0];

  // Render Metric Badges in summary row
  const extremeShare = selected.extreme_propensity_share ? (selected.extreme_propensity_share.value * 100).toFixed(1) + '%' : 'N/A';
  const ess = selected.effective_sample_size ? Math.round(selected.effective_sample_size.value).toLocaleString() : 'N/A';
  const maxWeight = selected.maximum_weight_share ? (selected.maximum_weight_share.value * 100).toFixed(2) + '%' : 'N/A';
  const isViolated = selected.overlap_flag;

  summaryRow.innerHTML = `
    <div class="diag-card">
      <div class="diag-label">Propensity Range</div>
      <div class="diag-val font-mono">[${selected.propensity_min !== null ? selected.propensity_min.toFixed(3) : '0'} - ${selected.propensity_max !== null ? selected.propensity_max.toFixed(3) : '1'}]</div>
    </div>
    <div class="diag-card">
      <div class="diag-label">Extreme Propensities (<0.05 or >0.95)</div>
      <div class="diag-val ${parseFloat(extremeShare) > 5 ? 'text-rose' : 'text-emerald'}">${extremeShare}</div>
    </div>
    <div class="diag-card">
      <div class="diag-label">Effective Sample Size (ESS)</div>
      <div class="diag-val text-cyan">${ess}</div>
    </div>
    <div class="diag-card">
      <div class="diag-label">Max Weight Share</div>
      <div class="diag-val">${maxWeight}</div>
    </div>
    <div class="diag-card">
      <div class="diag-label">Positivity Overlap Status</div>
      <div class="diag-val ${isViolated ? 'text-rose' : 'text-emerald'}">${isViolated ? '⚠️ Overlap Violated' : '✓ Sufficient Overlap'}</div>
    </div>
  `;

  // Render Covariate Balance SMD Bar Chart
  const covariates = selected.covariate_balance || [];
  if (covariates.length === 0) {
    container.innerHTML = '<div class="empty-state" style="padding: 1rem;">No individual covariate balance records.</div>';
    return;
  }

  const margin = { top: 20, right: 30, bottom: 40, left: 160 };
  const width = container.clientWidth || 550;
  const rowHeight = 30;
  const height = margin.top + margin.bottom + covariates.length * rowHeight;

  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', '100%')
    .attr('height', height);

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left}, ${margin.top})`);

  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const maxSMD = Math.max(0.15, d3.max(covariates, d => d.value || 0) * 1.25);

  const xScale = d3.scaleLinear()
    .domain([0, maxSMD])
    .range([0, innerWidth]);

  const yScale = d3.scaleBand()
    .domain(covariates.map(d => d.name.replace('smd_covariate_', '').replace('smd_', '')))
    .range([0, innerHeight])
    .padding(0.35);

  // 0.1 Threshold line (standard balance rule of thumb)
  g.append('line')
    .attr('x1', xScale(0.1))
    .attr('x2', xScale(0.1))
    .attr('y1', 0)
    .attr('y2', innerHeight)
    .attr('stroke', '#f43f5e')
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '4,4');

  g.append('text')
    .attr('x', xScale(0.1))
    .attr('y', -6)
    .attr('text-anchor', 'middle')
    .style('font-size', '10px')
    .style('fill', '#f43f5e')
    .text('0.1 Balance Threshold');

  // Axes
  const xAxis = d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('.2f'));
  const yAxis = d3.axisLeft(yScale);

  g.append('g')
    .attr('class', 'axis y-axis')
    .call(yAxis)
    .selectAll('text')
    .style('font-size', '11px')
    .style('fill', '#cbd5e1');

  g.append('g')
    .attr('class', 'axis x-axis')
    .attr('transform', `translate(0, ${innerHeight})`)
    .call(xAxis)
    .selectAll('text')
    .style('font-size', '11px')
    .style('fill', '#94a3b8');

  // Horizontal SMD Bars
  covariates.forEach(cov => {
    const cleanName = cov.name.replace('smd_covariate_', '').replace('smd_', '');
    const val = cov.value || 0;
    const isExceeded = val > 0.1;

    g.append('rect')
      .attr('x', 0)
      .attr('y', yScale(cleanName))
      .attr('width', xScale(val))
      .attr('height', yScale.bandwidth())
      .attr('fill', isExceeded ? '#f43f5e' : '#10b981')
      .attr('rx', 3);

    g.append('text')
      .attr('x', xScale(val) + 6)
      .attr('y', yScale(cleanName) + yScale.bandwidth() / 2 + 4)
      .style('font-size', '11px')
      .style('font-family', 'var(--font-mono)')
      .style('fill', isExceeded ? '#f43f5e' : '#cbd5e1')
      .text(val.toFixed(4));
  });
}

/**
 * 4. Decision-Stable Budget Allocation: Point Estimate vs Minimax Regret
 */
export function renderBudgetAllocation(containerId, metaId, budgetDecisions) {
  const container = document.getElementById(containerId);
  const metaContainer = document.getElementById(metaId);
  if (!container) return;
  container.innerHTML = '';
  if (metaContainer) metaContainer.innerHTML = '';

  if (!budgetDecisions || budgetDecisions.length === 0) {
    container.innerHTML = '<div class="empty-state">No budget decisions computed.</div>';
    return;
  }

  const pointDecision = budgetDecisions.find(b => b.strategy === 'point_estimate') || budgetDecisions[0];
  const robustDecision = budgetDecisions.find(b => b.strategy === 'robust_minimax_regret') || budgetDecisions[1] || pointDecision;

  // Render meta summary
  if (metaContainer) {
    metaContainer.innerHTML = `
      <div class="meta-row">
        <span><strong>Total Capital:</strong> $${pointDecision.total_budget.toLocaleString()}</span>
        <span><strong>Point Objective:</strong> ${pointDecision.expected_objective ? pointDecision.expected_objective.value.toFixed(2) : '-'} ${pointDecision.expected_objective ? pointDecision.expected_objective.unit : ''}</span>
        <span><strong>Worst-Case Regret:</strong> ${robustDecision.worst_case_regret ? robustDecision.worst_case_regret.value.toFixed(4) : '-'}</span>
        <span><strong>Breakdown Γ*:</strong> ${robustDecision.breakdown_gamma ? robustDecision.breakdown_gamma.value.toFixed(1) : '-'}</span>
      </div>
      <div class="meta-assumptions">
        <strong>Assumptions:</strong> ${robustDecision.assumptions ? robustDecision.assumptions.join(' • ') : 'None stated'}
      </div>
    `;
  }

  const channels = pointDecision.allocations.map(a => a.channel_id);
  const pointMap = Object.fromEntries(pointDecision.allocations.map(a => [a.channel_id, a.budget]));
  const robustMap = Object.fromEntries(robustDecision.allocations.map(a => [a.channel_id, a.budget]));

  const margin = { top: 30, right: 30, bottom: 40, left: 140 };
  const width = container.clientWidth || 550;
  const height = 240;

  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', '100%')
    .attr('height', height);

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left}, ${margin.top})`);

  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const maxBudget = d3.max(channels, ch => Math.max(pointMap[ch] || 0, robustMap[ch] || 0)) * 1.25 || 1000;

  const yScale = d3.scaleBand()
    .domain(channels)
    .range([0, innerHeight])
    .padding(0.3);

  const ySubScale = d3.scaleBand()
    .domain(['point_estimate', 'robust_minimax_regret'])
    .range([0, yScale.bandwidth()])
    .padding(0.15);

  const xScale = d3.scaleLinear()
    .domain([0, maxBudget])
    .range([0, innerWidth]);

  // Legend
  const legend = svg.append('g')
    .attr('transform', `translate(${margin.left}, 12)`);

  legend.append('rect').attr('x', 0).attr('y', 0).attr('width', 12).attr('height', 12).attr('fill', '#38bdf8').attr('rx', 2);
  legend.append('text').attr('x', 16).attr('y', 10).style('font-size', '11px').style('fill', '#cbd5e1').text('Point Estimate Strategy');

  legend.append('rect').attr('x', 160).attr('y', 0).attr('width', 12).attr('height', 12).attr('fill', '#a855f7').attr('rx', 2);
  legend.append('text').attr('x', 176).attr('y', 10).style('font-size', '11px').style('fill', '#cbd5e1').text('Robust Minimax Regret Strategy');

  // Axes
  const yAxis = d3.axisLeft(yScale);
  const xAxis = d3.axisBottom(xScale).ticks(5).tickFormat(d => `$${d3.format(',')(d)}`);

  g.append('g')
    .attr('class', 'axis y-axis')
    .call(yAxis)
    .selectAll('text')
    .style('font-size', '11px')
    .style('fill', '#e2e8f0');

  g.append('g')
    .attr('class', 'axis x-axis')
    .attr('transform', `translate(0, ${innerHeight})`)
    .call(xAxis)
    .selectAll('text')
    .style('font-size', '11px')
    .style('fill', '#94a3b8');

  // Draw grouped bars
  channels.forEach(ch => {
    const pBudget = pointMap[ch] || 0;
    const rBudget = robustMap[ch] || 0;

    // Point bar
    g.append('rect')
      .attr('x', 0)
      .attr('y', yScale(ch) + ySubScale('point_estimate'))
      .attr('width', xScale(pBudget))
      .attr('height', ySubScale.bandwidth())
      .attr('fill', '#38bdf8')
      .attr('rx', 2);

    g.append('text')
      .attr('x', xScale(pBudget) + 6)
      .attr('y', yScale(ch) + ySubScale('point_estimate') + ySubScale.bandwidth() / 2 + 4)
      .style('font-size', '10px')
      .style('font-family', 'var(--font-mono)')
      .style('fill', '#38bdf8')
      .text(`$${pBudget.toFixed(0)}`);

    // Robust bar
    g.append('rect')
      .attr('x', 0)
      .attr('y', yScale(ch) + ySubScale('robust_minimax_regret'))
      .attr('width', xScale(rBudget))
      .attr('height', ySubScale.bandwidth())
      .attr('fill', '#a855f7')
      .attr('rx', 2);

    g.append('text')
      .attr('x', xScale(rBudget) + 6)
      .attr('y', yScale(ch) + ySubScale('robust_minimax_regret') + ySubScale.bandwidth() / 2 + 4)
      .style('font-size', '10px')
      .style('font-family', 'var(--font-mono)')
      .style('fill', '#a855f7')
      .text(`$${rBudget.toFixed(0)}`);
  });
}

/**
 * 5. Value of Information (VOI) Candidate Experiments Table
 */
export function renderExperimentTable(tbodyId, experimentCandidates) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;
  tbody.innerHTML = '';

  if (!experimentCandidates || experimentCandidates.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="text-center" style="padding: 1.5rem; color: #94a3b8;">No experiment candidates generated for this dataset.</td></tr>';
    return;
  }

  tbody.innerHTML = experimentCandidates.map(exp => {
    const regretVal = exp.expected_regret_reduction ? exp.expected_regret_reduction.value.toFixed(2) : '-';
    const voiVal = exp.value_of_information ? exp.value_of_information.value.toFixed(2) : '-';
    const isVoiPositive = exp.value_of_information && exp.value_of_information.value > 0;

    return `
      <tr>
        <td class="font-mono text-cyan">${exp.experiment_id}</td>
        <td><strong>${exp.channel_id}</strong></td>
        <td class="font-mono">${exp.sample_size.toLocaleString()} users</td>
        <td class="font-mono">$${exp.experiment_cost.toLocaleString()}</td>
        <td class="font-mono text-amber">${regretVal} units</td>
        <td class="font-mono ${isVoiPositive ? 'text-emerald' : 'text-muted'} font-bold">$${voiVal}</td>
        <td><span class="badge badge-purple">${exp.selection_method}</span></td>
      </tr>
    `;
  }).join('');
}

/**
 * 6. Provenance & Evaluation Metrics Grid
 */
export function renderProvenanceGrid(containerId, result) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';

  const evalSummary = result.evaluation || {};
  const metrics = evalSummary.metrics || [];

  const metricsHtml = metrics.map(m => `
    <div class="provenance-metric">
      <span class="p-name">${m.name.replace(/_/g, ' ').toUpperCase()}:</span>
      <span class="p-val font-mono text-cyan">${m.value !== null ? m.value : '-'} ${m.unit || ''}</span>
    </div>
  `).join('');

  container.innerHTML = `
    <div class="provenance-card">
      <h3 class="provenance-title">Dataset Lineage</h3>
      <div class="provenance-item"><strong>Dataset ID:</strong> <span class="font-mono">${result.dataset_id}</span></div>
      <div class="provenance-item"><strong>Run ID:</strong> <span class="font-mono">${result.run_id}</span></div>
      <div class="provenance-item"><strong>Generated At:</strong> <span class="font-mono">${result.generated_at}</span></div>
      <div class="provenance-item"><strong>Schema Version:</strong> <span class="font-mono">${result.schema_version}</span></div>
    </div>
    <div class="provenance-card">
      <h3 class="provenance-title">Evaluation Metrics</h3>
      <div class="provenance-item"><strong>Scenario ID:</strong> <span class="font-mono">${evalSummary.scenario_id || 'N/A'}</span></div>
      <div class="provenance-item"><strong>Repetitions:</strong> <span class="font-mono">${evalSummary.repetitions || 1}</span></div>
      <div class="provenance-metrics-list">${metricsHtml || '<span class="text-muted">No explicit evaluation metrics logged.</span>'}</div>
    </div>
    <div class="provenance-card">
      <h3 class="provenance-title">Model Specifications</h3>
      <div class="provenance-item"><strong>Causal Estimator:</strong> AIPW Doubly Robust (Logistic Propensity + OLS/Ridge Outcome Regression)</div>
      <div class="provenance-item"><strong>Sensitivity Model:</strong> Marginal Sensitivity Model (Dorn & Guo 2022) with Calibration</div>
      <div class="provenance-item"><strong>Optimization:</strong> CVXPY Robust Minimax Regret under Uncertainty</div>
    </div>
  `;
}
