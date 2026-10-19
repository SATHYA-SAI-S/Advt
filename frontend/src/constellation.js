import * as d3 from 'd3';

/**
 * Causal Cosmos • Touchpoint Constellation Core Engine
 * 
 * Implements an interactive D3.js celestial star map integrating:
 * 1. ☄️ Comets: Real multi-touch customer journeys streaking through stars
 * 2. 🕳️ Gravity Wells: Confounding distortion horizons that expand with Γ sensitivity
 * 3. 💥 Stellar Heartbeat: Pulsars (stable), Supernovae (fragile), and Dark Matter (unidentifiable)
 * 4. 🪨 Asteroids: Experiment recommendations orbiting fragile stars with VOI velocities
 * 5. 🌊 Nebulae: Volumetric budget allocation clouds (Naive vs Robust Minimax Regret)
 */
export class TouchpointConstellation {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    this.svg = d3.select('#constellation-svg');
    this.canvas = document.getElementById('starfield-canvas');
    this.tooltip = document.getElementById('constellation-tooltip');
    this.options = options;

    this.data = null;
    this.activeModel = 'Causal AIPW'; // Default to Causal AIPW!
    this.activeFilter = 'all';        // all, converted, dropoff
    this.currentGamma = 1.0;          // Sensitivity parameter Γ ∈ [1.0, 3.0]

    this.selectedNodeId = null;
    this.selectedCompareNodeId = null;
    this.selectedArcKey = null;
    this.selectedCometId = null;
    this.isCompareMode = false;
    this.isIsolated = false;

    // Cosmic visual toggles
    this.showLabels = true;
    this.showArcs = true;
    this.showParticles = true;
    this.showHalos = true;
    this.showComets = true;
    this.showGravityWells = true;
    this.showHeartbeat = true;
    this.showAsteroids = true;
    this.showNebulae = true;

    // Dimensions
    this.width = 1200;
    this.height = 680;
    this.transform = d3.zoomIdentity;

    // Simulation & Animation loop states
    this.flowParticles = [];
    this.activeComets = [];
    this.orbitingAsteroids = [];
    this.animationFrameId = null;
    this.lastTickTime = performance.now();

    // Callbacks
    this.onChannelSelect = options.onChannelSelect || (() => {});
    this.onArcSelect = options.onArcSelect || (() => {});
    this.onModelChange = options.onModelChange || (() => {});
    this.onCometSelect = options.onCometSelect || (() => {});
    this.onAsteroidSelect = options.onAsteroidSelect || (() => {});

    this.init();
  }

  init() {
    this.setupResizeObserver();
    this.setupDefs();
    this.setupZoom();
    this.setupStarfieldBackground();
  }

  setupResizeObserver() {
    const updateDimensions = () => {
      const rect = this.container.getBoundingClientRect();
      this.width = Math.max(rect.width, 700);
      this.height = Math.max(rect.height, 500);

      this.canvas.width = this.width;
      this.canvas.height = this.height;

      this.svg
        .attr('viewBox', `0 0 ${this.width} ${this.height}`)
        .attr('width', '100%')
        .attr('height', '100%');

      if (this.data) {
        this.computeLayout();
        this.render();
      }
    };

    const ro = new ResizeObserver(() => updateDimensions());
    ro.observe(this.container);
    updateDimensions();
  }

  setupDefs() {
    this.defs = this.svg.append('defs');

    // Standard cosmic glow filter
    const glowFilter = this.defs.append('filter')
      .attr('id', 'cosmic-glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');

    glowFilter.append('feGaussianBlur')
      .attr('stdDeviation', 4)
      .attr('result', 'coloredBlur');

    const feMerge = glowFilter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Intense supernova flare filter
    const intenseFilter = this.defs.append('filter')
      .attr('id', 'intense-glow')
      .attr('x', '-100%')
      .attr('y', '-100%')
      .attr('width', '300%')
      .attr('height', '300%');

    intenseFilter.append('feGaussianBlur')
      .attr('stdDeviation', 8)
      .attr('result', 'blur1');
    intenseFilter.append('feGaussianBlur')
      .attr('stdDeviation', 16)
      .attr('result', 'blur2');

    const feMerge2 = intenseFilter.append('feMerge');
    feMerge2.append('feMergeNode').attr('in', 'blur2');
    feMerge2.append('feMergeNode').attr('in', 'blur1');
    feMerge2.append('feMergeNode').attr('in', 'SourceGraphic');

    // Gravity well distortion filter (lens blur)
    const wellFilter = this.defs.append('filter')
      .attr('id', 'gravity-blur')
      .attr('x', '-80%')
      .attr('y', '-80%')
      .attr('width', '260%')
      .attr('height', '260%');
    wellFilter.append('feGaussianBlur')
      .attr('stdDeviation', 6)
      .attr('result', 'blur');

    // Nebula volumetric blur filter
    const nebulaFilter = this.defs.append('filter')
      .attr('id', 'nebula-blur')
      .attr('x', '-100%')
      .attr('y', '-100%')
      .attr('width', '300%')
      .attr('height', '300%');
    nebulaFilter.append('feGaussianBlur')
      .attr('stdDeviation', 32)
      .attr('result', 'nebulaBlur');
  }

  setupZoom() {
    this.zoomBehavior = d3.zoom()
      .scaleExtent([0.45, 3.5])
      .on('zoom', (event) => {
        this.transform = event.transform;
        this.gMain.attr('transform', event.transform);
      });

    this.svg.call(this.zoomBehavior)
      .on('dblclick.zoom', null);

    this.gMain = this.svg.append('g').attr('class', 'constellation-world');

    // Layered celestial depth:
    this.gNebulae = this.gMain.append('g').attr('class', 'layer-nebulae');
    this.gGravityWells = this.gMain.append('g').attr('class', 'layer-gravity-wells');
    this.gArcs = this.gMain.append('g').attr('class', 'layer-arcs');
    this.gAsteroidOrbits = this.gMain.append('g').attr('class', 'layer-asteroid-orbits');
    this.gParticles = this.gMain.append('g').attr('class', 'layer-particles');
    this.gComets = this.gMain.append('g').attr('class', 'layer-comets');
    this.gAsteroids = this.gMain.append('g').attr('class', 'layer-asteroids');
    this.gNodes = this.gMain.append('g').attr('class', 'layer-nodes');
  }

  resetZoom() {
    this.svg.transition()
      .duration(750)
      .ease(d3.easeCubicOut)
      .call(this.zoomBehavior.transform, d3.zoomIdentity);
  }

  setupStarfieldBackground() {
    const ctx = this.canvas.getContext('2d');
    const starCount = 200;
    const stars = [];

    for (let i = 0; i < starCount; i++) {
      stars.push({
        x: Math.random() * this.width,
        y: Math.random() * this.height,
        radius: Math.random() * 1.5 + 0.3,
        alpha: Math.random() * 0.7 + 0.15,
        twinkleSpeed: (Math.random() * 0.02 + 0.005) * (Math.random() > 0.5 ? 1 : -1)
      });
    }

    const animateStarfield = () => {
      ctx.clearRect(0, 0, this.width, this.height);

      for (const s of stars) {
        s.alpha += s.twinkleSpeed;
        if (s.alpha > 0.85 || s.alpha < 0.12) {
          s.twinkleSpeed = -s.twinkleSpeed;
        }

        ctx.beginPath();
        ctx.arc(s.x, s.y, s.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${Math.max(0.05, Math.min(0.9, s.alpha))})`;
        ctx.fill();
      }

      requestAnimationFrame(animateStarfield);
    };

    animateStarfield();
  }

  setData(data) {
    this.data = data;
    this.computeLayout();
    this.setupNodeGradients();
    this.render();
    this.initFlowParticles();
    this.initCometsEngine();
    this.initAsteroidsEngine();
  }

  computeLayout() {
    if (!this.data) return;

    const cx = this.width / 2;
    const cy = this.height / 2;
    const rx = this.width * 0.35;
    const ry = this.height * 0.33;

    // Filter channel nodes vs endpoint nodes
    const channelNodes = this.data.nodes.filter(n => !n.is_endpoint);
    const n = channelNodes.length;

    // Dynamic celestial elliptical distribution for any number of channels (3, 5, or 7)
    channelNodes.forEach((node, idx) => {
      const angle = (idx / n) * 2 * Math.PI - Math.PI / 2;
      node.x = cx + rx * Math.cos(angle) * 0.95;
      node.y = cy + ry * Math.sin(angle) * 0.90;
    });

    // Endpoints in distinctive focal celestial positions
    const endpointMap = {
      "Conversion": { x: cx + rx * 0.82, y: cy + ry * 0.78 },
      "Drop-off":   { x: cx - rx * 0.82, y: cy + ry * 0.78 }
    };

    this.data.nodes.forEach(node => {
      if (node.is_endpoint && endpointMap[node.id]) {
        node.x = endpointMap[node.id].x;
        node.y = endpointMap[node.id].y;
      }
    });

    this.nodeMap = new Map(this.data.nodes.map(n => [n.id, n]));
  }

  setupNodeGradients() {
    this.data.nodes.forEach(node => {
      const gradId = `grad-${node.id.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase()}`;
      if (document.getElementById(gradId)) return;

      const grad = this.defs.append('radialGradient')
        .attr('id', gradId)
        .attr('cx', '35%')
        .attr('cy', '35%')
        .attr('r', '65%');

      grad.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', '#ffffff');

      grad.append('stop')
        .attr('offset', '40%')
        .attr('stop-color', node.color || '#38bdf8');

      grad.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', d3.color(node.color || '#38bdf8').darker(1.6).toString());
    });
  }

  render() {
    if (!this.data) return;

    this.renderNebulae();
    this.renderGravityWells();
    this.renderArcs();
    this.renderAsteroidOrbits();
    this.renderNodes();
    this.updateVisibility();
  }

  // =========================================================================
  // 1. NEBULAE (Capital Allocation Gas Clouds)
  // =========================================================================
  renderNebulae() {
    this.gNebulae.selectAll('*').remove();
    if (!this.showNebulae || !this.data.nebulae) return;

    const neb = this.data.nebulae;
    const channelNodes = this.data.nodes.filter(n => !n.is_endpoint);

    // Compute center of mass for naive vs robust allocations
    let naiveCx = 0, naiveCy = 0, naiveTotal = 0;
    let robustCx = 0, robustCy = 0, robustTotal = 0;

    channelNodes.forEach(node => {
      const nAlloc = (neb.naive_cloud.allocations && neb.naive_cloud.allocations[node.id]) || (node.budget && node.budget.naive) || 1000;
      const rAlloc = (neb.robust_cloud.allocations && neb.robust_cloud.allocations[node.id]) || (node.budget && node.budget.robust) || 1000;

      naiveCx += node.x * nAlloc;
      naiveCy += node.y * nAlloc;
      naiveTotal += nAlloc;

      robustCx += node.x * rAlloc;
      robustCy += node.y * rAlloc;
      robustTotal += rAlloc;
    });

    naiveCx = naiveTotal > 0 ? naiveCx / naiveTotal : this.width / 2;
    naiveCy = naiveTotal > 0 ? naiveCy / naiveTotal : this.height / 2;

    robustCx = robustTotal > 0 ? robustCx / robustTotal : this.width / 2;
    robustCy = robustTotal > 0 ? robustCy / robustTotal : this.height / 2;

    // Render Naive Capital Nebula (Cyan/Blue cloud)
    this.gNebulae.append('ellipse')
      .attr('class', 'nebula-cloud naive-nebula')
      .attr('cx', naiveCx)
      .attr('cy', naiveCy)
      .attr('rx', this.width * 0.28)
      .attr('ry', this.height * 0.26)
      .attr('fill', neb.naive_cloud.color || 'rgba(56, 189, 248, 0.16)')
      .attr('filter', 'url(#nebula-blur)')
      .attr('opacity', 0.85);

    // Render Robust Minimax Regret Nebula (Crimson/Rose cloud)
    this.gNebulae.append('ellipse')
      .attr('class', 'nebula-cloud robust-nebula')
      .attr('cx', robustCx)
      .attr('cy', robustCy)
      .attr('rx', this.width * 0.25)
      .attr('ry', this.height * 0.24)
      .attr('fill', neb.robust_cloud.color || 'rgba(244, 63, 94, 0.16)')
      .attr('filter', 'url(#nebula-blur)')
      .attr('opacity', 0.85);
  }

  // =========================================================================
  // 2. GRAVITY WELLS (Confounding Distortion Horizons)
  // =========================================================================
  renderGravityWells() {
    this.gGravityWells.selectAll('*').remove();
    if (!this.showGravityWells) return;

    const channelNodes = this.data.nodes.filter(n => !n.is_endpoint);

    channelNodes.forEach(node => {
      const gw = node.gravity_well || { base_radius: 35, intensity: 1.0, is_negative: false };
      const gGroup = this.gGravityWells.append('g')
        .attr('class', `gravity-well-group well-${node.id.replace(/[^a-zA-Z0-9]/g, '-')}`)
        .attr('transform', `translate(${node.x}, ${node.y})`);

      // Dynamic scaling with current Γ sensitivity parameter
      const gammaScale = Math.max(0.7, this.currentGamma / 1.0);
      const radius = (gw.base_radius || 35) * gammaScale;
      const isPastBreakdown = this.currentGamma >= (node.breakdown_gamma || 3.0);

      // Color scheme based on stability & breakdown
      let wellColor = 'rgba(56, 189, 248, 0.22)';
      let rimColor = '#38bdf8';

      if (isPastBreakdown || gw.is_negative) {
        wellColor = 'rgba(244, 63, 94, 0.32)';
        rimColor = '#f43f5e';
      } else if (node.stability_label === 'fragile') {
        wellColor = 'rgba(251, 191, 36, 0.24)';
        rimColor = '#fbbf24';
      } else if (node.stability_label === 'stable') {
        wellColor = 'rgba(16, 185, 129, 0.22)';
        rimColor = '#10b981';
      }

      // Outer Gravitational Horizon Ring (pulsating dashed ring)
      gGroup.append('circle')
        .attr('class', 'horizon-ring outer-horizon')
        .attr('r', radius * 1.35)
        .attr('fill', 'none')
        .attr('stroke', rimColor)
        .attr('stroke-dasharray', '4, 6')
        .attr('stroke-width', 1.2)
        .attr('opacity', 0.45);

      // Mid Lensing Distort Ring
      gGroup.append('circle')
        .attr('class', 'horizon-ring mid-horizon')
        .attr('r', radius)
        .attr('fill', wellColor)
        .attr('stroke', rimColor)
        .attr('stroke-width', isPastBreakdown ? 2.2 : 1.4)
        .attr('filter', 'url(#gravity-blur)')
        .attr('opacity', 0.65);

      // Core event boundary
      gGroup.append('circle')
        .attr('class', 'horizon-ring inner-horizon')
        .attr('r', radius * 0.65)
        .attr('fill', 'none')
        .attr('stroke', rimColor)
        .attr('stroke-dasharray', '2, 3')
        .attr('stroke-width', 1.0)
        .attr('opacity', 0.5);
    });
  }

  // =========================================================================
  // 3. ARCS (Gravitational Connectors)
  // =========================================================================
  renderArcs() {
    const activeArcs = this.getFilteredArcs();
    const maxVol = d3.max(activeArcs, d => d.volume) || 100;
    const strokeScale = d3.scaleLinear()
      .domain([10, maxVol])
      .range([1.8, 7.0])
      .clamp(true);

    const arcSel = this.gArcs.selectAll('.journey-arc')
      .data(activeArcs, d => `${d.source}->${d.target}`);

    arcSel.exit().remove();

    const arcEnter = arcSel.enter().append('path')
      .attr('class', 'journey-arc')
      .attr('fill', 'none')
      .attr('stroke-linecap', 'round')
      .attr('opacity', 0);

    const arcMerged = arcEnter.merge(arcSel);

    arcMerged
      .attr('d', d => this.calculateBézierPath(d))
      .attr('stroke-width', d => strokeScale(d.volume))
      .attr('stroke', d => {
        if (d.target === 'Conversion') return 'rgba(16, 185, 129, 0.45)';
        if (d.target === 'Drop-off') return 'rgba(100, 116, 139, 0.28)';
        const srcNode = this.nodeMap.get(d.source);
        return srcNode ? d3.color(srcNode.color).copy({ opacity: 0.35 }).toString() : 'rgba(148, 163, 184, 0.35)';
      })
      .attr('cursor', 'pointer')
      .transition()
      .duration(400)
      .attr('opacity', 0.65);

    arcMerged
      .on('mouseenter', (event, d) => this.handleArcHover(event, d))
      .on('mousemove', (event) => this.positionTooltip(event))
      .on('mouseleave', () => this.hideTooltip())
      .on('click', (event, d) => this.handleArcClick(event, d));

    this.arcElements = arcMerged;
  }

  calculateBézierPath(arc) {
    const src = this.nodeMap.get(arc.source);
    const dst = this.nodeMap.get(arc.target);
    if (!src || !dst) return '';

    const dx = dst.x - src.x;
    const dy = dst.y - src.y;
    const dist = Math.hypot(dx, dy);

    if (arc.source === arc.target) {
      const r = 26;
      return `M ${src.x} ${src.y - 10} C ${src.x - r * 1.5} ${src.y - r * 2.2}, ${src.x + r * 1.5} ${src.y - r * 2.2}, ${src.x + 8} ${src.y - 12}`;
    }

    const midX = (src.x + dst.x) / 2;
    const midY = (src.y + dst.y) / 2;
    const nx = -dy / dist;
    const ny = dx / dist;

    const curvature = Math.min(dist * 0.18, 55);
    const cx = midX + nx * curvature;
    const cy = midY + ny * curvature;

    return `M ${src.x} ${src.y} Q ${cx} ${cy} ${dst.x} ${dst.y}`;
  }

  getFilteredArcs() {
    if (!this.data) return [];
    if (this.activeFilter === 'converted') {
      return this.data.arcs.filter(a => a.target !== 'Drop-off');
    }
    if (this.activeFilter === 'dropoff') {
      return this.data.arcs.filter(a => a.target !== 'Conversion');
    }
    return this.data.arcs;
  }

  // =========================================================================
  // 4. ASTEROID ORBITS (Experiment Candidate System)
  // =========================================================================
  renderAsteroidOrbits() {
    this.gAsteroidOrbits.selectAll('*').remove();
    this.gAsteroids.selectAll('*').remove();
    if (!this.showAsteroids) return;

    this.orbitingAsteroids = [];
    const channelNodes = this.data.nodes.filter(n => !n.is_endpoint && n.asteroids && n.asteroids.length > 0);

    channelNodes.forEach(node => {
      node.asteroids.forEach(ast => {
        const orbitR = ast.orbit_radius || 36;

        // Render dashed orbital track
        this.gAsteroidOrbits.append('circle')
          .attr('cx', node.x)
          .attr('cy', node.y)
          .attr('r', orbitR)
          .attr('fill', 'none')
          .attr('stroke', 'rgba(255, 255, 255, 0.15)')
          .attr('stroke-dasharray', '2, 4')
          .attr('stroke-width', 1.0);

        // Track state for animation
        this.orbitingAsteroids.push({
          data: ast,
          node: node,
          radius: orbitR,
          angle: Math.random() * Math.PI * 2,
          speed: ast.orbit_speed || 0.02,
          size: ast.size || 5,
          color: ast.color || '#fbbf24'
        });
      });
    });

    // Render asteroid elements
    this.asteroidElements = this.gAsteroids.selectAll('.orbiting-asteroid')
      .data(this.orbitingAsteroids)
      .enter()
      .append('g')
      .attr('class', 'orbiting-asteroid')
      .attr('cursor', 'pointer')
      .on('mouseenter', (event, d) => this.handleAsteroidHover(event, d.data, d.node))
      .on('mousemove', (event) => this.positionTooltip(event))
      .on('mouseleave', () => this.hideTooltip())
      .on('click', (event, d) => this.handleAsteroidClick(event, d.data, d.node));

    this.asteroidElements.append('circle')
      .attr('class', 'asteroid-body')
      .attr('r', d => d.size)
      .attr('fill', d => d.color)
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 0.8)
      .attr('filter', 'url(#cosmic-glow)');
  }

  // =========================================================================
  // 5. NODES / STARS (Pulsar / Supernova / Dark Matter Heartbeat)
  // =========================================================================
  renderNodes() {
    const nodeRadiusScale = d3.scaleSqrt()
      .domain([0, 0.40])
      .range([18, 48])
      .clamp(true);

    const nodeSel = this.gNodes.selectAll('.constellation-node')
      .data(this.data.nodes, d => d.id);

    nodeSel.exit().remove();

    const nodeEnter = nodeSel.enter().append('g')
      .attr('class', d => `constellation-node ${d.is_endpoint ? 'endpoint-node' : 'channel-node'} type-${d.phenomenon_type || 'standard'}`)
      .attr('transform', d => `translate(${d.x}, ${d.y})`)
      .attr('cursor', 'pointer');

    // Pulsar Rotating Lighthouse Beams (for stable channels)
    const beamGroup = nodeEnter.filter(d => d.phenomenon_type === 'pulsar')
      .append('g')
      .attr('class', 'pulsar-beam-container');

    beamGroup.append('polygon')
      .attr('class', 'pulsar-beam beam-north')
      .attr('points', '-4,0 4,0 12,-65 -12,-65')
      .attr('fill', 'url(#cosmic-glow)')
      .attr('opacity', 0.25);

    beamGroup.append('polygon')
      .attr('class', 'pulsar-beam beam-south')
      .attr('points', '-4,0 4,0 12,65 -12,65')
      .attr('fill', 'url(#cosmic-glow)')
      .attr('opacity', 0.25);

    // Supernova Turbulent Flare Coronas (for fragile channels)
    nodeEnter.filter(d => d.phenomenon_type === 'supernova')
      .append('circle')
      .attr('class', 'supernova-corona')
      .attr('fill', 'none')
      .attr('stroke', '#f43f5e')
      .attr('stroke-width', 2.0)
      .attr('opacity', 0.7);

    // Uncertainty Halo (95% CI Range)
    nodeEnter.append('circle')
      .attr('class', 'uncertainty-halo')
      .attr('fill', 'none')
      .attr('stroke', d => d.color || '#38bdf8')
      .attr('stroke-dasharray', '3,3')
      .attr('stroke-width', 1.5)
      .attr('opacity', 0.6);

    // Star Aura Ring
    nodeEnter.append('circle')
      .attr('class', 'star-aura')
      .attr('fill', 'none')
      .attr('stroke', d => d.color || '#38bdf8')
      .attr('stroke-width', 1.5)
      .attr('opacity', 0.35);

    // Core Star Circle
    nodeEnter.append('circle')
      .attr('class', 'star-core')
      .attr('filter', d => d.phenomenon_type === 'supernova' ? 'url(#intense-glow)' : 'url(#cosmic-glow)');

    // Metric Text
    nodeEnter.append('text')
      .attr('class', 'node-metric-text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', '#ffffff')
      .attr('font-size', '11px')
      .attr('font-family', 'var(--font-mono)')
      .attr('font-weight', '700')
      .attr('pointer-events', 'none');

    // Phenomenon Badge Text (PULSAR / SUPERNOVA / DARK MATTER)
    nodeEnter.filter(d => !d.is_endpoint)
      .append('text')
      .attr('class', 'phenomenon-tag')
      .attr('text-anchor', 'middle')
      .attr('fill', d => d.stability_label === 'stable' ? '#10b981' : d.stability_label === 'fragile' ? '#fbbf24' : '#94a3b8')
      .attr('font-size', '9px')
      .attr('font-family', 'var(--font-mono)')
      .attr('font-weight', '700')
      .attr('pointer-events', 'none');

    // Channel Name Label
    nodeEnter.append('text')
      .attr('class', 'node-name-label')
      .attr('text-anchor', 'middle')
      .attr('fill', '#e2e8f0')
      .attr('font-size', '12px')
      .attr('font-family', 'var(--font-sans)')
      .attr('font-weight', '600')
      .attr('pointer-events', 'none')
      .attr('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.9))');

    const nodeMerged = nodeEnter.merge(nodeSel);
    this.updateNodeVisuals(nodeMerged, 650);

    nodeMerged
      .on('mouseenter', (event, d) => this.handleNodeHover(event, d))
      .on('mousemove', (event) => this.positionTooltip(event))
      .on('mouseleave', () => this.hideTooltip())
      .on('click', (event, d) => this.handleNodeClick(event, d));

    this.nodeElements = nodeMerged;
  }

  updateNodeVisuals(selection, duration = 600) {
    const nodeRadiusScale = d3.scaleSqrt()
      .domain([0, 0.40])
      .range([18, 48])
      .clamp(true);

    const t = d3.transition().duration(duration).ease(d3.easeCubicOut);

    selection.each((d, i, nodes) => {
      const g = d3.select(nodes[i]);
      let radius = 22;
      let haloRadius = 30;
      let sharePct = '';

      if (!d.is_endpoint) {
        const modelData = (d.models && d.models[this.activeModel]) || (d.models && d.models['Causal AIPW']) || { credit_share: 0.15 };
        const share = modelData.credit_share || 0.15;
        radius = nodeRadiusScale(share);
        sharePct = `${(share * 100).toFixed(1)}%`;

        const ciWidth = (d.ci && d.ci.length === 2) ? Math.abs(d.ci[1] - d.ci[0]) : 0.05;
        haloRadius = radius + Math.max(8, ciWidth * 180);
      } else {
        radius = d.id === 'Conversion' ? 30 : 22;
        sharePct = d.id === 'Conversion' ? 'CONV' : 'EXIT';
        haloRadius = radius + 6;
      }

      const gradId = `url(#grad-${d.id.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase()})`;
      g.select('.star-core')
        .transition(t)
        .attr('r', radius)
        .attr('fill', gradId);

      g.select('.star-aura')
        .transition(t)
        .attr('r', radius + 5);

      g.select('.uncertainty-halo')
        .transition(t)
        .attr('r', haloRadius);

      g.select('.supernova-corona')
        .transition(t)
        .attr('r', radius + 10);

      g.select('.node-metric-text')
        .text(sharePct)
        .transition(t)
        .attr('font-size', radius > 26 ? '12px' : '10px');

      if (!d.is_endpoint) {
        const tag = d.phenomenon_type ? d.phenomenon_type.toUpperCase() : d.stability_label.toUpperCase();
        g.select('.phenomenon-tag')
          .text(tag)
          .transition(t)
          .attr('dy', -(radius + 8));
      }

      g.select('.node-name-label')
        .text(d.name)
        .transition(t)
        .attr('dy', radius + 18);
    });
  }

  // =========================================================================
  // 6. COMETS ENGINE (Live Customer Journey Streaks)
  // =========================================================================
  initCometsEngine() {
    if (!this.data || !this.data.comets) return;

    this.activeComets = [];
    const cometCount = Math.min(12, this.data.comets.length);

    for (let i = 0; i < cometCount; i++) {
      const cometData = this.data.comets[i % this.data.comets.length];
      this.activeComets.push(this.spawnComet(cometData, Math.random()));
    }
  }

  spawnComet(cometData, initialProgress = 0) {
    // Build path nodes
    const pathNodes = cometData.touchpoints.map(ch => this.nodeMap.get(ch)).filter(Boolean);
    const endNode = this.nodeMap.get(cometData.target_endpoint);
    if (endNode) pathNodes.push(endNode);

    return {
      data: cometData,
      nodes: pathNodes,
      progress: initialProgress,
      speed: (cometData.speed || 1.0) * 0.0028,
      history: [],
      maxHistory: Math.max(8, (cometData.tail_length || 2) * 5)
    };
  }

  // =========================================================================
  // ANIMATION LOOP (Particles, Comets, Asteroids, Pulsar Beams)
  // =========================================================================
  initFlowParticles() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }

    const activeArcs = this.getFilteredArcs();
    this.flowParticles = [];
    for (let i = 0; i < 40; i++) {
      const arc = activeArcs[Math.floor(Math.random() * activeArcs.length)];
      if (!arc) continue;
      this.flowParticles.push({
        arc: arc,
        progress: Math.random(),
        speed: 0.003 + Math.random() * 0.004,
        size: 2.2 + Math.random() * 1.5,
        color: arc.target === 'Conversion' ? '#34d399' : '#ffffff'
      });
    }

    const animate = (time) => {
      // 1. Flow particles along arcs
      if (this.showParticles) {
        this.updateParticles();
      }

      // 2. Customer Journey Comets
      if (this.showComets) {
        this.updateComets();
      }

      // 3. Orbiting Asteroids
      if (this.showAsteroids) {
        this.updateAsteroids();
      }

      // 4. Pulsar Rotating Lighthouse Beams
      if (this.showHeartbeat) {
        this.updateHeartbeat(time);
      }

      this.animationFrameId = requestAnimationFrame(animate);
    };

    this.animationFrameId = requestAnimationFrame(animate);
  }

  updateParticles() {
    const pSel = this.gParticles.selectAll('.flow-particle')
      .data(this.flowParticles);

    pSel.exit().remove();

    const pMerged = pSel.enter().append('circle')
      .attr('class', 'flow-particle')
      .attr('r', d => d.size)
      .attr('fill', d => d.color)
      .attr('filter', 'url(#cosmic-glow)')
      .merge(pSel);

    this.flowParticles.forEach((p, idx) => {
      p.progress += p.speed;
      if (p.progress >= 1.0) {
        p.progress = 0;
        const arcs = this.getFilteredArcs();
        p.arc = arcs[Math.floor(Math.random() * arcs.length)];
      }

      const src = this.nodeMap.get(p.arc.source);
      const dst = this.nodeMap.get(p.arc.target);
      if (src && dst) {
        const pos = this.getPointOnBézier(src, dst, p.progress);
        d3.select(pMerged.nodes()[idx])
          .attr('cx', pos.x)
          .attr('cy', pos.y)
          .attr('opacity', Math.sin(p.progress * Math.PI) * 0.85);
      }
    });
  }

  updateComets() {
    if (!this.activeComets || this.activeComets.length === 0) return;

    this.gComets.selectAll('*').remove();

    this.activeComets.forEach(c => {
      if (c.nodes.length < 2) return;

      c.progress += c.speed;
      if (c.progress >= 1.0) {
        c.progress = 0;
        c.history = [];
        // Swap to random comet journey
        if (this.data.comets.length > 0) {
          const randData = this.data.comets[Math.floor(Math.random() * this.data.comets.length)];
          const newComet = this.spawnComet(randData, 0);
          c.data = newComet.data;
          c.nodes = newComet.nodes;
          c.speed = newComet.speed;
          c.maxHistory = newComet.maxHistory;
        }
      }

      // Calculate position along piecewise multi-hop path
      const totalSegments = c.nodes.length - 1;
      const segProg = c.progress * totalSegments;
      const segIdx = Math.min(Math.floor(segProg), totalSegments - 1);
      const segT = segProg - segIdx;

      const src = c.nodes[segIdx];
      const dst = c.nodes[segIdx + 1];
      const pos = this.getPointOnBézier(src, dst, segT);

      // Track tail history
      c.history.push({ x: pos.x, y: pos.y });
      if (c.history.length > c.maxHistory) c.history.shift();

      // Render blazing tail
      if (c.history.length > 1) {
        const lineGen = d3.line().x(d => d.x).y(d => d.y).curve(d3.curveBasis);
        this.gComets.append('path')
          .attr('class', 'comet-tail')
          .attr('d', lineGen(c.history))
          .attr('fill', 'none')
          .attr('stroke', c.data.converted === 1 ? '#10b981' : '#f472b6')
          .attr('stroke-width', 2.8)
          .attr('opacity', 0.65)
          .attr('filter', 'url(#cosmic-glow)');
      }

      // Render comet glowing nucleus head
      const headColor = c.data.converted === 1 ? '#34d399' : '#f43f5e';
      const headGroup = this.gComets.append('g')
        .attr('class', 'comet-head')
        .attr('transform', `translate(${pos.x}, ${pos.y})`)
        .attr('cursor', 'pointer')
        .on('click', () => this.handleCometClick(c.data));

      headGroup.append('circle')
        .attr('r', 4.5)
        .attr('fill', '#ffffff')
        .attr('stroke', headColor)
        .attr('stroke-width', 1.8)
        .attr('filter', 'url(#intense-glow)');
    });
  }

  updateAsteroids() {
    if (!this.orbitingAsteroids || this.orbitingAsteroids.length === 0) return;

    this.orbitingAsteroids.forEach((ast, idx) => {
      ast.angle += ast.speed;
      const x = ast.node.x + Math.cos(ast.angle) * ast.radius;
      const y = ast.node.y + Math.sin(ast.angle) * ast.radius;

      if (this.asteroidElements && this.asteroidElements.nodes()[idx]) {
        d3.select(this.asteroidElements.nodes()[idx])
          .attr('transform', `translate(${x}, ${y})`);
      }
    });
  }

  updateHeartbeat(time) {
    // Rotate pulsar lighthouse beams
    const rot = (time * 0.08) % 360;
    this.gNodes.selectAll('.pulsar-beam-container')
      .attr('transform', `rotate(${rot})`);

    // Pulsate supernova coronal ring
    const scale = 1.0 + Math.sin(time * 0.008) * 0.22;
    this.gNodes.selectAll('.supernova-corona')
      .attr('transform', `scale(${scale})`);
  }

  getPointOnBézier(src, dst, t) {
    if (src.id === dst.id) {
      const r = 26;
      return {
        x: src.x + Math.cos(t * Math.PI * 2) * r,
        y: src.y - 12 + Math.sin(t * Math.PI * 2) * r
      };
    }

    const dx = dst.x - src.x;
    const dy = dst.y - src.y;
    const dist = Math.hypot(dx, dy);

    const midX = (src.x + dst.x) / 2;
    const midY = (src.y + dst.y) / 2;
    const nx = -dy / dist;
    const ny = dx / dist;
    const curvature = Math.min(dist * 0.18, 55);

    const cx = midX + nx * curvature;
    const cy = midY + ny * curvature;

    const u = 1 - t;
    const x = u * u * src.x + 2 * u * t * cx + t * t * dst.x;
    const y = u * u * src.y + 2 * u * t * cy + t * t * dst.y;

    return { x, y };
  }

  // =========================================================================
  // INTERACTIVE CONTROLS & EVENT HANDLERS
  // =========================================================================
  setGamma(gammaValue) {
    this.currentGamma = parseFloat(gammaValue);
    this.renderGravityWells();
  }

  setModel(modelName) {
    this.activeModel = modelName;
    if (this.nodeElements) {
      this.updateNodeVisuals(this.nodeElements, 650);
    }
  }

  setFilter(filterName) {
    this.activeFilter = filterName;
    this.renderArcs();
    this.initFlowParticles();
  }

  toggleLabels(visible) {
    this.showLabels = visible;
    this.gNodes.selectAll('.node-name-label, .phenomenon-tag').classed('hidden', !visible);
  }

  toggleArcs(visible) {
    this.showArcs = visible;
    this.gArcs.classed('hidden', !visible);
  }

  toggleParticles(visible) {
    this.showParticles = visible;
    this.gParticles.classed('hidden', !visible);
  }

  toggleHalos(visible) {
    this.showHalos = visible;
    this.gNodes.selectAll('.uncertainty-halo').classed('hidden', !visible);
  }

  toggleComets(visible) {
    this.showComets = visible;
    this.gComets.classed('hidden', !visible);
  }

  toggleGravityWells(visible) {
    this.showGravityWells = visible;
    this.gGravityWells.classed('hidden', !visible);
  }

  toggleHeartbeat(visible) {
    this.showHeartbeat = visible;
    this.gNodes.selectAll('.pulsar-beam-container, .supernova-corona').classed('hidden', !visible);
  }

  toggleAsteroids(visible) {
    this.showAsteroids = visible;
    this.gAsteroids.classed('hidden', !visible);
    this.gAsteroidOrbits.classed('hidden', !visible);
  }

  toggleNebulae(visible) {
    this.showNebulae = visible;
    this.gNebulae.classed('hidden', !visible);
  }

  handleNodeHover(event, d) {
    if (d.is_endpoint) {
      this.tooltip.innerHTML = `
        <div class="tooltip-header">
          <span class="tooltip-title">${d.name}</span>
          <span class="tooltip-badge" style="background: ${d.color}; color: #000">${d.category}</span>
        </div>
        <div class="tooltip-row">
          <span class="t-label">Outcome Role:</span>
          <span class="t-val">${d.role}</span>
        </div>
      `;
    } else {
      const model = (d.models && d.models[this.activeModel]) || (d.models && d.models['Causal AIPW']) || {};
      const sharePct = ((model.credit_share || 0) * 100).toFixed(1);
      const isPastBreakdown = this.currentGamma >= (d.breakdown_gamma || 3.0);

      this.tooltip.innerHTML = `
        <div class="tooltip-header">
          <span class="tooltip-title">${d.name}</span>
          <span class="tooltip-badge" style="background: ${d.color}; color: #000">${d.category}</span>
        </div>
        <div class="tooltip-row">
          <span class="t-label">Phenomenon:</span>
          <span class="t-val text-cyan font-bold">${(d.phenomenon_type || d.stability_label).toUpperCase()}</span>
        </div>
        <div class="tooltip-row">
          <span class="t-label">Active Model (${this.activeModel}):</span>
          <span class="t-val font-bold">${sharePct}% Share</span>
        </div>
        <div class="tooltip-row">
          <span class="t-label">Causal ATE (AIPW):</span>
          <span class="t-val text-emerald font-bold">${d.causal_ate !== undefined ? (d.causal_ate > 0 ? '+' : '') + (d.causal_ate * 100).toFixed(2) + '%' : 'N/A'}</span>
        </div>
        <div class="tooltip-row">
          <span class="t-label">Breakdown Point (Γ*):</span>
          <span class="t-val ${isPastBreakdown ? 'text-rose font-bold' : 'text-amber'}">Γ* = ${d.breakdown_gamma} ${isPastBreakdown ? '⚠️ COLLAPSED' : '✓ Active'}</span>
        </div>
        <div class="tooltip-row">
          <span class="t-label">Minimax Robust Budget:</span>
          <span class="t-val text-cyan">$${(d.budget && d.budget.robust) ? d.budget.robust.toLocaleString() : 'N/A'}</span>
        </div>
        <div class="tooltip-sub" style="margin-top: 6px; font-size: 0.72rem; color: #94a3b8">
          ${d.gravity_well ? d.gravity_well.narrative : d.role}
        </div>
      `;
    }

    this.tooltip.classList.remove('hidden');
    this.positionTooltip(event);
  }

  handleArcHover(event, d) {
    this.tooltip.innerHTML = `
      <div class="tooltip-header">
        <span class="tooltip-title">Gravitational Connector</span>
        <span class="tooltip-badge badge-purple">${d.volume} Transitions</span>
      </div>
      <div class="tooltip-row">
        <span class="t-label">Origin:</span>
        <span class="t-val">${d.source}</span>
      </div>
      <div class="tooltip-row">
        <span class="t-label">Destination:</span>
        <span class="t-val">${d.target}</span>
      </div>
    `;
    this.tooltip.classList.remove('hidden');
    this.positionTooltip(event);
  }

  handleAsteroidHover(event, ast, node) {
    this.tooltip.innerHTML = `
      <div class="tooltip-header">
        <span class="tooltip-title">🪨 Experiment Asteroid</span>
        <span class="tooltip-badge badge-amber">VOI Rank #${ast.priority}</span>
      </div>
      <div class="tooltip-row">
        <span class="t-label">Mission:</span>
        <span class="t-val">${ast.name}</span>
      </div>
      <div class="tooltip-row">
        <span class="t-label">Recommended Sample:</span>
        <span class="t-val font-bold">${ast.sample_size.toLocaleString()} users</span>
      </div>
      <div class="tooltip-row">
        <span class="t-label">Experiment Cost:</span>
        <span class="t-val text-rose">$${ast.cost.toLocaleString()}</span>
      </div>
      <div class="tooltip-row">
        <span class="t-label">Net Value of Information:</span>
        <span class="t-val text-emerald font-bold">$${ast.voi}</span>
      </div>
      <div style="font-size: 0.7rem; color: #cbd5e1; margin-top: 4px">Click asteroid to open mission control</div>
    `;
    this.tooltip.classList.remove('hidden');
    this.positionTooltip(event);
  }

  handleNodeClick(event, d) {
    event.stopPropagation();
    this.selectedNodeId = d.id;
    this.onChannelSelect(d);
  }

  handleArcClick(event, d) {
    event.stopPropagation();
    this.onArcSelect(d);
  }

  handleAsteroidClick(event, ast, node) {
    event.stopPropagation();
    this.onAsteroidSelect({ asteroid: ast, node: node });
  }

  handleCometClick(cometData) {
    this.selectedCometId = cometData.id;
    this.onCometSelect(cometData);
  }

  positionTooltip(event) {
    const pad = 14;
    let x = event.clientX + pad;
    let y = event.clientY + pad;

    const tRect = this.tooltip.getBoundingClientRect();
    if (x + tRect.width > window.innerWidth) x = event.clientX - tRect.width - pad;
    if (y + tRect.height > window.innerHeight) y = event.clientY - tRect.height - pad;

    this.tooltip.style.left = `${x}px`;
    this.tooltip.style.top = `${y}px`;
  }

  hideTooltip() {
    this.tooltip.classList.add('hidden');
  }

  highlightJourneyPath(touchpoints) {
    if (!touchpoints || touchpoints.length === 0) return;
    this.gArcs.selectAll('.journey-arc')
      .attr('opacity', arc => {
        for (let i = 0; i < touchpoints.length - 1; i++) {
          if (arc.source === touchpoints[i] && arc.target === touchpoints[i + 1]) {
            return 1.0;
          }
        }
        return 0.15;
      })
      .attr('stroke', arc => {
        for (let i = 0; i < touchpoints.length - 1; i++) {
          if (arc.source === touchpoints[i] && arc.target === touchpoints[i + 1]) {
            return '#f43f5e';
          }
        }
        return 'rgba(100, 116, 139, 0.2)';
      });
  }
}
