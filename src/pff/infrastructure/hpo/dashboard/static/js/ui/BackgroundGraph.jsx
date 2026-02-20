/**
 * Provide BackgroundGraph module functionality for the HPO dashboard.
 */

import { useRef, useEffect } from "react";
import { useTheme } from "./ThemeContext.jsx";

/**
 * BackgroundGraph
 *
 * Renders a dynamic KG-like network animation:
 * - Community clusters (embedding-like topology)
 * - Stronger intra-cluster links than inter-cluster
 * - Connections that continuously appear/disappear
 * - Edge pulses to suggest message passing
 */
const BackgroundGraph = () => {
  const canvasRef = useRef(null);
  const { theme } = useTheme();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    let animationFrameId = 0;
    let lastTs = 0;
    let rewireAccum = 0;
    let nodes = [];
    let clusters = [];
    const edges = new Map();
    let pulses = [];
    const mouse = {
      x: window.innerWidth * 0.5,
      y: window.innerHeight * 0.5,
      targetX: window.innerWidth * 0.5,
      targetY: window.innerHeight * 0.5,
      active: false,
      influence: 0,
    };

    const prefersReducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;

    const isLight = theme === "light";
    const palette = isLight
      ? {
          nodeCore: [30, 41, 59],
          nodeHalo: [59, 130, 246],
          clusterHalo: [59, 130, 246],
          edgeIntra: [51, 65, 85],
          edgeInter: [51, 65, 85],
          pulse: [37, 99, 235],
        }
      : {
          nodeCore: [148, 163, 184],
          nodeHalo: [99, 102, 241],
          clusterHalo: [67, 56, 202],
          edgeIntra: [125, 211, 252],
          edgeInter: [125, 211, 252],
          pulse: [129, 140, 248],
        };

    const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
    const randomBetween = (min, max) => min + Math.random() * (max - min);
    const rgba = (rgb, a) => `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${a})`;
    const distanceNodes = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
    const distancePoints = (ax, ay, bx, by) => Math.hypot(ax - bx, ay - by);

    const getNodeCount = () =>
      clamp(
        Math.round((canvas.width * canvas.height) / 32000),
        prefersReducedMotion ? 48 : 64,
        prefersReducedMotion ? 120 : 180
      );
    const getClusterCount = () =>
      clamp(
        Math.round((canvas.width * canvas.height) / (prefersReducedMotion ? 130000 : 90000)),
        prefersReducedMotion ? 8 : 12,
        prefersReducedMotion ? 24 : 36
      );
    const getIntraDistance = () => clamp(Math.min(canvas.width, canvas.height) * 0.24, 130, 240);
    const getInterDistance = () => clamp(Math.min(canvas.width, canvas.height) * 0.17, 96, 170);
    const rewireIntervalSec = prefersReducedMotion ? 1.15 : 0.48;
    const getMouseInfluenceRadius = () =>
      clamp(Math.min(canvas.width, canvas.height) * 0.26, prefersReducedMotion ? 140 : 180, 340);
    const EDGE_ALPHA_BASE = 0.42;
    const EDGE_ALPHA_DISTANCE_GAIN = 0.4;

    const getIntraNeighbors = () => (prefersReducedMotion ? 3 : 4);
    const getInterNeighbors = () => (prefersReducedMotion ? 1 : 2);

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      mouse.x = clamp(mouse.x, 0, canvas.width);
      mouse.y = clamp(mouse.y, 0, canvas.height);
      mouse.targetX = clamp(mouse.targetX, 0, canvas.width);
      mouse.targetY = clamp(mouse.targetY, 0, canvas.height);
      init();
    };

    const createClusterLayout = (count) => {
      const aspect = canvas.width / Math.max(canvas.height, 1);
      const cols = Math.max(1, Math.round(Math.sqrt(count * aspect)));
      const rows = Math.max(1, Math.ceil(count / cols));
      const cellW = canvas.width / cols;
      const cellH = canvas.height / rows;
      const centers = [];

      for (let r = 0; r < rows; r += 1) {
        for (let c = 0; c < cols; c += 1) {
          if (centers.length >= count) break;
          const baseX = cellW * (c + 0.5);
          const baseY = cellH * (r + 0.5);
          centers.push({
            x: clamp(baseX + randomBetween(-cellW * 0.22, cellW * 0.22), 50, canvas.width - 50),
            y: clamp(baseY + randomBetween(-cellH * 0.22, cellH * 0.22), 50, canvas.height - 50),
          });
        }
      }
      return centers;
    };

    class Cluster {
      constructor(x, y) {
        this.x = x;
        this.y = y;
        this.vx = randomBetween(-0.04, 0.04);
        this.vy = randomBetween(-0.04, 0.04);
        this.radius = randomBetween(120, 260);
      }

      update(dt) {
        this.x += this.vx * dt * 60;
        this.y += this.vy * dt * 60;
        if (this.x < 64 || this.x > canvas.width - 64) this.vx *= -1;
        if (this.y < 64 || this.y > canvas.height - 64) this.vy *= -1;
      }
    }

    class Node {
      constructor(clusterId) {
        this.clusterId = clusterId;
        this.offsetPhase = randomBetween(0, Math.PI * 2);
        this.orbit = randomBetween(44, 140);
        this.orbitSpeed = randomBetween(0.08, prefersReducedMotion ? 0.14 : 0.24);
        this.radius = randomBetween(1.8, 3.0);
        this.phaseX = randomBetween(0, Math.PI * 2);
        this.phaseY = randomBetween(0, Math.PI * 2);
        this.speedX = randomBetween(0.12, 0.28);
        this.speedY = randomBetween(0.11, 0.26);
        this.amplitudeX = randomBetween(3.6, prefersReducedMotion ? 6 : 14);
        this.amplitudeY = randomBetween(3.6, prefersReducedMotion ? 6 : 14);
        this.repelX = 0;
        this.repelY = 0;
        this.repelVX = 0;
        this.repelVY = 0;
        this.x = 0;
        this.y = 0;
      }

      update(tSec, dt) {
        const cluster = clusters[this.clusterId];
        if (!cluster) return;
        const orbitX = Math.cos(tSec * this.orbitSpeed + this.offsetPhase) * this.orbit;
        const orbitY = Math.sin(tSec * this.orbitSpeed + this.offsetPhase) * this.orbit * 0.66;
        const driftX = Math.sin(tSec * this.speedX * 2 + this.phaseX) * this.amplitudeX;
        const driftY = Math.cos(tSec * this.speedY * 2 + this.phaseY) * this.amplitudeY;
        const velocityDamping = Math.pow(0.9, dt * 60);
        const displacementDamping = Math.pow(0.965, dt * 60);
        this.repelVX *= velocityDamping;
        this.repelVY *= velocityDamping;
        this.repelX = (this.repelX + this.repelVX) * displacementDamping;
        this.repelY = (this.repelY + this.repelVY) * displacementDamping;
        this.x = cluster.x + orbitX + driftX + this.repelX;
        this.y = cluster.y + orbitY + driftY + this.repelY;
      }

      applyRepulsion(sourceX, sourceY, strength) {
        const dx = this.x - sourceX;
        const dy = this.y - sourceY;
        const dist = Math.hypot(dx, dy);
        if (dist < 1e-5) return;
        this.repelVX += (dx / dist) * strength;
        this.repelVY += (dy / dist) * strength;
      }

      draw() {
        ctx.beginPath();
        ctx.fillStyle = rgba(palette.nodeHalo, 0.24);
        ctx.arc(this.x, this.y, this.radius * 3.4, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.fillStyle = rgba(palette.nodeCore, 0.84);
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    const edgeKey = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);

    const ensureEdge = (a, b, intra) => {
      const key = edgeKey(a, b);
        const existing = edges.get(key);
        if (existing) {
          existing.stale = false;
          existing.intra = existing.intra || intra;
          existing.weightTarget = randomBetween(1.35, 2.25);
          return;
        }
      edges.set(key, {
        key,
        a: Math.min(a, b),
        b: Math.max(a, b),
        intra,
        stale: false,
        life: 0,
        lifeTarget: randomBetween(0.72, 1),
        weight: randomBetween(1.05, 1.92),
        weightTarget: randomBetween(1.35, 2.25),
      });
    };

    const rebuildEdges = () => {
      for (const edge of edges.values()) {
        edge.stale = true;
      }

      const intraDistance = getIntraDistance();
      const interDistance = getInterDistance();
      const intraK = getIntraNeighbors();
      const interK = getInterNeighbors();

      for (let i = 0; i < nodes.length; i += 1) {
        const origin = nodes[i];
        const intra = [];
        const inter = [];
        for (let j = 0; j < nodes.length; j += 1) {
          if (i === j) continue;
          const target = nodes[j];
          const d = distanceNodes(origin, target);
          if (origin.clusterId === target.clusterId) {
            if (d <= intraDistance) intra.push({ j, d });
          } else if (d <= interDistance) {
            inter.push({ j, d });
          }
        }
        intra.sort((a, b) => a.d - b.d);
        inter.sort((a, b) => a.d - b.d);

        for (let k = 0; k < Math.min(intraK, intra.length); k += 1) {
          if (Math.random() < 0.9) ensureEdge(i, intra[k].j, true);
        }
        for (let k = 0; k < Math.min(interK, inter.length); k += 1) {
          if (Math.random() < 0.55) ensureEdge(i, inter[k].j, false);
        }
      }
    };

    const updateEdgeLifecycles = (dt) => {
      for (const edge of [...edges.values()]) {
        if (edge.stale) {
          edge.life -= dt * 1.45;
        } else {
          edge.life += dt * 2.3;
          edge.weight += (edge.weightTarget - edge.weight) * Math.min(1, dt * 7.4);
        }
        edge.life = clamp(edge.life, 0, edge.lifeTarget);
        if (edge.life <= 0.001 && edge.stale) {
          edges.delete(edge.key);
        }
      }
    };

    const maybeSpawnPulse = () => {
      if (edges.size === 0) return;
      const pulseBudget = prefersReducedMotion ? 18 : 42;
      if (pulses.length >= pulseBudget) return;
      const spawnChance = prefersReducedMotion ? 0.08 : 0.22;
      if (Math.random() > spawnChance) return;

      const candidates = [...edges.values()].filter((edge) => edge.life > edge.lifeTarget * 0.45);
      if (candidates.length === 0) return;
      const intraCandidates = candidates.filter((edge) => edge.intra);
      const pool = intraCandidates.length > 0 && Math.random() < 0.8 ? intraCandidates : candidates;
      const burst = prefersReducedMotion ? 1 : 2;
      for (let i = 0; i < burst; i += 1) {
        const edge = pool[Math.floor(Math.random() * pool.length)];
        pulses.push({
          edgeKey: edge.key,
          t: randomBetween(0, 0.18),
          speed: randomBetween(0.28, prefersReducedMotion ? 0.5 : 0.9),
          radius: randomBetween(edge.intra ? 2.0 : 1.5, edge.intra ? 3.2 : 2.4),
        });
      }
    };

    const drawClusterHalos = () => {
      for (const cluster of clusters) {
        const r = cluster.radius;
        const grad = ctx.createRadialGradient(
          cluster.x,
          cluster.y,
          r * 0.2,
          cluster.x,
          cluster.y,
          r
        );
        grad.addColorStop(0, rgba(palette.clusterHalo, isLight ? 0.085 : 0.12));
        grad.addColorStop(1, rgba(palette.clusterHalo, 0));
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cluster.x, cluster.y, r, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    const drawEdges = (timeSec) => {
      const pulseGain = 0.86 + Math.sin(timeSec * 1.35) * 0.14;
      for (const edge of edges.values()) {
        const from = nodes[edge.a];
        const to = nodes[edge.b];
        if (!from || !to) continue;
        const d = distancePoints(from.x, from.y, to.x, to.y);
        const maxD = edge.intra ? getIntraDistance() : getInterDistance();
        const distanceFactor = clamp(1 - d / maxD, 0, 1);
        const alpha = (EDGE_ALPHA_BASE + distanceFactor * EDGE_ALPHA_DISTANCE_GAIN) * pulseGain;
        ctx.beginPath();
        ctx.strokeStyle = rgba(palette.edgeIntra, alpha * edge.life);
        ctx.lineWidth = edge.weight * edge.life;
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
      }
    };

    const updateMouse = (dt) => {
      const motionSmoothing = 1 - Math.exp(-dt * 10);
      mouse.x += (mouse.targetX - mouse.x) * motionSmoothing;
      mouse.y += (mouse.targetY - mouse.y) * motionSmoothing;
      const targetInfluence = mouse.active ? 1 : 0;
      const influenceSmoothing = 1 - Math.exp(-dt * (mouse.active ? 7 : 2.5));
      mouse.influence += (targetInfluence - mouse.influence) * influenceSmoothing;
    };

    const applyMouseRepulsion = () => {
      if (prefersReducedMotion || mouse.influence <= 0.001) return;
      const radius = getMouseInfluenceRadius();
      const baseForce = 0.092 * mouse.influence;
      for (const node of nodes) {
        const d = distancePoints(node.x, node.y, mouse.x, mouse.y);
        if (d >= radius) continue;
        const falloff = 1 - d / radius;
        const strength = baseForce * falloff * (0.45 + falloff * 0.55);
        node.applyRepulsion(mouse.x, mouse.y, strength);
      }
    };

    const onPointerMove = (event) => {
      mouse.targetX = clamp(event.clientX, 0, canvas.width);
      mouse.targetY = clamp(event.clientY, 0, canvas.height);
      mouse.active = true;
    };

    const onPointerLeave = () => {
      mouse.active = false;
    };

    const drawPulses = (dt) => {
      for (let i = pulses.length - 1; i >= 0; i -= 1) {
        const pulse = pulses[i];
        pulse.t += pulse.speed * dt;
        if (pulse.t >= 1) {
          pulses.splice(i, 1);
          continue;
        }
        const edge = edges.get(pulse.edgeKey);
        if (!edge) {
          pulses.splice(i, 1);
          continue;
        }
        const from = nodes[edge.a];
        const to = nodes[edge.b];
        if (!from || !to) continue;
        const x = from.x + (to.x - from.x) * pulse.t;
        const y = from.y + (to.y - from.y) * pulse.t;

        ctx.beginPath();
        ctx.fillStyle = rgba(palette.pulse, 0.28);
        ctx.arc(x, y, pulse.radius * (edge.intra ? 4.5 : 3.9), 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.fillStyle = rgba(palette.pulse, 0.94);
        ctx.arc(x, y, pulse.radius, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    const init = () => {
      const clusterCount = getClusterCount();
      const clusterCenters = createClusterLayout(clusterCount);
      clusters = clusterCenters.map((p) => new Cluster(p.x, p.y));
      const nodeCount = clamp(Math.max(getNodeCount(), clusterCount * 5), clusterCount * 4, 220);
      nodes = Array.from({ length: nodeCount }, (_, i) => new Node(i % clusterCount));
      edges.clear();
      rebuildEdges();
      pulses = [];
      lastTs = 0;
      rewireAccum = 0;
    };

    const animate = (ts) => {
      const tSec = ts / 1000;
      const dt = lastTs ? (ts - lastTs) / 1000 : 0.016;
      lastTs = ts;
      rewireAccum += dt;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const cluster of clusters) {
        cluster.update(dt);
      }
      updateMouse(dt);
      for (const node of nodes) {
        node.update(tSec, dt);
      }
      applyMouseRepulsion();

      if (rewireAccum >= rewireIntervalSec) {
        rebuildEdges();
        for (const edge of edges.values()) {
          if (Math.random() < (prefersReducedMotion ? 0.09 : 0.18)) edge.stale = true;
        }
        rewireAccum = 0;
      }
      updateEdgeLifecycles(dt);
      drawClusterHalos();
      drawEdges(tSec);
      maybeSpawnPulse();
      drawPulses(dt);
      for (const node of nodes) {
        node.draw();
      }

      animationFrameId = requestAnimationFrame(animate);
    };

    resizeCanvas();
    animationFrameId = requestAnimationFrame(animate);

    window.addEventListener("resize", resizeCanvas);
    window.addEventListener("pointermove", onPointerMove, { passive: true });
    window.addEventListener("pointerleave", onPointerLeave);
    window.addEventListener("blur", onPointerLeave);

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerleave", onPointerLeave);
      window.removeEventListener("blur", onPointerLeave);
      cancelAnimationFrame(animationFrameId);
    };
  }, [theme]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed top-0 left-0 w-full h-full pointer-events-none z-0"
      style={{ opacity: theme === "light" ? 0.7 : 0.66 }}
    />
  );
};

export default BackgroundGraph;
