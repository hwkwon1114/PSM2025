"""Generate self-contained interactive cross-section viewer for calibrated zigzag models.

Precomputes fine regular grids of calibrated simulation and experimental scan surfaces
for Run 6 and Run 9, and embeds them into a zero-dependency, standalone HTML application.
Provides continuous scrubbing across transverse (Y-cuts) and longitudinal (X-cuts),
synchronized 2D plate error heatmaps, local residual HUD metrics, profile pinning,
and automated animation sweeps.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import matplotlib.tri as mtri
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from calibration.analysis.run6_run9_six_panel import (
    BASE as BASE_DIR,
    heights_at_xy,
    load_evaluation_data,
    pin,
    resolve_case_paths,
)
from calibration.ewcal.common import digest, save_json


def extract_case_grids(name, case_data, x_vals, y_vals):
    """Interpolate model and scan heights on regular 2D grid and assemble slice metadata."""
    q = case_data['prediction']
    faces = case_data['faces']
    scan = case_data['scan']
    path = case_data['path']
    stats = case_data['statistics']

    X, Y = np.meshgrid(x_vals, y_vals)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    # Evaluate model height
    h_model, support, _ = heights_at_xy(q, faces, pts)
    h_model = h_model.reshape(len(y_vals), len(x_vals))

    # Evaluate scan height via triangulation
    tri_scan = mtri.Triangulation(scan[:, 0], scan[:, 1])
    interp_scan = mtri.LinearTriInterpolator(tri_scan, scan[:, 2])
    h_scan = interp_scan(X, Y).filled(np.nan)

    # Signed residual: model - scan
    dz = h_model - h_scan

    # Discretize toolpath crossings for each slice
    # For transverse slices (y = y_val): find x where path crosses y
    transverse_crossings = []
    for y_v in y_vals:
        crossings = []
        for i in range(len(path) - 1):
            p0, p1 = path[i], path[i + 1]
            if (p0[1] - y_v) * (p1[1] - y_v) <= 0 and abs(p1[1] - p0[1]) > 1e-4:
                t = (y_v - p0[1]) / (p1[1] - p0[1])
                x_c = p0[0] + t * (p1[0] - p0[0])
                if -100 <= x_c <= 100:
                    crossings.append(round(float(x_c), 2))
        transverse_crossings.append(crossings)

    # For longitudinal slices (x = x_val): find y where path crosses x
    longitudinal_crossings = []
    for x_v in x_vals:
        crossings = []
        for i in range(len(path) - 1):
            p0, p1 = path[i], path[i + 1]
            if (p0[0] - x_v) * (p1[0] - x_v) <= 0 and abs(p1[0] - p0[0]) > 1e-4:
                t = (x_v - p0[0]) / (p1[0] - p0[0])
                y_c = p0[1] + t * (p1[1] - p0[1])
                if -150 <= y_c <= 150:
                    crossings.append(round(float(y_c), 2))
        longitudinal_crossings.append(crossings)

    # Downsampled discrete scan points for fast display (~4000 points)
    step = max(1, len(scan) // 4000)
    raw_sub = scan[::step]

    return {
        'name': name.capitalize(),
        'case_id': name,
        'repeat': stats['repeat'],
        'length_mm': stats['length_mm'],
        'scan_points': stats['accepted_scan_points'],
        'model_vertices': stats['prediction_vertices'],
        'rms_dz_mm': round(stats['rms_dz_mm'], 4),
        'mean_dz_mm': round(stats['mean_signed_dz_mm'], 4),
        'p95_dz_mm': round(stats['p95_absolute_dz_mm'], 4),
        'model_z': [[round(float(v), 3) if np.isfinite(v) else None for v in row] for row in h_model],
        'scan_z': [[round(float(v), 3) if np.isfinite(v) else None for v in row] for row in h_scan],
        'dz': [[round(float(v), 3) if np.isfinite(v) else None for v in row] for row in dz],
        'toolpath': [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in path],
        'transverse_crossings': transverse_crossings,
        'longitudinal_crossings': longitudinal_crossings,
        'raw_scan_sample': [[round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 3)] for p in raw_sub],
    }


def build_html_application(data_bundle, out_path):
    """Build single-file HTML containing interactive Canvas/SVG viewer and embedded data."""
    json_data = json.dumps(data_bundle, separators=(',', ':'))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Interactive Cross-Section Comparison: Calibrated Model vs. Scan STL</title>
<style>
  :root {{
    --bg-primary: #f8fafc;
    --bg-card: #ffffff;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --border: #e2e8f0;
    --accent: #2563eb;
    --model-color: #0072B2;
    --scan-color: #D55E00;
    --residual-red: #d73027;
    --residual-blue: #4575b4;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.4;
    padding: 16px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
    flex-shrink: 0;
  }}
  h1 {{ font-size: 1.15rem; font-weight: 600; color: #1e293b; }}
  .tagline {{ font-size: 0.8rem; color: var(--text-secondary); }}
  
  .badge-bar {{
    display: flex;
    gap: 8px;
    align-items: center;
  }}
  .badge {{
    font-size: 0.75rem;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 500;
  }}
  .badge-model {{ background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }}
  .badge-scan {{ background: #ffedd5; color: #c2410c; border: 1px solid #fed7aa; }}

  .main-layout {{
    display: grid;
    grid-template-columns: 360px 1fr;
    gap: 16px;
    flex: 1;
    min-height: 0;
  }}
  
  .card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    min-height: 0;
  }}
  
  /* Left Column: Plate Map and Controls */
  .control-group {{
    margin-bottom: 10px;
    flex-shrink: 0;
  }}
  .control-row {{
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
    align-items: center;
  }}
  label.ctrl-lbl {{
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    min-width: 65px;
  }}
  .btn-group {{
    display: flex;
    flex: 1;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }}
  .btn-group button {{
    flex: 1;
    padding: 5px 8px;
    font-size: 0.75rem;
    background: #f1f5f9;
    border: none;
    cursor: pointer;
    color: var(--text-secondary);
    font-weight: 500;
    transition: all 0.15s ease;
  }}
  .btn-group button.active {{
    background: var(--accent);
    color: white;
  }}
  .btn-group button:hover:not(.active) {{
    background: #e2e8f0;
  }}

  .map-container {{
    flex: 1;
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 0;
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px;
  }}
  #plateCanvas {{
    max-width: 100%;
    max-height: 100%;
    cursor: crosshair;
    display: block;
  }}
  .map-legend {{
    display: flex;
    justify-content: space-between;
    width: 100%;
    font-size: 0.7rem;
    color: var(--text-secondary);
    margin-top: 4px;
  }}

  /* Slider and Scrub Controls */
  .slider-box {{
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }}
  .slider-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
    margin-bottom: 6px;
  }}
  .slider-val {{
    font-weight: 700;
    color: var(--accent);
    font-family: monospace;
    font-size: 0.9rem;
  }}
  input[type="range"] {{
    width: 100%;
    height: 6px;
    border-radius: 3px;
    background: #cbd5e1;
    outline: none;
    cursor: pointer;
  }}
  .playback-bar {{
    display: flex;
    gap: 6px;
    margin-top: 8px;
    align-items: center;
  }}
  .btn-sm {{
    padding: 4px 10px;
    font-size: 0.75rem;
    border: 1px solid var(--border);
    background: white;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 500;
  }}
  .btn-sm:hover {{ background: #f1f5f9; }}

  /* Right Column: Profile Plots */
  .plots-wrapper {{
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 0;
  }}
  
  .plot-card {{
    background: white;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    min-height: 0;
    position: relative;
  }}
  .plot-card.top {{ flex: 1.3; }}
  .plot-card.bottom {{ flex: 0.9; }}

  .plot-title-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #334155;
  }}
  .legend-row {{
    display: flex;
    gap: 12px;
    font-size: 0.75rem;
    align-items: center;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
  }}
  .line-sample {{
    width: 16px;
    height: 2px;
    display: inline-block;
  }}

  .canvas-holder {{
    flex: 1;
    position: relative;
    min-height: 0;
  }}
  .canvas-holder canvas {{
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
  }}

  /* Metrics HUD footer */
  .hud-footer {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-top: 10px;
    flex-shrink: 0;
  }}
  .hud-stat {{
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    text-align: center;
  }}
  .hud-stat .lbl {{
    font-size: 0.68rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    font-weight: 600;
  }}
  .hud-stat .val {{
    font-size: 0.95rem;
    font-weight: 700;
    font-family: monospace;
    color: #1e293b;
    margin-top: 2px;
  }}

  /* Tooltip overlay */
  #tooltip {{
    position: absolute;
    display: none;
    background: rgba(15, 23, 42, 0.9);
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-family: monospace;
    pointer-events: none;
    z-index: 100;
  }}
</style>
</head>
<body>

<header>
  <div>
    <h1>Cross-Section Comparison: Calibrated Model vs. Scan STL</h1>
    <div class="tagline">Interactive spatial exploration of height deflection &amp; signed residuals for zigzag forming</div>
  </div>
  <div class="badge-bar">
    <div class="badge badge-model">&mdash; Calibrated Model (Attempt 209)</div>
    <div class="badge badge-scan">&bull; Experimental Scan (STL Midsurface)</div>
  </div>
</header>

<div class="main-layout">
  <!-- Left Side: Plate Overview & Controls -->
  <div class="card">
    <div class="control-group">
      <div class="control-row">
        <label class="ctrl-lbl">Case:</label>
        <div class="btn-group" id="caseSelector">
          <button class="active" data-case="run6">Run 6 (29-cycle)</button>
          <button data-case="run9">Run 9 (33-cycle)</button>
        </div>
      </div>
      <div class="control-row">
        <label class="ctrl-lbl">Cut Axis:</label>
        <div class="btn-group" id="cutSelector">
          <button class="active" data-cut="transverse">Transverse (Y-cut)</button>
          <button data-cut="longitudinal">Longitudinal (X-cut)</button>
        </div>
      </div>
      <div class="control-row">
        <label class="ctrl-lbl">Plate Map:</label>
        <div class="btn-group" id="mapSelector">
          <button class="active" data-map="dz">&Delta;z Residual</button>
          <button data-map="scan">Scan Height</button>
          <button data-map="model">Model Height</button>
        </div>
      </div>
    </div>

    <!-- 2D Canvas Map -->
    <div class="map-container">
      <canvas id="plateCanvas" width="340" height="490"></canvas>
      <div class="map-legend">
        <span id="mapLegLow">-1.5 mm</span>
        <span id="mapLegMid">0.0 mm</span>
        <span id="mapLegHigh">+1.5 mm</span>
      </div>
    </div>

    <!-- Slider & Animation Controls -->
    <div class="slider-box">
      <div class="slider-header">
        <span id="sliderLabel">Slice Station (Y):</span>
        <span class="slider-val" id="slicePosVal">Y = 0.0 mm</span>
      </div>
      <input type="range" id="sliceSlider" min="-145" max="145" step="2" value="0">
      <div class="playback-bar">
        <button class="btn-sm" id="btnPrev">&larr; -2mm</button>
        <button class="btn-sm" id="btnPlay">Play Sweep</button>
        <button class="btn-sm" id="btnNext">+2mm &rarr;</button>
        <button class="btn-sm" id="btnPin" style="margin-left: auto;">Pin Profile</button>
      </div>
    </div>
  </div>

  <!-- Right Side: 1D Cross-Section Curves & Residuals -->
  <div class="plots-wrapper">
    <!-- Top Plot: Physical Height Profile -->
    <div class="plot-card top">
      <div class="plot-title-row">
        <span id="profileTitle">Physical Height Profile z(x) at Y = 0.0 mm</span>
        <div class="legend-row">
          <div class="legend-item"><span class="line-sample" style="background: var(--model-color);"></span> Model</div>
          <div class="legend-item"><span class="line-sample" style="background: var(--scan-color);"></span> Scan Curve</div>
          <div class="legend-item"><span class="dot" style="background: var(--scan-color);"></span> Scan Data</div>
          <div class="legend-item" id="legPinned" style="display:none;"><span class="line-sample" style="background: #94a3b8; border-top: 1px dashed #64748b;"></span> Pinned</div>
          <div class="legend-item"><span class="line-sample" style="background: #111; height: 1px; border-top: 1px dotted #111;"></span> Toolpath</div>
        </div>
      </div>
      <div class="canvas-holder">
        <canvas id="profileCanvas"></canvas>
      </div>
    </div>

    <!-- Bottom Plot: Signed Residual Curve -->
    <div class="plot-card bottom">
      <div class="plot-title-row">
        <span id="residualTitle">Signed Height Residual &Delta;z = z_model &minus; z_scan (mm)</span>
        <div class="legend-row">
          <div class="legend-item"><span class="line-sample" style="background: #10b981;"></span> Within &plusmn;0.25mm</div>
          <div class="legend-item"><span class="line-sample" style="background: #ef4444;"></span> &gt; &plusmn;0.25mm</div>
        </div>
      </div>
      <div class="canvas-holder">
        <canvas id="residualCanvas"></canvas>
      </div>
    </div>

    <!-- Quantitative HUD Metrics for the active slice -->
    <div class="hud-footer">
      <div class="hud-stat">
        <div class="lbl">Slice RMS Error</div>
        <div class="val" id="hudRms">0.00 mm</div>
      </div>
      <div class="hud-stat">
        <div class="lbl">Slice Max |&Delta;z|</div>
        <div class="val" id="hudMax">0.00 mm</div>
      </div>
      <div class="hud-stat">
        <div class="lbl">Slice Mean Bias</div>
        <div class="val" id="hudBias">0.00 mm</div>
      </div>
      <div class="hud-stat">
        <div class="lbl">Full Case RMS</div>
        <div class="val" id="hudGlobalRms">0.65 mm</div>
      </div>
    </div>
  </div>
</div>

<div id="tooltip"></div>

<script>
// Embedded Evaluation Data
const DATA = {json_data};

// State
let currentCase = 'run6';
let cutMode = 'transverse'; // 'transverse' (Y-cut) or 'longitudinal' (X-cut)
let mapMode = 'dz';          // 'dz', 'scan', 'model'
let currentSliceIndex = 72; // default middle
let pinnedSlice = null;     // slice data frozen for comparison
let isPlaying = false;
let playTimer = null;

const xVals = DATA.x_vals;
const yVals = DATA.y_vals;

// DOM Elements
const plateCanvas = document.getElementById('plateCanvas');
const profileCanvas = document.getElementById('profileCanvas');
const residualCanvas = document.getElementById('residualCanvas');
const sliceSlider = document.getElementById('sliceSlider');
const slicePosVal = document.getElementById('slicePosVal');
const sliderLabel = document.getElementById('sliderLabel');
const profileTitle = document.getElementById('profileTitle');
const residualTitle = document.getElementById('residualTitle');
const btnPlay = document.getElementById('btnPlay');
const btnPin = document.getElementById('btnPin');
const legPinned = document.getElementById('legPinned');
const tooltip = document.getElementById('tooltip');

const hudRms = document.getElementById('hudRms');
const hudMax = document.getElementById('hudMax');
const hudBias = document.getElementById('hudBias');
const hudGlobalRms = document.getElementById('hudGlobalRms');

// Diverging RdBu colormap calculation
function getResidualColor(val, maxAbs) {{
  if (val === null || isNaN(val)) return 'rgba(200,200,200,0.5)';
  const norm = Math.max(-1, Math.min(1, val / maxAbs));
  // RdBu_r: negative is blue (model lower), positive is red (model higher)
  let r, g, b;
  if (norm < 0) {{
    const t = norm + 1; // 0 to 1
    r = Math.round(33 + t * (247 - 33));
    g = Math.round(102 + t * (247 - 102));
    b = Math.round(172 + t * (247 - 172));
  }} else {{
    const t = norm; // 0 to 1
    r = Math.round(247 + t * (178 - 247));
    g = Math.round(247 + t * (24 - 247));
    b = Math.round(247 + t * (43 - 247));
  }}
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function getViridisColor(val, minV, maxV) {{
  if (val === null || isNaN(val)) return 'rgba(200,200,200,0.5)';
  const t = Math.max(0, Math.min(1, (val - minV) / (maxV - minV || 1)));
  // Simplified viridis approximation
  const r = Math.round(68 + t * (253 - 68));
  const g = Math.round(1 + t * (231 - 1));
  const b = Math.round(84 + (1 - Math.abs(t - 0.5) * 2) * 60 + (1 - t) * 40);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function initSlider() {{
  if (cutMode === 'transverse') {{
    sliderLabel.textContent = 'Slice Station (Y):';
    sliceSlider.min = -145;
    sliceSlider.max = 145;
    sliceSlider.step = 2;
    currentSliceIndex = Math.min(currentSliceIndex, yVals.length - 1);
    sliceSlider.value = yVals[currentSliceIndex];
    slicePosVal.textContent = `Y = ${{yVals[currentSliceIndex].toFixed(1)}} mm`;
  }} else {{
    sliderLabel.textContent = 'Slice Station (X):';
    sliceSlider.min = -95;
    sliceSlider.max = 95;
    sliceSlider.step = 2;
    currentSliceIndex = Math.min(currentSliceIndex, xVals.length - 1);
    sliceSlider.value = xVals[currentSliceIndex];
    slicePosVal.textContent = `X = ${{xVals[currentSliceIndex].toFixed(1)}} mm`;
  }}
}}

function renderPlateMap() {{
  const ctx = plateCanvas.getContext('2d');
  const w = plateCanvas.width;
  const h = plateCanvas.height;
  ctx.clearRect(0, 0, w, h);

  const cData = DATA.cases[currentCase];
  const padding = 16;
  const mapW = w - padding * 2;
  const mapH = h - padding * 2;

  // Coordinate transforms: x in [-100, 100], y in [-150, 150]
  function toCanvas(x, y) {{
    const cx = padding + ((x + 100) / 200) * mapW;
    const cy = padding + ((150 - y) / 300) * mapH;
    return [cx, cy];
  }}

  // 1. Draw Grid Cells
  const cellW = mapW / xVals.length;
  const cellH = mapH / yVals.length;
  const maxAbsDz = 1.5;
  const minZ = -4.5, maxZ = 2.5;

  for (let j = 0; j < yVals.length; j++) {{
    for (let i = 0; i < xVals.length; i++) {{
      const x = xVals[i];
      const y = yVals[j];
      const [cx, cy] = toCanvas(x - 1, y + 1);

      let col;
      if (mapMode === 'dz') {{
        col = getResidualColor(cData.dz[j][i], maxAbsDz);
      }} else if (mapMode === 'scan') {{
        col = getViridisColor(cData.scan_z[j][i], minZ, maxZ);
      }} else {{
        col = getViridisColor(cData.model_z[j][i], minZ, maxZ);
      }}
      ctx.fillStyle = col;
      ctx.fillRect(cx, cy, cellW + 0.6, cellH + 0.6);
    }}
  }}

  // 2. Draw Toolpath
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1.0;
  ctx.beginPath();
  cData.toolpath.forEach((pt, idx) => {{
    const [cx, cy] = toCanvas(pt[0], pt[1]);
    if (idx === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  }});
  ctx.stroke();

  // Start Marker
  const [sx, sy] = toCanvas(cData.toolpath[0][0], cData.toolpath[0][1]);
  ctx.fillStyle = '#ea580c';
  ctx.beginPath();
  ctx.arc(sx, sy, 4, 0, Math.PI * 2);
  ctx.fill();

  // 3. Draw Active Slicing Line
  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 2.0;
  ctx.setLineDash([4, 2]);
  ctx.beginPath();
  if (cutMode === 'transverse') {{
    const curY = yVals[currentSliceIndex];
    const [, cy] = toCanvas(0, curY);
    ctx.moveTo(padding, cy);
    ctx.lineTo(w - padding, cy);
  }} else {{
    const curX = xVals[currentSliceIndex];
    const [cx,] = toCanvas(curX, 0);
    ctx.moveTo(cx, padding);
    ctx.lineTo(cx, h - padding);
  }}
  ctx.stroke();
  ctx.setLineDash([]);

  // Pinned line if active
  if (pinnedSlice && pinnedSlice.case === currentCase && pinnedSlice.mode === cutMode) {{
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    if (cutMode === 'transverse') {{
      const [, pcy] = toCanvas(0, pinnedSlice.pos);
      ctx.moveTo(padding, pcy);
      ctx.lineTo(w - padding, pcy);
    }} else {{
      const [pcx,] = toCanvas(pinnedSlice.pos, 0);
      ctx.moveTo(pcx, padding);
      ctx.lineTo(pcx, h - padding);
    }}
    ctx.stroke();
    ctx.setLineDash([]);
  }}
}}

function renderPlots() {{
  const cData = DATA.cases[currentCase];
  hudGlobalRms.textContent = `${{cData.rms_dz_mm.toFixed(2)}} mm`;

  let coordArr, modelArr, scanArr, dzArr, crossings, curPos, axisLabel;

  if (cutMode === 'transverse') {{
    curPos = yVals[currentSliceIndex];
    axisLabel = 'x (mm)';
    coordArr = xVals;
    modelArr = cData.model_z[currentSliceIndex];
    scanArr = cData.scan_z[currentSliceIndex];
    dzArr = cData.dz[currentSliceIndex];
    crossings = cData.transverse_crossings[currentSliceIndex];
    profileTitle.textContent = `Physical Height Profile z(x) at Y = ${{curPos.toFixed(1)}} mm (${{cData.name}})`;
    residualTitle.textContent = `Signed Height Residual &Delta;z(x) at Y = ${{curPos.toFixed(1)}} mm`;
  }} else {{
    curPos = xVals[currentSliceIndex];
    axisLabel = 'y (mm)';
    coordArr = yVals;
    modelArr = cData.model_z.map(r => r[currentSliceIndex]);
    scanArr = cData.scan_z.map(r => r[currentSliceIndex]);
    dzArr = cData.dz.map(r => r[currentSliceIndex]);
    crossings = cData.longitudinal_crossings[currentSliceIndex];
    profileTitle.textContent = `Physical Height Profile z(y) at X = ${{curPos.toFixed(1)}} mm (${{cData.name}})`;
    residualTitle.textContent = `Signed Height Residual &Delta;z(y) at X = ${{curPos.toFixed(1)}} mm`;
  }}

  // Calculate slice statistics
  const validDz = dzArr.filter(v => v !== null && !isNaN(v));
  if (validDz.length > 0) {{
    const mean = validDz.reduce((a, b) => a + b, 0) / validDz.length;
    const rms = Math.sqrt(validDz.reduce((a, b) => a + b * b, 0) / validDz.length);
    const maxErr = Math.max(...validDz.map(Math.abs));
    hudRms.textContent = `${{rms.toFixed(3)}} mm`;
    hudMax.textContent = `${{maxErr.toFixed(3)}} mm`;
    hudBias.textContent = `${{mean >= 0 ? '+' : ''}}${{mean.toFixed(3)}} mm`;
  }}

  // 1. Render Top Profile Plot
  renderProfileCanvas(coordArr, modelArr, scanArr, crossings, axisLabel);

  // 2. Render Bottom Residual Plot
  renderResidualCanvas(coordArr, dzArr, axisLabel);
}}

function renderProfileCanvas(coords, model, scan, crossings, axisLabel) {{
  const ctx = profileCanvas.getContext('2d');
  const rect = profileCanvas.parentElement.getBoundingClientRect();
  profileCanvas.width = rect.width * window.devicePixelRatio || rect.width;
  profileCanvas.height = rect.height * window.devicePixelRatio || rect.height;
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 48, padR = 20, padT = 16, padB = 26;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  // Extents
  const minCoord = coords[0], maxCoord = coords[coords.length - 1];
  const minZ = -5.0, maxZ = 3.5; // Fixed height scale for physical comparison

  function toPlot(u, z) {{
    const px = padL + ((u - minCoord) / (maxCoord - minCoord)) * plotW;
    const py = padT + ((maxZ - z) / (maxZ - minZ)) * plotH;
    return [px, py];
  }}

  // Axes and grid
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let z = -4; z <= 3; z += 2) {{
    const [, py] = toPlot(0, z);
    ctx.moveTo(padL, py);
    ctx.lineTo(w - padR, py);
    ctx.fillStyle = '#64748b';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`${{z}}`, padL - 6, py + 3);
  }}
  ctx.stroke();

  // Zero height line
  const [, pz0] = toPlot(0, 0);
  ctx.strokeStyle = '#cbd5e1';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(padL, pz0); ctx.lineTo(w - padR, pz0);
  ctx.stroke();

  // Toolpath Crossing Vertical Lines
  if (crossings) {{
    ctx.strokeStyle = 'rgba(30, 41, 59, 0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    crossings.forEach(c => {{
      const [cx,] = toPlot(c, 0);
      ctx.beginPath();
      ctx.moveTo(cx, padT); ctx.lineTo(cx, h - padB);
      ctx.stroke();
    }});
    ctx.setLineDash([]);
  }}

  // Pinned Curve if exists
  if (pinnedSlice && pinnedSlice.mode === cutMode) {{
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    let started = false;
    pinnedSlice.model.forEach((z, i) => {{
      if (z !== null && !isNaN(z)) {{
        const [px, py] = toPlot(coords[i], z);
        if (!started) {{ ctx.moveTo(px, py); started = true; }}
        else ctx.lineTo(px, py);
      }}
    }});
    ctx.stroke();
    ctx.setLineDash([]);
  }}

  // Scan Profile Curve
  ctx.strokeStyle = 'var(--scan-color)';
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  let startedScan = false;
  scan.forEach((z, i) => {{
    if (z !== null && !isNaN(z)) {{
      const [px, py] = toPlot(coords[i], z);
      if (!startedScan) {{ ctx.moveTo(px, py); startedScan = true; }}
      else ctx.lineTo(px, py);
    }}
  }});
  ctx.stroke();

  // Scan Data Points (scatter dots)
  ctx.fillStyle = 'var(--scan-color)';
  scan.forEach((z, i) => {{
    if (z !== null && !isNaN(z) && i % 2 === 0) {{
      const [px, py] = toPlot(coords[i], z);
      ctx.beginPath();
      ctx.arc(px, py, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }}
  }});

  // Model Profile Curve
  ctx.strokeStyle = 'var(--model-color)';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  let startedModel = false;
  model.forEach((z, i) => {{
    if (z !== null && !isNaN(z)) {{
      const [px, py] = toPlot(coords[i], z);
      if (!startedModel) {{ ctx.moveTo(px, py); startedModel = true; }}
      else ctx.lineTo(px, py);
    }}
  }});
  ctx.stroke();

  // Axis Labels
  ctx.fillStyle = '#475569';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(axisLabel, padL + plotW / 2, h - 6);
  ctx.save();
  ctx.translate(14, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('z (mm)', 0, 0);
  ctx.restore();
}}

function renderResidualCanvas(coords, dz, axisLabel) {{
  const ctx = residualCanvas.getContext('2d');
  const rect = residualCanvas.parentElement.getBoundingClientRect();
  residualCanvas.width = rect.width * window.devicePixelRatio || rect.width;
  residualCanvas.height = rect.height * window.devicePixelRatio || rect.height;
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 48, padR = 20, padT = 12, padB = 26;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const minCoord = coords[0], maxCoord = coords[coords.length - 1];
  const maxAbsDz = 1.2;

  function toPlot(u, d) {{
    const px = padL + ((u - minCoord) / (maxCoord - minCoord)) * plotW;
    const py = padT + ((maxAbsDz - d) / (maxAbsDz * 2)) * plotH;
    return [px, py];
  }}

  // Tolerance Band &plusmn;0.25mm
  const [, pyTolPos] = toPlot(0, 0.25);
  const [, pyTolNeg] = toPlot(0, -0.25);
  ctx.fillStyle = 'rgba(16, 185, 129, 0.12)';
  ctx.fillRect(padL, pyTolPos, plotW, pyTolNeg - pyTolPos);

  // Grid lines
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  ctx.beginPath();
  [-1.0, -0.5, 0.5, 1.0].forEach(d => {{
    const [, py] = toPlot(0, d);
    ctx.moveTo(padL, py); ctx.lineTo(w - padR, py);
    ctx.fillStyle = '#64748b';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`${{d > 0 ? '+' : ''}}${{d.toFixed(1)}}`, padL - 6, py + 3);
  }});
  ctx.stroke();

  // Zero Baseline
  const [, pyZero] = toPlot(0, 0);
  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(padL, pyZero); ctx.lineTo(w - padR, pyZero);
  ctx.stroke();

  // Tolerance Dash Lines
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(padL, pyTolPos); ctx.lineTo(w - padR, pyTolPos);
  ctx.moveTo(padL, pyTolNeg); ctx.lineTo(w - padR, pyTolNeg);
  ctx.stroke();
  ctx.setLineDash([]);

  // Residual Trace (color segments by tolerance)
  ctx.lineWidth = 2.0;
  for (let i = 0; i < coords.length - 1; i++) {{
    const d0 = dz[i], d1 = dz[i + 1];
    if (d0 !== null && d1 !== null && !isNaN(d0) && !isNaN(d1)) {{
      const [x0, y0] = toPlot(coords[i], d0);
      const [x1, y1] = toPlot(coords[i + 1], d1);
      ctx.strokeStyle = Math.abs((d0 + d1) / 2) <= 0.25 ? '#10b981' : '#ef4444';
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }}
  }}

  // Residual Dots
  dz.forEach((d, i) => {{
    if (d !== null && !isNaN(d) && i % 2 === 0) {{
      const [px, py] = toPlot(coords[i], d);
      ctx.fillStyle = Math.abs(d) <= 0.25 ? '#10b981' : '#ef4444';
      ctx.beginPath();
      ctx.arc(px, py, 2, 0, Math.PI * 2);
      ctx.fill();
    }}
  }});

  // Labels
  ctx.fillStyle = '#475569';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(axisLabel, padL + plotW / 2, h - 6);
  ctx.save();
  ctx.translate(14, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('&Delta;z (mm)', 0, 0);
  ctx.restore();
}}

// Interactive Handlers
function updateAll() {{
  renderPlateMap();
  renderPlots();
}}

// Slider Input
sliceSlider.addEventListener('input', (e) => {{
  const val = parseFloat(e.target.value);
  if (cutMode === 'transverse') {{
    currentSliceIndex = yVals.reduce((best, curr, idx) => Math.abs(curr - val) < Math.abs(yVals[best] - val) ? idx : best, 0);
    slicePosVal.textContent = `Y = ${{yVals[currentSliceIndex].toFixed(1)}} mm`;
  }} else {{
    currentSliceIndex = xVals.reduce((best, curr, idx) => Math.abs(curr - val) < Math.abs(xVals[best] - val) ? idx : best, 0);
    slicePosVal.textContent = `X = ${{xVals[currentSliceIndex].toFixed(1)}} mm`;
  }}
  updateAll();
}});

// Canvas Map Click & Drag
let isDraggingMap = false;
function setSliceFromPointer(e) {{
  const rect = plateCanvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (plateCanvas.width / rect.width);
  const my = (e.clientY - rect.top) * (plateCanvas.height / rect.height);
  const pad = 16;
  const mapW = plateCanvas.width - pad * 2;
  const mapH = plateCanvas.height - pad * 2;

  const physX = -100 + ((mx - pad) / mapW) * 200;
  const physY = 150 - ((my - pad) / mapH) * 300;

  if (cutMode === 'transverse') {{
    const targetY = Math.max(-145, Math.min(145, physY));
    currentSliceIndex = yVals.reduce((best, curr, idx) => Math.abs(curr - targetY) < Math.abs(yVals[best] - targetY) ? idx : best, 0);
    sliceSlider.value = yVals[currentSliceIndex];
    slicePosVal.textContent = `Y = ${{yVals[currentSliceIndex].toFixed(1)}} mm`;
  }} else {{
    const targetX = Math.max(-95, Math.min(95, physX));
    currentSliceIndex = xVals.reduce((best, curr, idx) => Math.abs(curr - targetX) < Math.abs(xVals[best] - targetX) ? idx : best, 0);
    sliceSlider.value = xVals[currentSliceIndex];
    slicePosVal.textContent = `X = ${{xVals[currentSliceIndex].toFixed(1)}} mm`;
  }}
  updateAll();
}}

plateCanvas.addEventListener('mousedown', (e) => {{ isDraggingMap = true; setSliceFromPointer(e); }});
window.addEventListener('mousemove', (e) => {{ if (isDraggingMap) setSliceFromPointer(e); }});
window.addEventListener('mouseup', () => {{ isDraggingMap = false; }});

// Case Selector Buttons
document.getElementById('caseSelector').addEventListener('click', (e) => {{
  if (e.target.tagName === 'BUTTON') {{
    document.querySelectorAll('#caseSelector button').forEach(b => b.classList.remove('active'));
    e.target.classList.add('active');
    currentCase = e.target.getAttribute('data-case');
    updateAll();
  }}
}});

// Cut Selector Buttons
document.getElementById('cutSelector').addEventListener('click', (e) => {{
  if (e.target.tagName === 'BUTTON') {{
    document.querySelectorAll('#cutSelector button').forEach(b => b.classList.remove('active'));
    e.target.classList.add('active');
    cutMode = e.target.getAttribute('data-cut');
    initSlider();
    updateAll();
  }}
}});

// Map Selector Buttons
document.getElementById('mapSelector').addEventListener('click', (e) => {{
  if (e.target.tagName === 'BUTTON') {{
    document.querySelectorAll('#mapSelector button').forEach(b => b.classList.remove('active'));
    e.target.classList.add('active');
    mapMode = e.target.getAttribute('data-map');
    const lLow = document.getElementById('mapLegLow');
    const lMid = document.getElementById('mapLegMid');
    const lHigh = document.getElementById('mapLegHigh');
    if (mapMode === 'dz') {{
      lLow.textContent = '-1.5 mm'; lMid.textContent = '0.0 mm'; lHigh.textContent = '+1.5 mm';
    }} else {{
      lLow.textContent = '-4.5 mm'; lMid.textContent = '-1.0 mm'; lHigh.textContent = '+2.5 mm';
    }}
    renderPlateMap();
  }}
}});

// Step Buttons
document.getElementById('btnPrev').addEventListener('click', () => {{
  currentSliceIndex = Math.max(0, currentSliceIndex - 1);
  const val = cutMode === 'transverse' ? yVals[currentSliceIndex] : xVals[currentSliceIndex];
  sliceSlider.value = val;
  slicePosVal.textContent = `${{cutMode === 'transverse' ? 'Y' : 'X'}} = ${{val.toFixed(1)}} mm`;
  updateAll();
}});

document.getElementById('btnNext').addEventListener('click', () => {{
  const maxIdx = cutMode === 'transverse' ? yVals.length - 1 : xVals.length - 1;
  currentSliceIndex = Math.min(maxIdx, currentSliceIndex + 1);
  const val = cutMode === 'transverse' ? yVals[currentSliceIndex] : xVals[currentSliceIndex];
  sliceSlider.value = val;
  slicePosVal.textContent = `${{cutMode === 'transverse' ? 'Y' : 'X'}} = ${{val.toFixed(1)}} mm`;
  updateAll();
}});

// Play Sweep Animation
btnPlay.addEventListener('click', () => {{
  if (isPlaying) {{
    clearInterval(playTimer);
    isPlaying = false;
    btnPlay.textContent = 'Play Sweep';
  }} else {{
    isPlaying = true;
    btnPlay.textContent = 'Pause';
    playTimer = setInterval(() => {{
      const maxIdx = cutMode === 'transverse' ? yVals.length - 1 : xVals.length - 1;
      currentSliceIndex = (currentSliceIndex + 1) % (maxIdx + 1);
      const val = cutMode === 'transverse' ? yVals[currentSliceIndex] : xVals[currentSliceIndex];
      sliceSlider.value = val;
      slicePosVal.textContent = `${{cutMode === 'transverse' ? 'Y' : 'X'}} = ${{val.toFixed(1)}} mm`;
      updateAll();
    }}, 40); // 25 fps sweep
  }}
}});

// Pin Profile for side-by-side comparison
btnPin.addEventListener('click', () => {{
  const cData = DATA.cases[currentCase];
  if (pinnedSlice) {{
    pinnedSlice = null;
    btnPin.textContent = 'Pin Profile';
    legPinned.style.display = 'none';
  }} else {{
    const curPos = cutMode === 'transverse' ? yVals[currentSliceIndex] : xVals[currentSliceIndex];
    const m = cutMode === 'transverse' ? cData.model_z[currentSliceIndex] : cData.model_z.map(r => r[currentSliceIndex]);
    pinnedSlice = {{
      case: currentCase,
      mode: cutMode,
      pos: curPos,
      model: [...m],
    }};
    btnPin.textContent = 'Unpin Profile';
    legPinned.style.display = 'inline-flex';
  }}
  updateAll();
}});

// Window Resize
window.addEventListener('resize', () => {{
  updateAll();
}});

// Initial boot
initSlider();
updateAll();
</script>

</body>
</html>
"""
    out_path.write_text(html_content, encoding='utf-8')
    print(f"Generated standalone interactive viewer: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, default=209, help='Attempt preset (default: 209 optimal)')
    parser.add_argument('--run6-case', type=Path, help='Run 6 case directory')
    parser.add_argument('--run9-case', type=Path, help='Run 9 case directory')
    parser.add_argument('--out', type=Path, help='Output HTML path')
    args = parser.parse_args()

    r6_case, r9_case, desc, attempt_id = resolve_case_paths(args, BASE)

    out_file = args.out if args.out else BASE / 'legacy/analysis/plots/run6_run9_cross_section_viewer.html'
    out_file.parent.mkdir(parents=True, exist_ok=True)

    pins = {}
    print(f"Loading Run 6 from: {r6_case}")
    d6 = load_evaluation_data('run6', r6_case, pins, BASE)
    print(f"Loading Run 9 from: {r9_case}")
    d9 = load_evaluation_data('run9', r9_case, pins, BASE)

    # Regular sampling grid: 2mm resolution across sheet
    x_vals = np.arange(-95, 96, 2.0).tolist()
    y_vals = np.arange(-145, 146, 2.0).tolist()

    print("Interpolating Run 6 2D grid...")
    g6 = extract_case_grids('run6', d6, x_vals, y_vals)
    print("Interpolating Run 9 2D grid...")
    g9 = extract_case_grids('run9', d9, x_vals, y_vals)

    bundle = {
        'x_vals': [round(x, 1) for x in x_vals],
        'y_vals': [round(y, 1) for y in y_vals],
        'description': desc,
        'attempt_id': attempt_id,
        'cases': {
            'run6': g6,
            'run9': g9,
        }
    }

    build_html_application(bundle, out_file)


if __name__ == '__main__':
    main()
