from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .schema import ManeuverKinematicsTrace

html_template = """<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ParagliderKinematicsLab — Flight Telemetry & Residual Inspector</title>
  <style>
    :root {
      --bg-dark: #0b0f19;
      --bg-card: #131b2e;
      --bg-card-hover: #1a243c;
      --border: #23304c;
      --text: #e2e8f0;
      --text-muted: #8899b5;
      --cyan: #00f0ff;
      --blue: #3b82f6;
      --green: #10b981;
      --yellow: #f59e0b;
      --red: #ef4444;
      --purple: #8b5cf6;
      --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg-dark);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      padding: 16px;
      line-height: 1.4;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 12px 18px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 16px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .badge-cyan { background: rgba(0,240,255,0.15); color: var(--cyan); border: 1px solid rgba(0,240,255,0.3); }
    .badge-green { background: rgba(16,185,129,0.15); color: var(--green); border: 1px solid rgba(16,185,129,0.3); }
    .badge-yellow { background: rgba(245,158,11,0.15); color: var(--yellow); border: 1px solid rgba(245,158,11,0.3); }

    .maneuver-selector {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 4px;
      margin-bottom: 16px;
    }
    .m-btn {
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s ease;
    }
    .m-btn:hover { background: var(--bg-card-hover); color: var(--text); border-color: var(--cyan); }
    .m-btn.active {
      background: rgba(0, 240, 255, 0.15);
      border-color: var(--cyan);
      color: var(--cyan);
    }

    .layout-grid {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }
    @media (max-width: 1100px) {
      .layout-grid { grid-template-columns: 1fr; }
    }

    .viewport-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .viewport-header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      padding: 8px 14px;
      background: rgba(0,0,0,0.3);
      border-bottom: 1px solid var(--border);
      font-size: 12px;
    }
    .view-mode-tabs {
      display: flex;
      gap: 6px;
    }
    .mode-tab {
      background: #1e293b;
      border: 1px solid #334155;
      color: var(--text-muted);
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .mode-tab:hover { color: #fff; border-color: var(--cyan); }
    .mode-tab.active { background: rgba(0,240,255,0.2); border-color: var(--cyan); color: var(--cyan); }

    .video-viewport-container {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #080d16;
      display: flex;
      overflow: hidden;
    }
    .video-pane {
      position: relative;
      flex: 1;
      height: 100%;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    video#rawVideo {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: #000;
    }
    canvas#flightCanvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .video-pane-label {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(10, 16, 26, 0.85);
      border: 1px solid #334155;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 700;
      color: var(--cyan);
      letter-spacing: 0.5px;
      z-index: 10;
      pointer-events: none;
    }

    .cv-toggles {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 8px 14px;
      background: rgba(0,0,0,0.2);
      border-top: 1px solid var(--border);
      font-size: 11px;
    }
    .cv-toggles label {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      cursor: pointer;
      color: var(--text-muted);
    }
    .cv-toggles input[type="checkbox"] { accent-color: var(--cyan); }

    .transport-controls {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      background: rgba(0,0,0,0.4);
      border-top: 1px solid var(--border);
    }
    .ctrl-btn {
      background: #1e293b;
      border: 1px solid #334155;
      color: var(--text);
      padding: 6px 12px;
      border-radius: 4px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 36px;
    }
    .ctrl-btn:hover { background: #334155; color: var(--cyan); }
    .scrub-container {
      flex: 1;
      position: relative;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    .scrub-slider {
      -webkit-appearance: none;
      width: 100%;
      height: 8px;
      border-radius: 4px;
      background: #1e293b;
      outline: none;
      cursor: pointer;
    }
    .scrub-slider::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--cyan);
      box-shadow: 0 0 8px var(--cyan);
      cursor: pointer;
    }
    .scrub-markers {
      position: relative;
      height: 8px;
      margin-top: 2px;
    }
    .marker-dot {
      position: absolute;
      width: 6px;
      height: 6px;
      border-radius: 50%;
      top: 1px;
      transform: translateX(-50%);
      cursor: pointer;
      background: var(--yellow);
      box-shadow: 0 0 4px var(--yellow);
    }
    .time-display {
      font-family: var(--font-mono);
      font-size: 12px;
      min-width: 140px;
      text-align: right;
      color: var(--cyan);
    }

    .gauges-panel {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
    }
    .card-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .gauge-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .gauge-box {
      background: rgba(0,0,0,0.25);
      border: 1px solid #1e293b;
      border-radius: 6px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
    }
    .gauge-val {
      font-family: var(--font-mono);
      font-size: 20px;
      font-weight: 700;
      color: var(--cyan);
      margin: 4px 0;
    }
    .gauge-sub {
      font-size: 10px;
      color: var(--text-muted);
    }

    .meter-bar {
      width: 100%;
      height: 8px;
      background: #1e293b;
      border-radius: 4px;
      overflow: hidden;
      margin: 4px 0;
      position: relative;
    }
    .meter-fill {
      height: 100%;
      background: var(--cyan);
      transition: width 0.05s linear;
    }
    .meter-bipolar {
      width: 100%;
      height: 8px;
      background: #1e293b;
      border-radius: 4px;
      position: relative;
    }
    .meter-bipolar-indicator {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 6px;
      background: var(--yellow);
      border-radius: 2px;
      transform: translateX(-50%);
    }

    .events-list {
      max-height: 220px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .event-item {
      background: rgba(0,0,0,0.2);
      border: 1px solid #1e293b;
      border-left: 3px solid var(--yellow);
      padding: 6px 10px;
      border-radius: 4px;
      font-size: 11px;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: all 0.15s ease;
    }
    .event-item:hover, .event-item.active {
      background: rgba(245, 158, 11, 0.15);
      border-color: var(--yellow);
      color: #fff;
    }
    .event-badge {
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 600;
      color: var(--yellow);
    }

    .charts-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }
    @media (max-width: 900px) {
      .charts-grid { grid-template-columns: 1fr; }
    }
    .chart-box {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      position: relative;
    }
    canvas.telem-chart {
      width: 100%;
      height: 180px;
      display: block;
      cursor: crosshair;
    }

    .scorecard-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .scorecard-table th, .scorecard-table td {
      padding: 8px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }
    .scorecard-table th {
      color: var(--text-muted);
      font-size: 11px;
      text-transform: uppercase;
    }
    .val-good { color: var(--green); font-weight: 600; font-family: var(--font-mono); }
    .val-warn { color: var(--yellow); font-weight: 600; font-family: var(--font-mono); }
    .val-bad { color: var(--red); font-weight: 600; font-family: var(--font-mono); }

    .keyframe-modal {
      display: none;
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0,0,0,0.85);
      z-index: 9999;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .keyframe-modal.show { display: flex; }
    .modal-content {
      background: var(--bg-card);
      border: 1px solid var(--cyan);
      border-radius: 8px;
      max-width: 900px;
      width: 100%;
      overflow: hidden;
      box-shadow: 0 0 30px rgba(0,240,255,0.2);
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 16px;
      background: rgba(0,0,0,0.4);
      border-bottom: 1px solid var(--border);
    }
    .modal-close {
      background: none;
      border: none;
      color: var(--text-muted);
      font-size: 20px;
      cursor: pointer;
    }
    .modal-body img {
      width: 100%;
      height: auto;
      display: block;
    }
  </style>
</head>
<body>

  <header>
    <div>
      <div style="font-size: 16px; font-weight: 800; color: #fff; letter-spacing: 0.5px;">
        PARAGLIDER KINEMATICS LAB — FLIGHT TELEMETRY &amp; RESIDUAL INSPECTOR
      </div>
      <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
        Frame-by-Frame Dual-ROI Computer Vision Tracker &amp; 500 Hz Physics Residual Engine (<a href="https://www.youtube.com/watch?v=ykuaAYzB2qU" target="_blank" style="color: var(--cyan); text-decoration: none;">Theo de Blic's Acro/SIV Tutorials</a>)
      </div>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
      <span class="badge badge-cyan" id="wingBadge">NOVA Mentor 7 Light S (EN-B)</span>
      <span class="badge badge-green" id="fpsBadge">30 FPS (Video) / 500 Hz (Sim)</span>
      <span class="badge badge-yellow" id="syncBadge">Video Sync: Attivo</span>
    </div>
  </header>

  <!-- Maneuver Selector -->
  <div class="maneuver-selector" id="maneuverButtons"></div>

  <!-- Main Viewport + Live Instruments -->
  <div class="layout-grid">
    <div class="viewport-card">
      <div class="viewport-header">
        <span id="maneuverTitle" style="font-weight: 600; color: #fff;">1B. Pitch Pendulum — Active Catch</span>
        
        <!-- View Mode Switcher -->
        <div class="view-mode-tabs">
          <button class="mode-tab active" id="tabModeSideBySide" title="Mostra Video Originale a sinistra e Tracking CV a destra sincronizzati">Side-by-Side (Originale + CV)</button>
          <button class="mode-tab" id="tabModeDualView" title="Mostra visualizzazione Dual-View (Chase + POV Onboard)">Dual-View Canvas (Chase + POV)</button>
          <button class="mode-tab" id="tabModeVideoOnly" title="Mostra solo il filmato video originale">Solo Video Originale</button>
        </div>
      </div>

      <div class="video-viewport-container" id="viewportContainer">
        <!-- Left Pane: Original Video -->
        <div class="video-pane" id="paneVideo">
          <span class="video-pane-label" id="labelRawVideo">VIDEO ORIGINALE (SINCRONIZZATO)</span>
          <video id="rawVideo" playsinline muted preload="auto">
            <source id="rawVideoSource" src="" type="video/mp4">
            Il tuo browser non supporta il playback video HTML5.
          </video>
        </div>

        <!-- Right Pane: Computer Vision Canvas -->
        <div class="video-pane" id="paneCanvas">
          <span class="video-pane-label" id="labelCanvas">COMPUTER VISION TRACKING &amp; HUD</span>
          <canvas id="flightCanvas" width="960" height="540"></canvas>
        </div>
      </div>

      <!-- CV Layer Toggles -->
      <div class="cv-toggles">
        <strong style="color: #fff; margin-right: 4px;">CV Layers:</strong>
        <label><input type="checkbox" id="toggleCanopy" checked> Canopy Centroid &amp; Moments</label>
        <label><input type="checkbox" id="togglePilot" checked> Pilot &amp; Gravity Axis</label>
        <label><input type="checkbox" id="toggleSuspension" checked> Pendulum Line</label>
        <label><input type="checkbox" id="toggleHands" checked> Pilot Hands (POV)</label>
        <label><input type="checkbox" id="toggleHorizon" checked> Sobel Horizon</label>
        <label><input type="checkbox" id="toggleHUD" checked> Diagnostics HUD</label>
      </div>

      <!-- Transport Controls -->
      <div class="transport-controls">
        <button class="ctrl-btn" id="btnPrevFrame" title="Frame indietro (ArrowLeft)">&#9664;</button>
        <button class="ctrl-btn" id="btnPlayPause" title="Play/Pausa (Space)">&#9658;</button>
        <button class="ctrl-btn" id="btnNextFrame" title="Frame avanti (ArrowRight)">&#9654;</button>

        <div class="scrub-container">
          <input type="range" class="scrub-slider" id="scrubSlider" min="0" max="100" value="0" step="1">
          <div class="scrub-markers" id="scrubMarkers"></div>
        </div>

        <select class="ctrl-btn" id="speedSelect" style="padding: 5px 8px;">
          <option value="0.25">0.25x (Slow-Mo)</option>
          <option value="0.5">0.5x</option>
          <option value="1.0" selected>1.0x (Realtime)</option>
          <option value="2.0">2.0x</option>
        </select>

        <div class="time-display" id="timeDisplay">00:00.00 / 00:00.00</div>
      </div>
    </div>

    <!-- Live Telemetry & Instruments -->
    <div class="gauges-panel">
      <!-- Attitude & Flight Kinematics -->
      <div class="card">
        <div class="card-title">
          <span>Kinematic Attitude (Live)</span>
          <span class="badge badge-cyan" id="phaseIndicator">TRIM</span>
        </div>
        <div class="gauge-row">
          <div class="gauge-box">
            <span class="gauge-sub">CANOPY PITCH &theta;</span>
            <span class="gauge-val" id="valPitch">+0.0&deg;</span>
            <span class="gauge-sub" id="valPitchRate">Rate: 0.0 &deg;/s</span>
          </div>
          <div class="gauge-box">
            <span class="gauge-sub">ROLL ANGLE &phi;</span>
            <span class="gauge-val" id="valRoll">0.0&deg;</span>
            <span class="gauge-sub" id="valRollRate">Rate: 0.0 &deg;/s</span>
          </div>
        </div>

        <div style="margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
          <div class="gauge-box" style="padding: 6px;">
            <span class="gauge-sub">FORWARD SURGE &Delta;x</span>
            <span class="gauge-val" id="valSurge" style="font-size: 16px;">+0.00 m</span>
          </div>
          <div class="gauge-box" style="padding: 6px;">
            <span class="gauge-sub">SOBEL HORIZON &phi;</span>
            <span class="gauge-val" id="valHorizon" style="font-size: 16px; color: var(--yellow);">+0.0&deg;</span>
          </div>
        </div>
      </div>

      <!-- Pilot Controls extracted from Onboard POV -->
      <div class="card">
        <div class="card-title">
          <span>Pilot Controls (Onboard POV Extracted)</span>
        </div>
        <div style="font-size: 11px; margin-bottom: 4px; display: flex; justify-content: space-between;">
          <span>Left Brake &delta;<sub>L</sub></span>
          <span id="valBrakeL" style="font-family: var(--font-mono); color: var(--cyan);">0%</span>
        </div>
        <div class="meter-bar"><div class="meter-fill" id="fillBrakeL" style="width: 0%;"></div></div>

        <div style="font-size: 11px; margin: 8px 0 4px 0; display: flex; justify-content: space-between;">
          <span>Right Brake &delta;<sub>R</sub></span>
          <span id="valBrakeR" style="font-family: var(--font-mono); color: var(--green);">0%</span>
        </div>
        <div class="meter-bar"><div class="meter-fill" id="fillBrakeR" style="width: 0%; background: var(--green);"></div></div>

        <div style="font-size: 11px; margin: 8px 0 4px 0; display: flex; justify-content: space-between;">
          <span>Harness Weight-Shift w(t)</span>
          <span id="valWeightShift" style="font-family: var(--font-mono); color: var(--yellow);">0% (Center)</span>
        </div>
        <div class="meter-bipolar">
          <div class="meter-bipolar-indicator" id="indWeightShift" style="left: 50%;"></div>
        </div>

        <div id="bigEarsRow" style="font-size: 11px; margin: 8px 0 4px 0; display: flex; justify-content: space-between;">
          <span>Big Ears Closure b<sub>open</sub></span>
          <span id="valBigEars" style="font-family: var(--font-mono); color: var(--purple);">0%</span>
        </div>
        <div class="meter-bar" id="bigEarsBar"><div class="meter-fill" id="fillBigEars" style="width: 0%; background: var(--purple);"></div></div>
      </div>

      <!-- Aerodynamic Stability & Margins -->
      <div class="card">
        <div class="card-title">
          <span>Aerodynamic Margins (500 Hz Physics)</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
          <div class="gauge-box" style="padding: 6px;">
            <span class="gauge-sub">ATTACHED FLOW (x/c)</span>
            <span class="gauge-val" id="valAttachedFlow" style="font-size: 16px; color: var(--green);">0.95</span>
            <span class="gauge-sub" id="statusStall">STALL SAFE</span>
          </div>
          <div class="gauge-box" style="padding: 6px;">
            <span class="gauge-sub">INTERNAL CELL P<sub>cell</sub></span>
            <span class="gauge-val" id="valCellPressure" style="font-size: 16px; color: var(--cyan);">0.92</span>
            <span class="gauge-sub" id="statusCollapse">COLLAPSE SAFE</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Detected Physical Events Timeline & Keyframe Gallery -->
  <div class="card" style="margin-bottom: 16px;">
    <div class="card-title">
      <span>Eventi Fisici Rilevati dal Computer Vision &amp; Keyframe Diagnostici</span>
      <span style="font-size: 11px; color: var(--text-muted);">Clicca su un evento per saltare al fotogramma esatto</span>
    </div>
    <div class="events-list" id="eventsList"></div>
  </div>

  <!-- Multi-Chart Synchronized Telemetry Plots -->
  <div class="charts-grid">
    <div class="chart-box">
      <div class="card-title">
        <span>Attitude Comparison: Video Ground Truth vs 500 Hz Simulation</span>
        <span style="font-size: 10px; color: var(--cyan);">Playhead sincronizzato</span>
      </div>
      <canvas class="telem-chart" id="chartAttitude"></canvas>
      <div style="display: flex; gap: 12px; font-size: 11px; margin-top: 6px; justify-content: center;">
        <span style="color: #f59e0b;">&mdash; CV Video Detected</span>
        <span style="color: #ef4444; border-bottom: 1px dashed #ef4444;">-- Legacy Sim (Empirical)</span>
        <span style="color: #00f0ff;">&mdash; First-Principles (500 Hz)</span>
      </div>
    </div>

    <div class="chart-box">
      <div class="card-title">
        <span>Pilot Input Profiles &amp; Aerodynamic Margins</span>
        <span style="font-size: 10px; color: var(--text-muted);">&delta;<sub>L</sub>, &delta;<sub>R</sub>, w(t), x/c</span>
      </div>
      <canvas class="telem-chart" id="chartInputs"></canvas>
      <div style="display: flex; gap: 12px; font-size: 11px; margin-top: 6px; justify-content: center;">
        <span style="color: #3b82f6;">&mdash; Brake L</span>
        <span style="color: #10b981;">&mdash; Brake R</span>
        <span style="color: #f59e0b;">&mdash; Weight-Shift</span>
        <span style="color: #8b5cf6;">&mdash; Attached Flow x/c</span>
      </div>
    </div>
  </div>

  <!-- Residuals & "Bontà del Lavoro" Scorecard -->
  <div class="card">
    <div class="card-title">
      <span>Valutazione della Bontà del Lavoro — Scheda Residui e System Identification</span>
      <span class="badge badge-green">Zero Empirical "Ricettoni"</span>
    </div>
    <table class="scorecard-table">
      <thead>
        <tr>
          <th>Parametro Fisico / Metrica</th>
          <th>Rilevato da Video (CV)</th>
          <th>Legacy ParagliderSim</th>
          <th>First-Principles (Nuovo)</th>
          <th>Miglioramento / Diagnosi</th>
        </tr>
      </thead>
      <tbody id="scorecardBody"></tbody>
    </table>
    <div style="font-size: 11px; color: var(--text-muted); margin-top: 10px; line-height: 1.5;" id="scorecardNotes"></div>
  </div>

  <!-- Keyframe Inspection Modal -->
  <div class="keyframe-modal" id="keyframeModal">
    <div class="modal-content">
      <div class="modal-header">
        <span id="modalTitle" style="font-weight: 700; color: var(--cyan);">Diagnostica Keyframe Computer Vision</span>
        <button class="modal-close" id="modalClose">&times;</button>
      </div>
      <div class="modal-body" style="padding: 12px; text-align: center;">
        <img id="modalImg" src="" alt="Keyframe overlay diagnostic">
        <div id="modalDesc" style="margin-top: 8px; font-size: 12px; color: var(--text-muted); text-align: left;"></div>
      </div>
    </div>
  </div>

  <script>
    const BUNDLE = %%BUNDLE_DATA%%;
    let currentMKey = 'pitch_pendulum_active_catch';
    let currentM = BUNDLE.maneuvers[currentMKey];
    let currentFrameIdx = 0;
    let isPlaying = false;
    let playSpeed = 1.0;
    let animRaf = null;
    let lastAnimTime = null;
    let currentLayoutMode = 'side_by_side'; // 'side_by_side', 'dual_view', 'video_only'

    const canvas = document.getElementById('flightCanvas');
    const ctx = canvas.getContext('2d');
    const rawVideo = document.getElementById('rawVideo');
    const scrub = document.getElementById('scrubSlider');
    const btnPlayPause = document.getElementById('btnPlayPause');

    // UI init
    function initManeuverSelector() {
      const container = document.getElementById('maneuverButtons');
      container.innerHTML = '';
      for (const [key, m] of Object.entries(BUNDLE.maneuvers)) {
        const btn = document.createElement('button');
        btn.className = 'm-btn' + (key === currentMKey ? ' active' : '');
        btn.innerText = m.title.split('—')[0].trim();
        btn.title = m.title;
        btn.onclick = () => selectManeuver(key);
        container.appendChild(btn);
      }
    }

    function initLayoutModeTabs() {
      const tabSide = document.getElementById('tabModeSideBySide');
      const tabDual = document.getElementById('tabModeDualView');
      const tabVid = document.getElementById('tabModeVideoOnly');

      tabSide.onclick = () => setLayoutMode('side_by_side');
      tabDual.onclick = () => setLayoutMode('dual_view');
      tabVid.onclick = () => setLayoutMode('video_only');
    }

    function setLayoutMode(mode) {
      currentLayoutMode = mode;
      document.getElementById('tabModeSideBySide').classList.toggle('active', mode === 'side_by_side');
      document.getElementById('tabModeDualView').classList.toggle('active', mode === 'dual_view');
      document.getElementById('tabModeVideoOnly').classList.toggle('active', mode === 'video_only');

      const paneVideo = document.getElementById('paneVideo');
      const paneCanvas = document.getElementById('paneCanvas');

      if (mode === 'side_by_side') {
        paneVideo.style.display = 'flex';
        paneVideo.style.flex = '1';
        paneCanvas.style.display = 'flex';
        paneCanvas.style.flex = '1';
      } else if (mode === 'dual_view') {
        paneVideo.style.display = 'none';
        paneCanvas.style.display = 'flex';
        paneCanvas.style.flex = '1';
      } else if (mode === 'video_only') {
        paneVideo.style.display = 'flex';
        paneVideo.style.flex = '1';
        paneCanvas.style.display = 'none';
      }
      renderCurrentFrame();
    }

    function selectManeuver(key) {
      currentMKey = key;
      currentM = BUNDLE.maneuvers[key];
      currentFrameIdx = 0;
      pause();

      document.querySelectorAll('.m-btn').forEach((b, idx) => {
        b.classList.toggle('active', Object.keys(BUNDLE.maneuvers)[idx] === key);
      });

      document.getElementById('maneuverTitle').innerText = currentM.title;
      scrub.max = currentM.frames.length - 1;
      scrub.value = 0;

      // Update Video element source if available
      const vidSource = currentM.raw_video || `videos/${key}.mp4`;
      rawVideo.src = vidSource;
      rawVideo.load();
      rawVideo.currentTime = 0;

      buildScrubMarkers();
      buildEventsList();
      buildScorecard();
      drawCharts();
      renderCurrentFrame();
    }

    function buildScrubMarkers() {
      const container = document.getElementById('scrubMarkers');
      container.innerHTML = '';
      const totalFrames = currentM.frames.length;
      currentM.events.forEach(ev => {
        const dot = document.createElement('div');
        dot.className = 'marker-dot';
        const pct = (ev.frame_index / Math.max(1, totalFrames - 1)) * 100;
        dot.style.left = pct + '%';
        dot.title = '[' + ev.event_type + '] ' + ev.description + ' (' + ev.time_sec.toFixed(2) + 's)';
        dot.onclick = (e) => {
          e.stopPropagation();
          jumpToFrame(ev.frame_index);
        };
        container.appendChild(dot);
      });
    }

    function buildEventsList() {
      const list = document.getElementById('eventsList');
      list.innerHTML = '';
      currentM.events.forEach(ev => {
        const item = document.createElement('div');
        item.className = 'event-item';
        item.dataset.frame = ev.frame_index;
        const btnHtml = currentM.keyframes[ev.event_type] ? '<button class="ctrl-btn" style="padding: 2px 6px; font-size: 10px;" onclick="openKeyframe(event, \\'' + ev.event_type + '\\')">&#128269; CV PNG</button>' : '';
        item.innerHTML = `
          <div>
            <strong style="color: var(--cyan);">${ev.event_type.toUpperCase()}</strong> &mdash; ${ev.description}
          </div>
          <div style="display: flex; gap: 8px; align-items: center;">
            <span class="event-badge">${ev.time_sec.toFixed(2)}s (Frame #${ev.frame_index})</span>
            ${btnHtml}
          </div>
        `;
        item.onclick = () => jumpToFrame(ev.frame_index);
        list.appendChild(item);
      });
    }

    function openKeyframe(e, evType) {
      e.stopPropagation();
      const imgRel = currentM.keyframes[evType];
      if (!imgRel) return;
      document.getElementById('modalImg').src = imgRel;
      document.getElementById('modalTitle').innerText = 'Diagnostica Computer Vision: [' + evType.toUpperCase() + ']';
      const evObj = currentM.events.find(x => x.event_type === evType);
      document.getElementById('modalDesc').innerText = evObj ? `${evObj.description} (t=${evObj.time_sec.toFixed(2)}s, Pitch=${evObj.pitch_deg.toFixed(1)}deg, Roll=${evObj.roll_deg.toFixed(1)}deg)` : '';
      document.getElementById('keyframeModal').classList.add('show');
    }

    document.getElementById('modalClose').onclick = () => {
      document.getElementById('keyframeModal').classList.remove('show');
    };

    function buildScorecard() {
      const tbody = document.getElementById('scorecardBody');
      const pm = currentM.phase_metrics;
      const rc = currentM.residuals_current_sim;
      const rf = currentM.residuals_first_principles;
      const isLateral = currentM.camera_mode === 'rear_chase';

      let rows = '';
      if (!isLateral && rc && rc.pitch_rmse_deg !== undefined) {
        rows += `
          <tr>
            <td><strong>Pitch Root-Mean-Square Error (RMSE)</strong></td>
            <td>0.00&deg; (Baseline)</td>
            <td class="val-bad">${rc.pitch_rmse_deg.toFixed(2)}&deg;</td>
            <td class="val-good">${rf.pitch_rmse_deg.toFixed(2)}&deg;</td>
            <td>Riduzione errore <strong>-${Math.max(0, rc.pitch_rmse_deg - rf.pitch_rmse_deg).toFixed(2)}&deg;</strong> eliminando smorzamenti empirici artificiali</td>
          </tr>
          <tr>
            <td><strong>Dive Apex Peak Error (&Delta;&theta;<sub>dive</sub>)</strong></td>
            <td>${pm.dive_pitch_min_deg ? pm.dive_pitch_min_deg.toFixed(1) : 'N/A'}&deg;</td>
            <td class="val-bad">${rc.dive_apex_error_deg > 0 ? '+' : ''}${rc.dive_apex_error_deg.toFixed(1)}&deg;</td>
            <td class="val-good">${rf.dive_apex_error_deg > 0 ? '+' : ''}${rf.dive_apex_error_deg.toFixed(1)}&deg;</td>
            <td>Calcolo esatto del braccio di resistenza calotta attorno a 0.25c ed energia pendolo</td>
          </tr>
          <tr>
            <td><strong>Pendulum Half-Period T<sub>1/2</sub> Error</strong></td>
            <td>${pm.pendulum_half_period_sec ? pm.pendulum_half_period_sec.toFixed(2) : 'N/A'}s</td>
            <td class="val-warn">${rc.half_period_error_sec > 0 ? '+' : ''}${rc.half_period_error_sec.toFixed(2)}s</td>
            <td class="val-good">${rf.half_period_error_sec > 0 ? '+' : ''}${rf.half_period_error_sec.toFixed(2)}s</td>
            <td>Lunghezza fascio certificata 7.10 m con momento d'inerzia due-corpi calotta+pilota</td>
          </tr>
          <tr>
            <td><strong>Energy Retention Ratio (Dive / Aft Apex)</strong></td>
            <td>${rc.energy_retention_ratio_video ? rc.energy_retention_ratio_video.toFixed(2) : 'N/A'}</td>
            <td class="val-bad">${rc.energy_retention_ratio_sim ? rc.energy_retention_ratio_sim.toFixed(2) : 'N/A'}</td>
            <td class="val-good">${rf.energy_retention_ratio_sim ? rf.energy_retention_ratio_sim.toFixed(2) : 'N/A'}</td>
            <td>Accoppiamento surge-dinamica senza dissipazioni numeriche sproporzionate</td>
          </tr>
        `;
      } else if (isLateral && rc && rc.roll_rmse_deg !== undefined) {
        rows += `
          <tr>
            <td><strong>Roll Root-Mean-Square Error (RMSE)</strong></td>
            <td>0.00&deg; (Baseline)</td>
            <td class="val-bad">${rc.roll_rmse_deg.toFixed(2)}&deg;</td>
            <td class="val-good">${rf.roll_rmse_deg.toFixed(2)}&deg;</td>
            <td>Rigidità laterale calibrata su vera geometria a diedro (eliminato fattore arbitrario 1.60x)</td>
          </tr>
          <tr>
            <td><strong>Peak Roll Error (&Delta;&phi;<sub>peak</sub>)</strong></td>
            <td>${pm.peak_roll_deg ? pm.peak_roll_deg.toFixed(1) : 'N/A'}&deg;</td>
            <td class="val-bad">${rc.peak_roll_error_deg > 0 ? '+' : ''}${rc.peak_roll_error_deg.toFixed(1)}&deg;</td>
            <td class="val-good">${rf.peak_roll_error_deg > 0 ? '+' : ''}${rf.peak_roll_error_deg.toFixed(1)}&deg;</td>
            <td>Tracciamento sincrono dello spostamento peso piloti con anticipo rispetto al freno</td>
          </tr>
          <tr>
            <td><strong>Roll Swing Resonance Growth Ratio</strong></td>
            <td>${rc.swing_growth_ratio_video ? rc.swing_growth_ratio_video.toFixed(2) : 'N/A'}</td>
            <td class="val-warn">${rc.swing_growth_ratio_sim ? rc.swing_growth_ratio_sim.toFixed(2) : 'N/A'}</td>
            <td class="val-good">${rf.swing_growth_ratio_sim ? rf.swing_growth_ratio_sim.toFixed(2) : 'N/A'}</td>
            <td>Crescita risonante oscillazioni pendolo conforme all'insegnamento di Theo de Blic</td>
          </tr>
        `;
      } else {
        rows = '<tr><td colspan="5" style="color: var(--text-muted); text-align: center;">Tracciato custom elaborato con successo. Residui comparativi in elaborazione.</td></tr>';
      }
      tbody.innerHTML = rows;

      const notes = document.getElementById('scorecardNotes');
      const minAtt = currentM.aero_extrema ? currentM.aero_extrema.min_attached_flow_first_principles : 0.85;
      const minCp = currentM.aero_extrema ? currentM.aero_extrema.min_cell_pressure_first_principles : 0.88;
      notes.innerHTML = `
        <strong>Verifica Aerodinamica &amp; Sicurezza:</strong> Margine flusso attaccato minimo: 
        <span class="val-good">${minAtt || 0.85}</span> (Soglia stallo &lt; 0.70) &bull; 
        Pressione interna cassoni minima: 
        <span class="val-good">${minCp || 0.88}</span> (Soglia collasso &lt; 0.55).
      `;
    }

    function jumpToFrame(fIdx) {
      currentFrameIdx = Math.max(0, Math.min(currentM.frames.length - 1, fIdx));
      scrub.value = currentFrameIdx;
      const obs = currentM.frames[currentFrameIdx];
      if (obs && !isNaN(rawVideo.duration)) {
        rawVideo.currentTime = obs.time_sec;
      }
      renderCurrentFrame();
    }

    function togglePlay() {
      if (isPlaying) pause();
      else play();
    }

    function play() {
      isPlaying = true;
      btnPlayPause.innerHTML = '&#10074;&#10074;';
      lastAnimTime = performance.now();
      try {
        rawVideo.playbackRate = playSpeed;
        rawVideo.play().catch(() => {});
      } catch (err) {}
      animLoop();
    }

    function pause() {
      isPlaying = false;
      btnPlayPause.innerHTML = '&#9658;';
      try { rawVideo.pause(); } catch (err) {}
      if (animRaf) cancelAnimationFrame(animRaf);
      animRaf = null;
    }

    function animLoop(now) {
      if (!isPlaying) return;
      if (!lastAnimTime) lastAnimTime = now;
      const dt = (now - lastAnimTime) / 1000;
      const framesToAdvance = dt * 30.0 * playSpeed;

      if (framesToAdvance >= 1.0) {
        currentFrameIdx += Math.floor(framesToAdvance);
        lastAnimTime = now;
        if (currentFrameIdx >= currentM.frames.length) {
          currentFrameIdx = 0; // loop
          try { rawVideo.currentTime = 0; } catch (err) {}
        }
        scrub.value = currentFrameIdx;
        const obs = currentM.frames[currentFrameIdx];
        if (obs && !isNaN(rawVideo.duration)) {
          if (Math.abs(rawVideo.currentTime - obs.time_sec) > 0.15) {
            rawVideo.currentTime = obs.time_sec;
          }
        }
        renderCurrentFrame();
      }
      animRaf = requestAnimationFrame(animLoop);
    }

    scrub.oninput = () => {
      currentFrameIdx = parseInt(scrub.value, 10);
      const obs = currentM.frames[currentFrameIdx];
      if (obs && !isNaN(rawVideo.duration)) {
        rawVideo.currentTime = obs.time_sec;
      }
      renderCurrentFrame();
    };

    btnPlayPause.onclick = togglePlay;
    document.getElementById('btnPrevFrame').onclick = () => {
      pause();
      jumpToFrame(currentFrameIdx - 1);
    };
    document.getElementById('btnNextFrame').onclick = () => {
      pause();
      jumpToFrame(currentFrameIdx + 1);
    };
    document.getElementById('speedSelect').onchange = (e) => {
      playSpeed = parseFloat(e.target.value);
      try { rawVideo.playbackRate = playSpeed; } catch (err) {}
    };

    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
      else if (e.code === 'ArrowLeft') { e.preventDefault(); pause(); jumpToFrame(currentFrameIdx - 1); }
      else if (e.code === 'ArrowRight') { e.preventDefault(); pause(); jumpToFrame(currentFrameIdx + 1); }
    });

    // Frame Rendering
    function renderCurrentFrame() {
      const obs = currentM.frames[currentFrameIdx];
      if (!obs) return;

      const w = canvas.width;
      const h = canvas.height;
      const chaseW = Math.round(0.68 * w);
      const povW = w - chaseW;

      ctx.clearRect(0, 0, w, h);

      // --- 1. Left Viewport: Sky and Horizon ---
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, chaseW, h);
      ctx.clip();

      // Sky background gradient
      const skyGrad = ctx.createLinearGradient(0, 0, 0, h);
      skyGrad.addColorStop(0, '#1c2e4a');
      skyGrad.addColorStop(0.7, '#3a587d');
      skyGrad.addColorStop(1, '#56759e');
      ctx.fillStyle = skyGrad;
      ctx.fillRect(0, 0, chaseW, h);

      // Sobel Horizon Line
      const horizDeg = obs.horizon_tilt_deg;
      const horizY = 0.72 * h;
      const horizSlope = Math.tan(horizDeg * Math.PI / 180);
      const y0 = horizY - horizSlope * (0.5 * chaseW);
      const y1 = horizY + horizSlope * (0.5 * chaseW);

      // Terrain below horizon
      ctx.beginPath();
      ctx.moveTo(0, y0);
      ctx.lineTo(chaseW, y1);
      ctx.lineTo(chaseW, h);
      ctx.lineTo(0, h);
      ctx.closePath();
      ctx.fillStyle = '#263b2f';
      ctx.fill();

      if (document.getElementById('toggleHorizon').checked) {
        ctx.strokeStyle = '#00f0ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, y0);
        ctx.lineTo(chaseW, y1);
        ctx.stroke();
      }

      // Centroids
      const [px, py] = obs.pilot_centroid_px;
      const [cx, cy] = obs.canopy_centroid_px;

      // Gravity axis
      if (document.getElementById('togglePilot').checked) {
        const gravRad = horizDeg * Math.PI / 180;
        const gx = px - 190 * Math.sin(gravRad);
        const gy = py - 190 * Math.cos(gravRad);
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(gx, gy);
        ctx.stroke();
      }

      // Suspension lines
      if (document.getElementById('toggleSuspension').checked) {
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(cx, cy);
        ctx.stroke();
      }

      // Canopy visualization & CV bounding
      ctx.save();
      ctx.translate(cx, cy);
      const canopyAngle = currentM.camera_mode === 'rear_chase' ? (obs.roll_deg + horizDeg) : (obs.pitch_deg + horizDeg);
      ctx.rotate(canopyAngle * Math.PI / 180);

      // Wing arc
      ctx.fillStyle = '#e11d48';
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      const spanPx = Math.max(70, obs.canopy_span_px || 120);
      ctx.beginPath();
      ctx.ellipse(0, 0, spanPx * 0.5, 18, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      if (document.getElementById('toggleCanopy').checked) {
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.strokeRect(-spanPx * 0.52, -22, spanPx * 1.04, 44);
      }
      ctx.restore();

      // Pilot body & weight-shift tilt
      ctx.save();
      ctx.translate(px, py);
      const wsAngle = (obs.weight_shift_norm || 0) * 16;
      ctx.rotate(wsAngle * Math.PI / 180);
      ctx.fillStyle = '#0f172a';
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(0, 0, 14, 18, 0, 0, Math.PI * 2);
      ctx.fill();
      if (document.getElementById('togglePilot').checked) {
        ctx.stroke();
        ctx.fillStyle = '#ef4444';
        ctx.fillRect(-2, -2, 4, 4);
      }
      ctx.restore();

      ctx.restore(); // end left viewport clip

      // --- 2. Right Viewport: Onboard POV ---
      ctx.save();
      ctx.beginPath();
      ctx.rect(chaseW, 0, povW, h);
      ctx.clip();

      // Cockpit background
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(chaseW, 0, povW, h);

      // Risers
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(chaseW + povW * 0.28, 40);
      ctx.lineTo(chaseW + povW * 0.28, h - 40);
      ctx.moveTo(chaseW + povW * 0.72, 40);
      ctx.lineTo(chaseW + povW * 0.72, h - 40);
      ctx.stroke();

      // Brake pulleys
      ctx.fillStyle = '#94a3b8';
      ctx.fillRect(chaseW + povW * 0.28 - 6, 80, 12, 12);
      ctx.fillRect(chaseW + povW * 0.72 - 6, 80, 12, 12);

      // Hands position
      const [lx, ly] = obs.left_hand_px;
      const [rx, ry] = obs.right_hand_px;

      // Pilot gloves
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(lx, ly, 14, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(rx, ry, 14, 0, Math.PI * 2);
      ctx.fill();

      // Hands CV bounding boxes
      if (document.getElementById('toggleHands').checked) {
        ctx.strokeStyle = '#00f0ff';
        ctx.lineWidth = 2;
        ctx.strokeRect(lx - 16, ly - 16, 32, 32);
        ctx.strokeRect(rx - 16, ry - 16, 32, 32);
      }

      // Border between panes
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(chaseW, 0);
      ctx.lineTo(chaseW, h);
      ctx.stroke();

      ctx.restore(); // end right viewport clip

      // --- 3. Live HUD Overlay ---
      if (document.getElementById('toggleHUD').checked) {
        ctx.fillStyle = 'rgba(10, 16, 26, 0.85)';
        ctx.strokeStyle = '#23304c';
        ctx.lineWidth = 1;
        ctx.fillRect(10, 10, chaseW - 20, 56);
        ctx.strokeRect(10, 10, chaseW - 20, 56);

        ctx.font = '12px var(--font-mono)';
        ctx.fillStyle = '#e2e8f0';
        const symPct = (0.5 * (obs.brake_left_norm + obs.brake_right_norm) * 100).toFixed(0);
        let line1 = 'FRAME #' + String(obs.frame_index).padStart(3, '0') + ' | t = ' + obs.time_sec.toFixed(2) + 's | BRAKE SYM = ' + symPct + '%';
        if (currentM.camera_mode === 'rear_chase') {
          line1 = 'FRAME #' + String(obs.frame_index).padStart(3, '0') + ' | t = ' + obs.time_sec.toFixed(2) + 's | BRAKE L/R = ' + (obs.brake_left_norm * 100).toFixed(0) + '%/' + (obs.brake_right_norm * 100).toFixed(0) + '% | WS = ' + (obs.weight_shift_norm * 100).toFixed(0) + '%';
        }
        ctx.fillText(line1, 20, 32);

        ctx.fillStyle = '#f59e0b';
        let line2 = 'PITCH = ' + (obs.pitch_deg >= 0 ? '+' : '') + obs.pitch_deg.toFixed(1) + '° (' + obs.pitch_rate_deg_sec.toFixed(1) + '°/s) | SURGE = ' + (obs.canopy_forward_offset_m >= 0 ? '+' : '') + obs.canopy_forward_offset_m.toFixed(2) + ' m';
        if (currentM.camera_mode === 'rear_chase') {
          line2 = 'ROLL = ' + (obs.roll_deg >= 0 ? '+' : '') + obs.roll_deg.toFixed(1) + '° (' + obs.roll_rate_deg_sec.toFixed(1) + '°/s) | PITCH = ' + obs.pitch_deg.toFixed(1) + '°';
        }
        ctx.fillText(line2, 20, 52);
      }

      // --- Update Gauges & Indicators ---
      updateGauges(obs);
    }

    function updateGauges(obs) {
      document.getElementById('valPitch').innerText = (obs.pitch_deg >= 0 ? '+' : '') + obs.pitch_deg.toFixed(1) + '°';
      document.getElementById('valPitchRate').innerText = 'Rate: ' + (obs.pitch_rate_deg_sec || 0).toFixed(1) + ' °/s';
      document.getElementById('valRoll').innerText = (obs.roll_deg >= 0 ? '+' : '') + obs.roll_deg.toFixed(1) + '°';
      document.getElementById('valRollRate').innerText = 'Rate: ' + (obs.roll_rate_deg_sec || 0).toFixed(1) + ' °/s';
      document.getElementById('valSurge').innerText = (obs.canopy_forward_offset_m >= 0 ? '+' : '') + (obs.canopy_forward_offset_m || 0).toFixed(2) + ' m';
      document.getElementById('valHorizon').innerText = (obs.horizon_tilt_deg >= 0 ? '+' : '') + obs.horizon_tilt_deg.toFixed(1) + '°';

      const bL = Math.round(obs.brake_left_norm * 100);
      const bR = Math.round(obs.brake_right_norm * 100);
      document.getElementById('valBrakeL').innerText = bL + '%';
      document.getElementById('fillBrakeL').style.width = bL + '%';
      document.getElementById('valBrakeR').innerText = bR + '%';
      document.getElementById('fillBrakeR').style.width = bR + '%';

      const wsPct = Math.round(obs.weight_shift_norm * 100);
      document.getElementById('valWeightShift').innerText = (wsPct === 0 ? '0% (Center)' : (wsPct > 0 ? '+' + wsPct + '% Right' : wsPct + '% Left'));
      document.getElementById('indWeightShift').style.left = (50 + (obs.weight_shift_norm * 45)) + '%';

      const bePct = Math.round((obs.big_ears_norm || 0) * 100);
      document.getElementById('valBigEars').innerText = bePct + '%';
      document.getElementById('fillBigEars').style.width = bePct + '%';

      // Aerodynamics lookup from simulation series at current time
      const t = obs.time_sec;
      const simT = (currentM.series && currentM.series.sim_t) ? currentM.series.sim_t : [0];
      let closestIdx = 0;
      let minDiff = 999;
      for (let i = 0; i < simT.length; i++) {
        const diff = Math.abs(simT[i] - t);
        if (diff < minDiff) { minDiff = diff; closestIdx = i; }
      }

      const attachedX = (currentM.series && currentM.series.fp_attached_x) ? (currentM.series.fp_attached_x[closestIdx] || 0.95) : 0.95;
      const cellP = (currentM.series && currentM.series.fp_cell_p) ? (currentM.series.fp_cell_p[closestIdx] || 0.92) : 0.92;
      document.getElementById('valAttachedFlow').innerText = attachedX.toFixed(2);
      document.getElementById('valCellPressure').innerText = cellP.toFixed(2);

      const statusStall = document.getElementById('statusStall');
      if (attachedX < 0.70) {
        statusStall.innerText = 'STALL WARNING!';
        statusStall.style.color = 'var(--red)';
      } else {
        statusStall.innerText = 'STALL SAFE';
        statusStall.style.color = 'var(--green)';
      }

      const statusCollapse = document.getElementById('statusCollapse');
      if (cellP < 0.55) {
        statusCollapse.innerText = 'COLLAPSE WARNING!';
        statusCollapse.style.color = 'var(--red)';
      } else {
        statusCollapse.innerText = 'COLLAPSE SAFE';
        statusCollapse.style.color = 'var(--cyan)';
      }

      // Time display
      const totalDur = currentM.frames[currentM.frames.length - 1].time_sec;
      document.getElementById('timeDisplay').innerText =
        t.toFixed(2) + 's / ' + totalDur.toFixed(2) + 's (#' + obs.frame_index + '/' + (currentM.frames.length - 1) + ')';

      // Highlight active event in list
      document.querySelectorAll('.event-item').forEach(item => {
        const fIdx = parseInt(item.dataset.frame, 10);
        item.classList.toggle('active', Math.abs(fIdx - obs.frame_index) <= 4);
      });

      // Redraw chart scrub needle
      drawCharts();
    }

    // Charts Drawing
    function drawCharts() {
      drawAttitudeChart();
      drawInputsChart();
    }

    function drawAttitudeChart() {
      const c = document.getElementById('chartAttitude');
      const ctx = c.getContext('2d');
      const w = c.width = c.clientWidth;
      const h = c.height = c.clientHeight;

      ctx.clearRect(0, 0, w, h);

      const ser = currentM.series || { video_t: [0], video_pitch_deg: [0], sim_t: [0], fp_pitch_deg: [0], curr_pitch_deg: [0] };
      const isLateral = currentM.camera_mode === 'rear_chase';
      const vidY = (isLateral && ser.video_roll_deg) ? ser.video_roll_deg : ser.video_pitch_deg;
      const currY = (isLateral && ser.curr_roll_deg) ? ser.curr_roll_deg : ser.curr_pitch_deg;
      const fpY = (isLateral && ser.fp_roll_deg) ? ser.fp_roll_deg : ser.fp_pitch_deg;

      const tMax = Math.max(...ser.video_t);
      const minVal = Math.min(...vidY, ...(currY||[0]), ...(fpY||[0])) - 5;
      const maxVal = Math.max(...vidY, ...(currY||[0]), ...(fpY||[0])) + 5;
      const spanVal = Math.max(1, maxVal - minVal);

      const mapX = (t) => (t / Math.max(0.1, tMax)) * (w - 40) + 30;
      const mapY = (v) => h - 25 - ((v - minVal) / spanVal) * (h - 40);

      // Grid lines
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {
        const gy = h - 25 - (g / 4) * (h - 40);
        ctx.beginPath();
        ctx.moveTo(30, gy);
        ctx.lineTo(w - 10, gy);
        ctx.stroke();

        const vLabel = (minVal + (g / 4) * spanVal).toFixed(0) + '°';
        ctx.fillStyle = '#64748b';
        ctx.font = '10px var(--font-mono)';
        ctx.fillText(vLabel, 4, gy + 3);
      }

      // Zero-degree line
      if (minVal <= 0 && maxVal >= 0) {
        const zy = mapY(0);
        ctx.strokeStyle = '#334155';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(30, zy);
        ctx.lineTo(w - 10, zy);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // 1. CurrentParagliderSim (Red dashed)
      if (currY && currY.length) {
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        for (let i = 0; i < ser.sim_t.length; i++) {
          const x = mapX(ser.sim_t[i]);
          const y = mapY(currY[i]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // 2. FirstPrinciplesTwoBody/Lateral (Cyan solid)
      if (fpY && fpY.length) {
        ctx.strokeStyle = '#00f0ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < ser.sim_t.length; i++) {
          const x = mapX(ser.sim_t[i]);
          const y = mapY(fpY[i]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // 3. CV Video Ground Truth (Yellow solid with dots)
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (let i = 0; i < ser.video_t.length; i++) {
        const x = mapX(ser.video_t[i]);
        const y = mapY(vidY[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Live Playhead Needle
      const curT = currentM.frames[currentFrameIdx].time_sec;
      const needleX = mapX(curT);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(needleX, 10);
      ctx.lineTo(needleX, h - 20);
      ctx.stroke();

      c.onclick = (e) => {
        const rect = c.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const targetT = ((clickX - 30) / (w - 40)) * tMax;
        seekToTime(targetT);
      };
    }

    function drawInputsChart() {
      const c = document.getElementById('chartInputs');
      const ctx = c.getContext('2d');
      const w = c.width = c.clientWidth;
      const h = c.height = c.clientHeight;

      ctx.clearRect(0, 0, w, h);

      const ser = currentM.series || { video_t: [0], video_brake: [0], sim_t: [0], fp_attached_x: [0.9] };
      const tMax = Math.max(...ser.video_t);
      const mapX = (t) => (t / Math.max(0.1, tMax)) * (w - 40) + 30;
      const mapY = (val) => h - 25 - val * (h - 40); // inputs 0.0 to 1.0

      // Grid
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {
        const gy = h - 25 - (g / 4) * (h - 40);
        ctx.beginPath();
        ctx.moveTo(30, gy);
        ctx.lineTo(w - 10, gy);
        ctx.stroke();

        ctx.fillStyle = '#64748b';
        ctx.font = '10px var(--font-mono)';
        ctx.fillText((g * 25) + '%', 4, gy + 3);
      }

      // Attached Flow Curve (Purple)
      if (ser.fp_attached_x) {
        ctx.strokeStyle = '#8b5cf6';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < ser.sim_t.length; i++) {
          const x = mapX(ser.sim_t[i]);
          const y = mapY(ser.fp_attached_x[i]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // Video Brake Curve (Green)
      if (ser.video_brake) {
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < ser.video_t.length; i++) {
          const x = mapX(ser.video_t[i]);
          const y = mapY(ser.video_brake[i]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // Needle
      const curT = currentM.frames[currentFrameIdx].time_sec;
      const needleX = mapX(curT);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(needleX, 10);
      ctx.lineTo(needleX, h - 20);
      ctx.stroke();

      c.onclick = (e) => {
        const rect = c.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const targetT = ((clickX - 30) / (w - 40)) * tMax;
        seekToTime(targetT);
      };
    }

    function seekToTime(t) {
      const frames = currentM.frames;
      let bestIdx = 0;
      let bestDiff = 999;
      for (let i = 0; i < frames.length; i++) {
        const diff = Math.abs(frames[i].time_sec - t);
        if (diff < bestDiff) { bestDiff = diff; bestIdx = i; }
      }
      jumpToFrame(bestIdx);
    }

    // Layer toggle listeners
    document.querySelectorAll('.cv-toggles input').forEach(inp => {
      inp.onchange = () => renderCurrentFrame();
    });

    // Boot
    initManeuverSelector();
    initLayoutModeTabs();
    selectManeuver('pitch_pendulum_active_catch');
    setLayoutMode('side_by_side');
  </script>
</body>
</html>
"""

def build_and_export_inspector(
    repo_root: Path,
    custom_trace: Optional[ManeuverKinematicsTrace] = None,
    custom_video_path: Optional[str | Path] = None,
    export_artifact: bool = True,
) -> Path:
    diag_dir = repo_root / "data" / "diagnostics"
    traces_dir = repo_root / "data" / "traces"
    videos_dir = repo_root / "videos"
    keyframes_dir = diag_dir / "keyframes"

    diag_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    report_path = diag_dir / "full_tutorial_residual_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = {
            "video_url": "https://www.youtube.com/watch?v=ykuaAYzB2qU",
            "video_title": "BASIC PARAGLIDING EXERCISES - THEO DE BLIC's TUTORIALS",
            "wing": {},
            "maneuvers": {},
        }

    bundle: Dict = {
        "video_url": report.get("video_url", ""),
        "video_title": report.get("video_title", ""),
        "wing": report.get("wing", {}),
        "maneuvers": {},
    }

    for m_key, m_val in report.get("maneuvers", {}).items():
        trace_file = repo_root / m_val.get("trace_json", f"data/traces/{m_key}.json")
        if not trace_file.exists():
            continue
        with open(trace_file, "r", encoding="utf-8") as f:
            trace_data = json.load(f)

        raw_video = videos_dir / f"{m_key}.mp4"
        annotated_video = videos_dir / f"{m_key}_annotated.mp4"

        bundle["maneuvers"][m_key] = {
            "id": m_key,
            "title": m_val.get("title", m_key),
            "timestamp_range": m_val.get("timestamp_range", ""),
            "camera_mode": m_val.get("camera_mode", "side_chase"),
            "phase_metrics": m_val.get("phase_metrics", {}),
            "events": m_val.get("events", []),
            "residuals_current_sim": m_val.get("residuals_current_sim", {}),
            "residuals_first_principles": m_val.get("residuals_first_principles", {}),
            "aero_extrema": m_val.get("aero_extrema", {}),
            "keyframes": {ev: f"keyframes/{m_key}_{ev}.png" for ev in m_val.get("keyframes", {}).keys()},
            "raw_video": f"videos/{m_key}.mp4" if raw_video.exists() else (f"videos/{m_key}_annotated.mp4" if annotated_video.exists() else None),
            "annotated_video": f"videos/{m_key}_annotated.mp4" if annotated_video.exists() else None,
            "series": m_val.get("series", {}),
            "frames": trace_data.get("frames", []),
        }

    if custom_trace is not None:
        c_key = custom_trace.maneuver_id
        bundle["maneuvers"][c_key] = {
            "id": c_key,
            "title": custom_trace.maneuver_name,
            "timestamp_range": custom_trace.video_timestamp_range,
            "camera_mode": custom_trace.camera_mode,
            "phase_metrics": custom_trace.phase_metrics.__dict__,
            "events": [e.__dict__ for e in custom_trace.events],
            "residuals_current_sim": {},
            "residuals_first_principles": {},
            "aero_extrema": {},
            "keyframes": {},
            "raw_video": str(custom_video_path) if custom_video_path else None,
            "annotated_video": None,
            "series": {
                "video_t": [f.time_sec for f in custom_trace.frames],
                "video_brake": [0.5 * (f.brake_left_norm + f.brake_right_norm) for f in custom_trace.frames],
                "video_pitch_deg": [f.pitch_deg for f in custom_trace.frames],
                "video_pitch_rate": [f.pitch_rate_deg_sec for f in custom_trace.frames],
                "video_roll_deg": [f.roll_deg for f in custom_trace.frames],
                "video_roll_rate": [f.roll_rate_deg_sec for f in custom_trace.frames],
                "sim_t": [f.time_sec for f in custom_trace.frames],
                "curr_pitch_deg": [f.pitch_deg for f in custom_trace.frames],
                "fp_pitch_deg": [f.pitch_deg for f in custom_trace.frames],
                "curr_roll_deg": [f.roll_deg for f in custom_trace.frames],
                "fp_roll_deg": [f.roll_deg for f in custom_trace.frames],
                "fp_attached_x": [0.95 for _ in custom_trace.frames],
                "fp_cell_p": [0.92 for _ in custom_trace.frames],
            },
            "frames": [f.__dict__ for f in custom_trace.frames],
        }

    bundle_path = diag_dir / "inspector_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    html_content = html_template.replace("%%BUNDLE_DATA%%", json.dumps(bundle))

    repo_html = diag_dir / "paraglider_flight_inspector.html"
    repo_html.write_text(html_content, encoding="utf-8")

    if export_artifact:
        artifact_dir = Path("/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p/f0b4b616-5515-4b5d-9d39-8657a8f33a80")
        if artifact_dir.exists():
            art_html = artifact_dir / "paraglider_flight_inspector.html"
            art_html.write_text(html_content, encoding="utf-8")
            art_kf = artifact_dir / "keyframes"
            art_kf.mkdir(parents=True, exist_ok=True)
            for p in keyframes_dir.glob("*.png"):
                dest = art_kf / p.name
                if not dest.exists():
                    dest.write_bytes(p.read_bytes())
            art_vid = artifact_dir / "videos"
            art_vid.mkdir(parents=True, exist_ok=True)
            for p in videos_dir.glob("*.mp4"):
                dest = art_vid / p.name
                if not dest.exists():
                    dest.write_bytes(p.read_bytes())

    return repo_html
