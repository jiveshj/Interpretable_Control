#!/usr/bin/env python
"""
make_probe_html_ltfv6.py

Single combined HTML report for ALL LTFv6 probe labels.

Reads four JSON files:
  main sweep (ungrouped + grouped): probe_results_ltfv6.json / probe_results_ltfv6_grouped_cv.json
  extended labels (ungrouped + grouped): probe_extended_results_ltfv6.json / probe_extended_results_ltfv6_grouped_cv.json

Output: one HTML with a label tab per label and a CV toggle per label.

Label types and rendering:
  main_dist  : dist_nearest_front     -- 4-col global table (R²dist, R²log, R²bins, Acc) + per-bin table
  dist_reg   : dist_2nd_nearest_*     -- 2-col global table (R², MAE m) + per-bin table
  reg        : nearest_front_heading  -- 2-col global table (R², MAE rad)
  clf        : has_vehicle_front, same_lane_binary, opposing_lane_binary -- acc + balanced-acc table

Usage:
    python make_probe_html_ltfv6.py
    python make_probe_html_ltfv6.py --out /path/probe_results_ltfv6.html
"""

import argparse
import json
import os

MAIN_U_DEFAULT  = "/jet/home/jjain2/Interpretable_Control/probe_results_ltfv6.json"
MAIN_G_DEFAULT  = "/jet/home/jjain2/Interpretable_Control/probe_results_ltfv6_grouped_cv.json"
EXT_U_DEFAULT   = "/jet/home/jjain2/Interpretable_Control/probe_extended_results_ltfv6.json"
EXT_G_DEFAULT   = "/jet/home/jjain2/Interpretable_Control/probe_extended_results_ltfv6_grouped_cv.json"
OUT_DEFAULT     = "/jet/home/jjain2/Interpretable_Control/probe_results_ltfv6.html"

LABEL_META = {
    "dist_nearest_front": {
        "title": "Nearest front vehicle distance",
        "desc": (
            "Euclidean distance (metres) to the nearest valid traffic agent in the "
            "front half (X&nbsp;&gt;&nbsp;0) of the ego frame. Samples &ge;&nbsp;40&nbsp;m "
            "are clipped into the '20–40 m+' bin. "
            "This is the primary label — probed with 4 global metrics "
            "(R²(dist), R²(log), R²(bins), Acc(bins)) plus a per-bin specialist probe. "
            "R²(dist): predicts raw metres. R²(log): predicts ln(distance). "
            "R²(bins): predicts bin index 0–3 as regression. Acc(bins): 4-class bin classifier, fraction 0–1."
        ),
        "mae_unit": "m",
        "type": "main_dist",
    },
    "dist_2nd_nearest_any": {
        "title": "2nd nearest agent (any direction)",
        "desc": (
            "Euclidean distance (metres) to the 2nd-closest valid traffic agent "
            "in any direction. NaN (excluded) when fewer than 2 agents are present."
        ),
        "mae_unit": "m",
        "type": "dist_reg",
    },
    "dist_2nd_nearest_front": {
        "title": "2nd nearest front agent",
        "desc": (
            "Euclidean distance (metres) to the 2nd-closest valid traffic agent "
            "with X&nbsp;&gt;&nbsp;0 (in front of ego). "
            "NaN when fewer than 2 front agents are present."
        ),
        "mae_unit": "m",
        "type": "dist_reg",
    },
    "nearest_front_heading": {
        "title": "Nearest front agent heading",
        "desc": (
            "Heading of the nearest front agent in ego frame, in "
            "<strong>radians</strong> (range &minus;&pi; to +&pi;). "
            "0&nbsp;rad = same direction as ego. "
            "&plusmn;&pi;&nbsp;rad (&plusmn;180&deg;) = oncoming. "
            "NaN when no front agent is present. "
            "<strong>MAE unit is radians, not metres.</strong> "
            "Same-direction threshold: |h|&nbsp;&lt;&nbsp;&pi;/6 (&asymp;30&deg;). "
            "Oncoming threshold: |h|&nbsp;&gt;&nbsp;5&pi;/6 (&asymp;150&deg;)."
        ),
        "mae_unit": "rad",
        "type": "reg",
    },
    "has_vehicle_front": {
        "title": "Vehicle in front (binary)",
        "desc": (
            "Binary: 1 if any valid traffic agent has X&nbsp;&gt;&nbsp;0 "
            "(is in front of ego), 0 otherwise. Always 0 or 1, no NaN. "
            "Metrics: accuracy and balanced accuracy, both as fraction 0–1."
        ),
        "type": "clf",
    },
    "same_lane_binary": {
        "title": "Same-direction traffic (binary)",
        "desc": (
            "Binary: 1 if nearest front agent heading |h|&nbsp;&lt;&nbsp;&pi;/6 (&asymp;30&deg;) "
            "— same-direction traffic. 0 otherwise. "
            "NaN when no front agent (excluded from probe)."
        ),
        "type": "clf",
    },
    "opposing_lane_binary": {
        "title": "Oncoming traffic (binary)",
        "desc": (
            "Binary: 1 if nearest front agent heading |h|&nbsp;&gt;&nbsp;5&pi;/6 (&asymp;150&deg;) "
            "— oncoming traffic. 0 otherwise. "
            "NaN when no front agent (excluded from probe)."
        ),
        "type": "clf",
    },
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LTFv6 Probe Results — {label_name}</title>
<style>
  :root {{
    --bg: #0f1117; --panel: #1a1d27; --border: #2a2e3a; --text: #e6e6e6;
    --muted: #9aa0ad; --accent: #5fb3ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); margin: 0; padding: 24px 32px 80px;
    font-size: 14px;
  }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  h2 {{ font-size: 17px; margin-top: 32px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  p.sub {{ color: var(--muted); margin-top: 0; }}
  p.note {{ color: var(--muted); font-size: 12.5px; margin: 4px 0 14px; max-width: 920px; }}
  .desc-box {{ background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
               padding: 10px 14px; margin: 10px 0 16px; font-size: 13px; color: #c8cdd8;
               max-width: 920px; line-height: 1.6; }}

  table {{ border-collapse: collapse; width: 100%; margin-bottom: 8px; }}
  th, td {{ padding: 6px 9px; text-align: center; border: 1px solid var(--border); white-space: nowrap; }}
  th {{ background: #181b24; cursor: pointer; user-select: none; position: sticky; top: 0; }}
  th.layer-col, td.layer-col {{ text-align: left; position: sticky; left: 0;
    background: #14161e; z-index: 2; font-family: monospace; font-size: 12px; }}
  th.sortasc::after {{ content: " \\25B2"; }}
  th.sortdesc::after {{ content: " \\25BC"; }}
  tr:hover td {{ filter: brightness(1.15); }}
  .tablewrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 6px; }}
  .bin-group {{ border-left: 2px solid var(--border); }}
  .r2-val {{ color: #9cc4ff; }}
  .mae-val {{ color: #ffd9a0; }}

  .cell {{ border-radius: 3px; padding: 4px 6px; display: inline-block; min-width: 56px; font-weight: 600; }}
  .best {{ outline: 2px solid #ffd24c; outline-offset: -2px; }}

  .findings {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
               padding: 16px 20px; margin: 18px 0; }}
  .findings li {{ margin-bottom: 6px; }}
  .badge {{ display:inline-block; background:#26304a; color:#9cc4ff; border-radius:4px;
            padding:1px 6px; font-size:11px; margin-left:6px; }}
  code {{ background:#222536; padding:1px 5px; border-radius:3px; }}

  .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 0; }}
  .tab-btn {{ background: var(--panel); border: 1px solid var(--border); color: var(--text);
              padding: 7px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }}
  .tab-btn.active {{ background: var(--accent); color: #08110a; font-weight: 600; }}

  .toggle {{ display: inline-flex; border: 1px solid var(--border); border-radius: 8px;
             overflow: hidden; margin: 14px 0 20px; }}
  .toggle button {{ background: #1a1d27; color: var(--text); border: none;
                   padding: 8px 18px; cursor: pointer; font-size: 13px; }}
  .toggle button.active {{ background: var(--accent); color: #08110a; font-weight: 600; }}

  footer {{ color: var(--muted); font-size: 12px; margin-top: 50px; }}
</style>
</head>
<body>

<h1>LatentTransfuserV6 — Probe Results</h1>
<p class="sub">{n_layers} layers &times; 5-fold cross-validated linear (Ridge) probes &mdash; {n_labels} labels.</p>

<div class="findings">
  <strong>Reading this report</strong>
  <ul>
    <li><strong>Ungrouped</strong> KFold shuffles all frames randomly: frames from the same drive can
      appear in both train and test (leaky, optimistic). <strong>Grouped</strong> KFold holds out
      entire logs — no frame from a test log was seen in training (honest).
      <span class="badge">use grouped for real conclusions</span></li>
    <li>The <strong>global table</strong> for <em>dist_nearest_front</em> has one probe per layer
      trained on the full 3–166 m range. The <strong>per-bin table</strong> has a completely separate
      probe per layer <em>per bin</em> — their R² and MAE are a <em>pair</em> from the same model
      and must not be cross-compared with the global table.
      <span class="badge">two different models</span></li>
    <li>Heading MAE is in <strong>radians</strong> (1 rad &asymp; 57&deg;), not metres.
      Acc/balanced-acc for binary labels are <strong>fractions 0–1</strong>, not %.</li>
  </ul>
</div>

<!-- Label tabs -->
<div class="tabs" id="label-tabs">{tab_buttons}</div>

<!-- CV toggle -->
<div class="toggle" id="cv-toggle">
  <button id="btn-ungrouped" class="active">Ungrouped (leaky KFold)</button>
  <button id="btn-grouped">Grouped (clean, by log)</button>
</div>

<!-- Label description -->
<div class="desc-box" id="label-desc"></div>

<!-- Table area -->
<div id="table-area"></div>

<footer>Generated from probe_results_ltfv6.json, probe_results_ltfv6_grouped_cv.json,
  probe_extended_results_ltfv6.json, probe_extended_results_ltfv6_grouped_cv.json
  &mdash; RidgeCV/RidgeClassifierCV, alphas=[0.1,1,10,100,1e3,1e4]</footer>

<script>
const DATA = {{ ungrouped: {data_ungrouped}, grouped: {data_grouped} }};
const LAYERS = {layers_json};
const LABEL_META = {label_meta_json};
let MODE = "ungrouped";
let LABEL = Object.keys(LABEL_META)[0];

function pm(mean, std, d=3) {{
  if (mean == null || Number.isNaN(mean)) return "—";
  return mean.toFixed(d) + "±" + (std != null ? std.toFixed(d) : "?");
}}
function maeFmt(v, std, unit) {{
  if (v == null || Number.isNaN(v)) return "—";
  if (unit === "rad") return v.toFixed(3) + " rad (" + Math.round(v*180/Math.PI) + "°) ±" + std.toFixed(3);
  return v.toFixed(3) + " m ±" + std.toFixed(3) + " m";
}}
function colorFor(v, lo, hi) {{
  let t = Math.max(0, Math.min(1, (v-lo)/(hi-lo)));
  return `rgb(${{Math.round(178-t*120)}},${{Math.round(58+t*140)}},${{Math.round(58+t*30)}})`;
}}
function makeSortable(table) {{
  table.querySelectorAll("th").forEach((th, idx) => {{
    th.addEventListener("click", () => {{
      const tbody = table.querySelector("tbody");
      const rows = Array.from(tbody.querySelectorAll("tr"));
      const asc = !th.classList.contains("sortasc");
      table.querySelectorAll("th").forEach(t => t.classList.remove("sortasc","sortdesc"));
      th.classList.add(asc ? "sortasc" : "sortdesc");
      rows.sort((a,b) => {{
        const va = a.children[idx].dataset.val !== undefined ? parseFloat(a.children[idx].dataset.val) : a.children[idx].textContent;
        const vb = b.children[idx].dataset.val !== undefined ? parseFloat(b.children[idx].dataset.val) : b.children[idx].textContent;
        if (!Number.isNaN(+va) && !Number.isNaN(+vb)) return asc ? va-vb : vb-va;
        return asc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
      }});
      rows.forEach(r => tbody.appendChild(r));
    }});
  }});
}}
function colorizeColumn(table, colIdx, lo, hi, bestIsMax=true) {{
  const tds = Array.from(table.querySelectorAll("tbody tr")).map(r => r.children[colIdx]);
  let best = bestIsMax ? -Infinity : Infinity;
  tds.forEach(td => {{ const v=parseFloat(td.dataset.val); if (!Number.isNaN(v)) best=bestIsMax?Math.max(best,v):Math.min(best,v); }});
  tds.forEach(td => {{
    const v = parseFloat(td.dataset.val); if (Number.isNaN(v)) return;
    td.innerHTML = `<span class="cell ${{v===best?'best':''}}" style="background:${{colorFor(v,lo,hi)}};color:#08110a">${{td.textContent.trim()}}</span>`;
  }});
}}

function buildMainDistGlobal(d) {{
  const maj = d.majority_baseline;
  let html = `<h2>Global table <span style="font-size:13px;color:var(--muted);font-weight:normal">(full-range probe, one per layer)</span></h2>
  <p class="note">n_samples=<code>${{d.n_valid}}</code> &nbsp; majority-class baseline (bins)=<code>${{maj.toFixed(3)}}</code>
  &nbsp; Gold outline = best layer per column. Acc(bins) is a fraction 0–1.</p>
  <div class="tablewrap"><table id="global-table"><thead><tr>
    <th class="layer-col">Layer</th>
    <th title="R² predicting raw distance in metres. 1=perfect, 0=no better than mean, negative=worse.">R²(dist)</th>
    <th title="R² predicting ln(distance). Captures proportional distance (5→10 m same as 10→20 m).">R²(log)</th>
    <th title="R² predicting bin index 0–3 as regression. Higher than R²(dist) = layer encodes coarse separation better than fine metric distance.">R²(bins)</th>
    <th title="4-class classification accuracy: which 3–5/5–10/10–20/20–40m+ bin? Fraction 0–1.">Acc(bins) [0–1]</th>
  </tr></thead><tbody>`;
  LAYERS.forEach(k => {{
    const L = d.layers[k]; if (!L) return;
    html += `<tr><td class="layer-col">${{k}}</td>
      <td data-val="${{L.r2_dist.mean}}">${{pm(L.r2_dist.mean,L.r2_dist.std)}}</td>
      <td data-val="${{L.r2_log.mean}}">${{pm(L.r2_log.mean,L.r2_log.std)}}</td>
      <td data-val="${{L.r2_bins.mean}}">${{pm(L.r2_bins.mean,L.r2_bins.std)}}</td>
      <td data-val="${{L.acc.mean}}">${{pm(L.acc.mean,L.acc.std)}}</td>
    </tr>`;
  }});
  html += `</tbody></table></div>`;
  return html;
}}

function buildMainDistPerBin(d) {{
  const bins = d.bin_names;
  let html = `<h2>Per-bin table <span style="font-size:13px;color:var(--muted);font-weight:normal">(specialist probe per layer per bin — different model from global table above)</span></h2>
  <p class="note">R² (<span class="r2-val">blue</span>) and MAE (<span class="mae-val">orange</span>) are
  from the SAME bin-restricted model — read them as a pair. Bin edges: <code>${{d.bin_edges}}</code> →
  bins are 3–5 m (2 m wide), 5–10 m (5 m), 10–20 m (10 m), 20–40 m+ (20 m+).
  Negative R² = worse than predicting the bin's mean distance.</p>
  <div class="tablewrap"><table id="perbin-table"><thead><tr>
    <th class="layer-col">Layer</th>`;
  bins.forEach(b => {{
    html += `<th class="bin-group" title="R² of specialist probe for ${{b}}">${{b}} R²</th>
             <th title="MAE (m) of specialist probe for ${{b}}">${{b}} MAE (m)</th>`;
  }});
  html += `</tr></thead><tbody>`;
  LAYERS.forEach(k => {{
    const L = d.layers[k]; if (!L) return;
    html += `<tr><td class="layer-col">${{k}}</td>`;
    bins.forEach(b => {{
      const pb = L.per_bin?.[b];
      if (!pb || pb.note) {{ html += `<td class="bin-group" data-val="NaN">—</td><td data-val="NaN">—</td>`; return; }}
      html += `<td class="bin-group" data-val="${{pb.r2.mean}}"><span class="r2-val">${{pm(pb.r2.mean,pb.r2.std)}}</span></td>`;
      html += `<td data-val="${{pb.mae.mean}}"><span class="mae-val">${{pb.mae.mean.toFixed(3)}}m±${{pb.mae.std.toFixed(3)}}m</span></td>`;
    }});
    html += `</tr>`;
  }});
  html += `</tbody></table></div>`;
  return html;
}}

function buildRegGlobal(d, meta) {{
  const unit = meta.mae_unit;
  const maeHead = unit==="rad" ? `MAE (rad) — 1 rad ≈ 57°` : `MAE (m)`;
  let html = `<h2>Global table</h2>
  <p class="note">n_valid=<code>${{d.n_valid}}</code></p>
  <div class="tablewrap"><table id="global-table"><thead><tr>
    <th class="layer-col">Layer</th>
    <th title="R²: 1=perfect, 0=no better than mean, negative=worse than mean">R²</th>
    <th title="Mean Absolute Error in ${{unit}}">${{maeHead}}</th>
  </tr></thead><tbody>`;
  LAYERS.forEach(k => {{
    const L = d.layers[k]; if (!L) return;
    html += `<tr><td class="layer-col">${{k}}</td>
      <td data-val="${{L.r2.mean}}">${{pm(L.r2.mean,L.r2.std)}}</td>
      <td data-val="${{L.mae.mean}}">${{maeFmt(L.mae.mean,L.mae.std,unit)}}</td>
    </tr>`;
  }});
  html += `</tbody></table></div>`;
  return html;
}}

function buildDistRegPerBin(d) {{
  const bins = ["3-5m","5-10m","10-20m","20-40m+"];
  let html = `<h2>Per-bin table <span style="font-size:13px;color:var(--muted);font-weight:normal">(specialist probe per layer per bin)</span></h2>
  <p class="note">Negative R² = probe is worse than predicting the bin's own mean distance.</p>
  <div class="tablewrap"><table id="perbin-table"><thead><tr>
    <th class="layer-col">Layer</th>`;
  bins.forEach(b => {{
    html += `<th class="bin-group">${{b}} R²</th><th>${{b}} MAE (m)</th>`;
  }});
  html += `</tr></thead><tbody>`;
  LAYERS.forEach(k => {{
    const L = d.layers[k]; if (!L) return;
    html += `<tr><td class="layer-col">${{k}}</td>`;
    bins.forEach(b => {{
      const pb = L.per_bin?.[b];
      if (!pb || pb.note || pb.r2==null) {{ html += `<td class="bin-group" data-val="NaN">—</td><td data-val="NaN">—</td>`; return; }}
      html += `<td class="bin-group" data-val="${{pb.r2.mean}}"><span class="r2-val">${{pm(pb.r2.mean,pb.r2.std)}}</span></td>`;
      html += `<td data-val="${{pb.mae.mean}}"><span class="mae-val">${{pb.mae.mean.toFixed(3)}}m±${{pb.mae.std.toFixed(3)}}m</span></td>`;
    }});
    html += `</tr>`;
  }});
  html += `</tbody></table></div>`;
  return html;
}}

function buildClfTable(d) {{
  const firstLayer = Object.values(d.layers)[0];
  const maj = firstLayer.majority_baseline;
  const counts = firstLayer.class_counts;
  let html = `<h2>Classification table</h2>
  <p class="note">n_valid=<code>${{d.n_valid}}</code> &nbsp;
    Majority-class baseline: <code>${{maj.toFixed(3)}}</code> &nbsp;
    Class counts: <code>0→${{counts[0]}}, 1→${{counts[1]}}</code> &nbsp;
    All metrics are fractions 0–1.</p>
  <div class="tablewrap"><table id="global-table"><thead><tr>
    <th class="layer-col">Layer</th>
    <th title="Fraction of samples correctly classified (0–1).">Accuracy [0–1]</th>
    <th title="Average per-class recall: (recall_class0 + recall_class1)/2. Chance = 0.5. Better for imbalanced classes.">Balanced Acc [0–1]</th>
  </tr></thead><tbody>`;
  LAYERS.forEach(k => {{
    const L = d.layers[k]; if (!L) return;
    html += `<tr><td class="layer-col">${{k}}</td>
      <td data-val="${{L.acc.mean}}">${{pm(L.acc.mean,L.acc.std)}}</td>
      <td data-val="${{L.balanced_acc.mean}}">${{pm(L.balanced_acc.mean,L.balanced_acc.std)}}</td>
    </tr>`;
  }});
  html += `</tbody></table></div>`;
  return html;
}}

function colorizeAll() {{
  const globalTable = document.getElementById("global-table");
  if (globalTable) {{
    makeSortable(globalTable);
    const meta = LABEL_META[LABEL];
    const d = DATA[MODE][LABEL];
    if (meta.type === "main_dist") {{
      ["r2_dist","r2_log","r2_bins","acc"].forEach((key,i) => {{
        const vals = LAYERS.map(k=>d.layers[k]?.[key]?.mean).filter(v=>v!=null&&!Number.isNaN(v));
        colorizeColumn(globalTable, i+1, Math.min(...vals), Math.max(...vals), true);
      }});
    }} else if (meta.type === "clf") {{
      [["acc",true],["balanced_acc",true]].forEach(([key,max],i) => {{
        const vals = LAYERS.map(k=>d.layers[k]?.[key]?.mean).filter(v=>v!=null&&!Number.isNaN(v));
        colorizeColumn(globalTable, i+1, Math.min(...vals), Math.max(...vals), max);
      }});
    }} else {{
      const r2s = LAYERS.map(k=>d.layers[k]?.r2?.mean).filter(v=>v!=null&&!Number.isNaN(v));
      const maes= LAYERS.map(k=>d.layers[k]?.mae?.mean).filter(v=>v!=null&&!Number.isNaN(v));
      colorizeColumn(globalTable, 1, Math.min(...r2s), Math.max(...r2s), true);
      colorizeColumn(globalTable, 2, Math.min(...maes), Math.max(...maes), false);
    }}
  }}
  const pbTable = document.getElementById("perbin-table");
  if (pbTable) {{
    makeSortable(pbTable);
    const d = DATA[MODE][LABEL];
    const bins = d.bin_names || ["3-5m","5-10m","10-20m","20-40m+"];
    bins.forEach((b,i) => {{
      const r2s  = LAYERS.map(k=>d.layers[k]?.per_bin?.[b]?.r2?.mean).filter(v=>v!=null&&!Number.isNaN(v));
      const maes = LAYERS.map(k=>d.layers[k]?.per_bin?.[b]?.mae?.mean).filter(v=>v!=null&&!Number.isNaN(v));
      if (r2s.length)  colorizeColumn(pbTable, 1+i*2,   Math.min(...r2s),  Math.max(...r2s),  true);
      if (maes.length) colorizeColumn(pbTable, 1+i*2+1, Math.min(...maes), Math.max(...maes), false);
    }});
  }}
}}

function rebuild() {{
  const meta = LABEL_META[LABEL];
  const d = DATA[MODE][LABEL];
  document.getElementById("label-desc").innerHTML = meta.desc;
  if (!d) {{ document.getElementById("table-area").innerHTML = "<p>No data for this label/mode.</p>"; return; }}
  let out = "";
  if (meta.type === "main_dist") {{
    out = buildMainDistGlobal(d) + buildMainDistPerBin(d);
  }} else if (meta.type === "dist_reg") {{
    out = buildRegGlobal(d, meta) + buildDistRegPerBin(d);
  }} else if (meta.type === "reg") {{
    out = buildRegGlobal(d, meta);
  }} else {{
    out = buildClfTable(d);
  }}
  document.getElementById("table-area").innerHTML = out;
  colorizeAll();
}}

document.getElementById("btn-ungrouped").addEventListener("click", () => {{
  MODE = "ungrouped";
  document.getElementById("btn-ungrouped").classList.add("active");
  document.getElementById("btn-grouped").classList.remove("active");
  rebuild();
}});
document.getElementById("btn-grouped").addEventListener("click", () => {{
  MODE = "grouped";
  document.getElementById("btn-grouped").classList.add("active");
  document.getElementById("btn-ungrouped").classList.remove("active");
  rebuild();
}});
document.querySelectorAll(".tab-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    LABEL = btn.dataset.label;
    rebuild();
  }});
}});

rebuild();
</script>
</body>
</html>
"""


def load_and_normalize(main_u_path, main_g_path, ext_u_path, ext_g_path):
    """Load all four JSONs and normalize into a single {ungrouped, grouped} dict."""
    with open(main_u_path) as f: main_u = json.load(f)
    with open(main_g_path) as f: main_g = json.load(f)
    with open(ext_u_path)  as f: ext_u  = json.load(f)
    with open(ext_g_path)  as f: ext_g  = json.load(f)

    def wrap_main(d):
        return {
            "type": "main_dist",
            "n_valid": d["n_samples"],
            "majority_baseline": d["majority_baseline"],
            "bin_names": d["bin_names"],
            "bin_edges": d["bin_edges"],
            "layers": d["layers"],
        }

    all_u = {"dist_nearest_front": wrap_main(main_u)}
    all_g = {"dist_nearest_front": wrap_main(main_g)}
    all_u.update(ext_u)
    all_g.update(ext_g)
    return all_u, all_g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main_ungrouped", default=MAIN_U_DEFAULT)
    p.add_argument("--main_grouped",   default=MAIN_G_DEFAULT)
    p.add_argument("--ext_ungrouped",  default=EXT_U_DEFAULT)
    p.add_argument("--ext_grouped",    default=EXT_G_DEFAULT)
    p.add_argument("--out",            default=OUT_DEFAULT)
    args = p.parse_args()

    all_u, all_g = load_and_normalize(
        args.main_ungrouped, args.main_grouped,
        args.ext_ungrouped,  args.ext_grouped,
    )

    labels = [k for k in LABEL_META if k in all_u]
    layers = list(all_u["dist_nearest_front"]["layers"].keys())

    tab_buttons = "".join(
        f'<button class="tab-btn{" active" if i==0 else ""}" data-label="{k}">'
        f'{LABEL_META[k]["title"]}</button>'
        for i, k in enumerate(labels)
    )
    label_meta_js = {
        k: {"title": LABEL_META[k]["title"], "desc": LABEL_META[k]["desc"],
            "type": LABEL_META[k]["type"], "mae_unit": LABEL_META[k].get("mae_unit", "")}
        for k in labels
    }

    html = HTML_TEMPLATE.format(
        label_name=all_u["dist_nearest_front"].get("label", "dist_nearest_front"),
        n_layers=len(layers),
        n_labels=len(labels),
        tab_buttons=tab_buttons,
        data_ungrouped=json.dumps(all_u),
        data_grouped=json.dumps(all_g),
        layers_json=json.dumps(layers),
        label_meta_json=json.dumps(label_meta_js),
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
