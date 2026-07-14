#!/usr/bin/env python
"""
make_probe_viz_html_ltfv6.py

Builds a qualitative HTML from probe_predictions_ltfv6.json (computed by
compute_probe_predictions_ltfv6.py).  Loads images from the feature.gz files
only for the selected representative examples — fast once predictions exist.

For each distance bin (3-5m, 5-10m, 10-20m, 20-40m+) shows three categories:
  Best prediction   -- grouped probe got it right (smallest |error|)
  Worst prediction  -- grouped probe failed badly  (largest |error|)
  Traffic scene     -- 2nd-nearest vehicle < 15m (dense traffic; most-wrong first)

Uses the primary layer specified by --layer (default: backbone_transformers_2).

Usage:
    python make_probe_viz_html_ltfv6.py
    python make_probe_viz_html_ltfv6.py --layer backbone_image_encoder_layer1
    python make_probe_viz_html_ltfv6.py --n_per_bin 4
"""

import argparse
import base64
import gzip
import io
import json
import os
import pickle

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PRED_DEFAULT = "/jet/home/jjain2/Interpretable_Control/probe_predictions_ltfv6.json"
OUT_DEFAULT  = "/jet/home/jjain2/Interpretable_Control/probe_qualitative_viz_ltfv6.html"

IMG_W, IMG_H = 960, 135      # display size (original is 1920x270)
TRAFFIC_THRESH = 15.0        # 2nd-nearest distance threshold for "traffic" category

# Rough front-camera intrinsics for approximate 2D agent marker
# NAVSIM front camera: wide-angle, ~60° FOV horizontally on center portion
CAM_FX, CAM_FY = 1000.0, 1000.0
CAM_CX, CAM_CY = 960.0, 135.0   # principal point in original 1920x270 image
CAM_H = 1.5                       # camera height above ground (metres)

BIN_NAMES = ["3-5m", "5-10m", "10-20m", "20-40m+"]


# ---------------------------------------------------------------------------
# Image loading + annotation
# ---------------------------------------------------------------------------

def load_image_from_feature(feat_path):
    with gzip.open(feat_path, "rb") as f:
        feat = pickle.load(f)
    raw = feat["camera_feature"].numpy().tobytes()
    return Image.open(io.BytesIO(raw)).convert("RGB")


def load_nearest_front_agent(feat_path):
    """Return (x_ego, y_ego) of nearest front agent, or (None, None)."""
    target_path = feat_path.replace("transfuser_feature.gz", "transfuser_target.gz")
    try:
        with gzip.open(target_path, "rb") as f:
            tgt = pickle.load(f)
    except FileNotFoundError:
        return None, None
    states = tgt["agent_states"]; labels = tgt["agent_labels"]
    if hasattr(states, "numpy"): states = states.numpy()
    if hasattr(labels, "numpy"): labels = labels.numpy()
    valid = labels.astype(bool)
    if not valid.any(): return None, None
    xy = states[valid, :2]
    front = xy[:, 0] > 0.0
    if not front.any(): return None, None
    fxy = xy[front]
    nearest = fxy[np.argmin(np.linalg.norm(fxy, axis=1))]
    return float(nearest[0]), float(nearest[1])


def project_to_image(x_ego, y_ego):
    """Rough ego-frame → pixel projection for the front camera."""
    if x_ego <= 0.1: return None
    u = int(CAM_FX * (-y_ego) / x_ego + CAM_CX)
    v = int(CAM_FY * CAM_H    / x_ego + CAM_CY)
    return (u, v)


def annotate(img, true_dist, pred_grouped, pred_leaky, ax, ay):
    img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (IMG_W, 36)], fill=(0, 0, 0))
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSansMono.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    err_g = pred_grouped - true_dist
    err_l = pred_leaky   - true_dist
    gcol = (80, 200, 80) if abs(err_g) < 2.0 else (220, 80, 80)
    lcol = (200, 160, 60)

    draw.text((6,  2), f"GT: {true_dist:.1f} m", fill=(255, 255, 100), font=font)
    draw.text((6, 18), f"Grouped: {pred_grouped:.1f} m  (err {err_g:+.1f} m)", fill=gcol,  font=font)
    draw.text((340, 18), f"Leaky: {pred_leaky:.1f} m  (err {err_l:+.1f} m)", fill=lcol, font=font)

    if ax is not None:
        uv = project_to_image(ax, ay)
        if uv:
            u = int(uv[0] * IMG_W / 1920)
            v = int(uv[1] * IMG_H / 270)
            r = 9
            draw.line([(u-r, v), (u+r, v)], fill=(255, 50, 50), width=2)
            draw.line([(u, v-r), (u, v+r)], fill=(255, 50, 50), width=2)
            draw.ellipse([(u-r, v-r), (u+r, v+r)], outline=(255, 50, 50), width=2)
    return img


def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------------

def select_examples(samples, layer, n_per_bin):
    key_g = f"pred_grouped_{layer}"
    key_l = f"pred_leaky_{layer}"
    results = []
    for bname in BIN_NAMES:
        bin_s = [s for s in samples if s["bin"] == bname
                 and key_g in s and key_l in s]
        if not bin_s:
            continue
        abs_err = np.array([abs(s[key_g] - s["true_dist"]) for s in bin_s])
        order   = np.argsort(abs_err)

        for idxs, cat in [(order[:n_per_bin], "best"), (order[-n_per_bin:][::-1], "worst")]:
            for i in idxs:
                s = bin_s[i]
                results.append({**s, "category": cat,
                                 "pred_grouped": s[key_g], "pred_leaky": s[key_l],
                                 "abs_err": float(abs_err[i])})

        # Traffic: 2nd-nearest < threshold, sorted by worst grouped error
        traffic = [s for s in bin_s
                   if s.get("dist_2nd") is not None and s["dist_2nd"] < TRAFFIC_THRESH]
        if traffic:
            t_err   = np.array([abs(s[key_g] - s["true_dist"]) for s in traffic])
            t_order = np.argsort(t_err)[::-1]
            for i in t_order[:n_per_bin]:
                s = traffic[i]
                results.append({**s, "category": "traffic",
                                 "pred_grouped": s[key_g], "pred_leaky": s[key_l],
                                 "abs_err": float(t_err[i])})
    return results


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CAT_STYLE = {
    "best":    ("Best prediction",  "cat-best"),
    "worst":   ("Worst prediction", "cat-worst"),
    "traffic": ("Traffic scene",    "cat-traffic"),
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LTFv6 Probe Qualitative Visualization</title>
<style>
  :root {{ --bg:#0f1117; --panel:#1a1d27; --border:#2a2e3a; --text:#e6e6e6;
          --muted:#9aa0ad; --accent:#5fb3ff; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
         background:var(--bg);color:var(--text);margin:0;padding:24px 32px 80px;font-size:14px; }}
  h1 {{ font-size:22px;margin-bottom:4px; }}
  p.sub {{ color:var(--muted);margin-top:0; }}
  p.note {{ color:var(--muted);font-size:12.5px;max-width:900px;margin:4px 0 14px; }}
  .findings {{ background:var(--panel);border:1px solid var(--border);border-radius:8px;
               padding:14px 18px;margin:18px 0; }}
  .findings li {{ margin-bottom:5px; }}
  code {{ background:#222536;padding:1px 5px;border-radius:3px; }}
  .tabs {{ display:flex;flex-wrap:wrap;gap:8px;margin:14px 0; }}
  .tab-btn {{ background:var(--panel);border:1px solid var(--border);color:var(--text);
              padding:7px 14px;border-radius:6px;cursor:pointer;font-size:13px; }}
  .tab-btn.active {{ background:var(--accent);color:#08110a;font-weight:600; }}
  .tabpanel {{ display:none; }}
  .tabpanel.active {{ display:block; }}
  .grid {{ display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));margin-top:10px; }}
  .card {{ background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden; }}
  .card img {{ width:100%;display:block; }}
  .card-body {{ padding:8px 10px; }}
  .card-body p {{ margin:2px 0;font-size:12px; }}
  .cat-label {{ display:inline-block;font-size:11px;font-weight:600;padding:2px 7px;
                border-radius:4px;margin-bottom:5px; }}
  .cat-best    {{ background:#1c3a28;color:#5dba6e; }}
  .cat-worst   {{ background:#3a1c1c;color:#e0695e; }}
  .cat-traffic {{ background:#2b2a10;color:#d4c846; }}
  .err-good {{ color:#5dba6e; }}
  .err-bad  {{ color:#e0695e; }}
  .err-leaky {{ color:#c8a060; }}
  footer {{ color:var(--muted);font-size:12px;margin-top:50px; }}
</style>
</head>
<body>
<h1>LTFv6 — Probe Qualitative Visualization</h1>
<p class="sub">Layer: <code>{layer}</code> &nbsp; Out-of-fold predictions (GroupKFold by log = honest).</p>

<div class="findings">
  <strong>How to read this</strong>
  <ul>
    <li>Front camera image (1920&times;270, shown at half size).
      <span style="color:#ff5555">&#9711;</span> = approximate projected position of the nearest front vehicle
      (rough projection; not pixel-accurate).</li>
    <li><strong style="color:#ffff64">GT</strong> = ground-truth distance.
      <strong style="color:#5dba6e">Grouped pred</strong> = out-of-fold prediction under GroupKFold
      (entire logs held out — honest, no leakage).
      <span style="color:#5dba6e">Green</span> = |error| &lt; 2 m, <span style="color:#e0695e">red</span> = larger.</li>
    <li><strong style="color:#c8a060">Leaky pred</strong> = out-of-fold prediction under plain KFold
      (frames from same drive in both train/test). Compare to see how much leakage inflates accuracy.</li>
    <li><strong>2nd-nearest (any dir)</strong> = distance to the 2nd closest valid agent in <em>any</em> direction (including behind/beside ego).
      This can be smaller than GT because GT is nearest-front only (forward half), while 2nd-nearest counts all directions.</li>
    <li><strong>Traffic</strong> tab = scenes where 2nd-nearest (any dir) &lt; {traffic_thresh} m.
      These test whether the probe confuses multiple close vehicles.</li>
  </ul>
</div>

<!-- Bin-level tabs -->
<div class="tabs" id="bin-tabs">{bin_tab_buttons}</div>
{bin_panels}

<footer>Predictions from <code>probe_predictions_ltfv6.json</code> &nbsp;|&nbsp;
  layer: {layer} &nbsp;|&nbsp; n_valid={n_valid}</footer>

<script>
document.querySelectorAll(".tab-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    const g = btn.dataset.group;
    document.querySelectorAll(".tab-btn[data-group='" + g + "']").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tabpanel[data-group='" + g + "']").forEach(p => p.classList.remove("active"));
    document.getElementById(btn.dataset.target).classList.add("active");
  }});
}});
</script>
</body>
</html>
"""


def build_card(ex):
    img = load_image_from_feature(ex["path"])
    ax, ay = load_nearest_front_agent(ex["path"])
    img_ann = annotate(img, ex["true_dist"], ex["pred_grouped"], ex["pred_leaky"], ax, ay)
    b64  = img_to_b64(img_ann)
    err  = ex["pred_grouped"] - ex["true_dist"]
    ecls = "err-good" if abs(err) < 2.0 else "err-bad"
    lbl, lcls = CAT_STYLE[ex["category"]]
    d2 = f"  2nd-nearest (any dir): {ex['dist_2nd']:.1f} m" if ex.get("dist_2nd") is not None else ""
    return (
        f'<div class="card">'
        f'<img src="data:image/jpeg;base64,{b64}" alt="scene">'
        f'<div class="card-body">'
        f'<span class="cat-label {lcls}">{lbl}</span>'
        f'<p>GT: <strong>{ex["true_dist"]:.2f} m</strong> &nbsp; Bin: {ex["bin"]}{d2}</p>'
        f'<p class="{ecls}">Grouped pred: {ex["pred_grouped"]:.2f} m &nbsp; (err {err:+.2f} m)</p>'
        f'<p class="err-leaky">Leaky pred: {ex["pred_leaky"]:.2f} m &nbsp; (err {ex["pred_leaky"]-ex["true_dist"]:+.2f} m)</p>'
        f'</div></div>'
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default=PRED_DEFAULT)
    p.add_argument("--layer",       default="backbone_transformers_2")
    p.add_argument("--n_per_bin",   type=int, default=3)
    p.add_argument("--out",         default=OUT_DEFAULT)
    args = p.parse_args()

    with open(args.predictions) as f:
        data = json.load(f)

    layer = args.layer
    if f"pred_grouped_{layer}" not in data["samples"][0]:
        available = [k.replace("pred_grouped_", "") for k in data["samples"][0] if k.startswith("pred_grouped_")]
        print(f"Layer '{layer}' not in predictions. Available: {available}")
        layer = available[0]
        print(f"Using {layer}")

    examples = select_examples(data["samples"], layer, args.n_per_bin)
    bins_present = [b for b in BIN_NAMES if any(e["bin"] == b for e in examples)]
    cats = ["best", "worst", "traffic"]

    bin_tab_buttons = ""
    bin_panels = ""
    for bi, bname in enumerate(bins_present):
        safe = bname.replace("+", "plus").replace("-", "_")
        active = " active" if bi == 0 else ""
        bin_tab_buttons += (
            f'<button class="tab-btn{active}" data-group="bin" data-target="bin-{safe}">{bname}</button>\n'
        )

        cat_tabs = '<div class="tabs">'
        cat_panels = ""
        bin_exs = [e for e in examples if e["bin"] == bname]
        cats_here = [c for c in cats if any(e["category"] == c for e in bin_exs)]
        for ci, cat in enumerate(cats_here):
            cat_exs = [e for e in bin_exs if e["category"] == cat]
            csafe = f"{safe}_{cat}"
            cactive = " active" if ci == 0 else ""
            lbl = CAT_STYLE[cat][0]
            cat_tabs += (
                f'<button class="tab-btn{cactive}" data-group="{safe}" data-target="{csafe}">'
                f'{lbl} ({len(cat_exs)})</button>\n'
            )
            print(f"  Rendering {len(cat_exs)} images  bin={bname}  cat={cat}...")
            cards = "".join(build_card(e) for e in cat_exs)
            cpactive = " active" if ci == 0 else ""
            cat_panels += (
                f'<div class="tabpanel{cpactive}" id="{csafe}" data-group="{safe}">'
                f'<div class="grid">{cards}</div></div>\n'
            )
        cat_tabs += "</div>"
        bpactive = " active" if bi == 0 else ""
        bin_panels += (
            f'<div class="tabpanel{bpactive}" id="bin-{safe}" data-group="bin">'
            f'{cat_tabs}{cat_panels}</div>\n'
        )

    html = HTML_TEMPLATE.format(
        layer=layer,
        n_valid=data["n"],
        traffic_thresh=int(TRAFFIC_THRESH),
        bin_tab_buttons=bin_tab_buttons,
        bin_panels=bin_panels,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
