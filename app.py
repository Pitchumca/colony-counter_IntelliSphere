"""
Automated Bacterial Colony Counter
AVIT Faculty Hackathon 2026 — Project 07
Streamlit web application
"""

import io
import sys

import streamlit as st  # must be imported before matplotlib on Streamlit Cloud

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2
except ModuleNotFoundError:
    st.error(
        "**OpenCV not found.** "
        "Make sure `packages.txt` (with `libgl1`) and `requirements.txt` "
        "(with `opencv-python-headless==4.9.0.80`) are both committed to your repo, "
        "then **Reboot app** from the Streamlit Cloud dashboard."
    )
    st.stop()

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Colony Counter | AVIT",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0f1923;
    border-right: 1px solid #1e3a4a;
  }
  [data-testid="stSidebar"] * { color: #c9d8e0 !important; }
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stSelectbox label { color: #7fb3c8 !important; font-size: 0.78rem; letter-spacing: .04em; text-transform: uppercase; }

  /* Main background */
  .stApp { background: #f0f4f7; }

  /* Hero banner */
  .hero {
    background: linear-gradient(135deg, #0d2233 0%, #0a3d55 60%, #0f5e7a 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1.2rem;
  }
  .hero-icon { font-size: 3.2rem; line-height: 1; }
  .hero h1 { color: #e0f3ff; font-size: 1.65rem; font-weight: 700; margin: 0; letter-spacing: -0.02em; }
  .hero p  { color: #7fb3c8; font-size: 0.85rem; margin: 0.25rem 0 0; }

  /* Metric cards */
  .metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
  .metric-card {
    flex: 1; min-width: 130px;
    background: white;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border-top: 3px solid #0a8abf;
    box-shadow: 0 1px 6px rgba(0,0,0,.07);
  }
  .metric-card.warn  { border-top-color: #e07b2a; }
  .metric-card.good  { border-top-color: #28a87d; }
  .metric-card.info  { border-top-color: #7c4dde; }
  .metric-card .val  { font-size: 2rem; font-weight: 700; color: #0d2233; font-family: 'JetBrains Mono', monospace; }
  .metric-card .lbl  { font-size: 0.72rem; color: #6b8a9a; text-transform: uppercase; letter-spacing: .05em; margin-top: .2rem; }

  /* Decision badge */
  .badge-countable     { background:#d1f5e8; color:#117a57; padding:.25rem .7rem; border-radius:20px; font-size:.8rem; font-weight:600; }
  .badge-not-countable { background:#fde8d0; color:#a04010; padding:.25rem .7rem; border-radius:20px; font-size:.8rem; font-weight:600; }

  /* Upload zone */
  [data-testid="stFileUploader"] { border-radius: 10px; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; border-bottom: 2px solid #d0dde5; }
  .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; font-size: .85rem; font-weight: 500; padding: .45rem 1rem; }

  /* Divider text */
  .section-head { font-size: .72rem; color: #7fb3c8; text-transform: uppercase; letter-spacing: .1em; margin: 1.2rem 0 .5rem; font-weight: 600; }

  /* Footer */
  .footer { text-align:center; color:#9ab3be; font-size:.75rem; margin-top:2rem; padding-top:1rem; border-top:1px solid #d0dde5; }
</style>
""", unsafe_allow_html=True)


# ── CV helper functions ─────────────────────────────────────────────────────────

def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    """PIL → numpy RGB array, resized to max_width."""
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    max_w = 900
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def detect_petri_dish(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 9)
    h, w = gray.shape

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min(h, w) // 2,
        param1=80, param2=35,
        minRadius=int(min(h, w) * 0.25),
        maxRadius=int(min(h, w) * 0.52),
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx, cy, r = circles[0][0]
    else:
        cx, cy = w // 2, h // 2
        r = int(min(h, w) * 0.42)

    inner_r = int(r * 0.78)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), inner_r, 255, -1)
    return int(cx), int(cy), inner_r, mask


def illumination_correct(image: np.ndarray, plate_mask: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masked  = cv2.bitwise_and(gray, gray, mask=plate_mask)
    bg      = cv2.medianBlur(masked, 51)
    corr    = cv2.subtract(masked, bg)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enh     = clahe.apply(corr)
    return cv2.bitwise_and(enh, enh, mask=plate_mask)


def baseline_threshold_counter(image, min_area=20, max_area=2000):
    output = image.copy()
    cx, cy, pr, plate_mask = detect_petri_dish(image)
    enh  = illumination_correct(image, plate_mask)
    blur = cv2.GaussianBlur(enh, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)
    kernel = np.ones((3, 3), np.uint8)
    clean  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(clean)
    count = 0
    cv2.circle(output, (cx, cy), pr, (255, 220, 0), 2)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            x, y = centroids[i]
            count += 1
            cv2.circle(output, (int(x), int(y)), 7, (255, 80, 0), 2)
    return count, output, binary


def watershed_counter(image, min_area=20, max_area=2000, min_distance=10):
    output = image.copy()
    cx, cy, pr, plate_mask = detect_petri_dish(image)
    enh  = illumination_correct(image, plate_mask)
    blur = cv2.GaussianBlur(enh, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)
    kernel  = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    coords   = peak_local_max(distance, min_distance=min_distance,
                               threshold_abs=2, labels=opening)
    markers  = np.zeros(distance.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx
    markers   = ndi.label(markers > 0)[0]
    label_map = watershed(-distance, markers, mask=opening)
    count = 0
    cv2.circle(output, (cx, cy), pr, (255, 220, 0), 2)
    for label in np.unique(label_map):
        if label == 0:
            continue
        comp = np.zeros(enh.shape, dtype=np.uint8)
        comp[label_map == label] = 255
        area = cv2.countNonZero(comp)
        if min_area <= area <= max_area:
            M = cv2.moments(comp)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if plate_mask[cY, cX] == 255:
                    count += 1
                    cv2.circle(output, (cX, cY), 8, (0, 210, 100), 2)
    return count, output, binary


def extract_plate_features(image):
    cx, cy, pr, plate_mask = detect_petri_dish(image)
    enh  = illumination_correct(image, plate_mask)
    blur = cv2.GaussianBlur(enh, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)
    plate_area = max(int(np.sum(plate_mask > 0)), 1)
    fg_area    = int(np.sum(binary > 0))
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, len(stats))
             if stats[i, cv2.CC_STAT_AREA] > 10]
    if not areas:
        areas = [0]
    areas = np.array(areas)
    edges = cv2.Canny(enh, 50, 150)
    return {
        "coverage_ratio"      : float(fg_area / plate_area),
        "component_count"     : int(len(areas)),
        "mean_component_area" : float(np.mean(areas)),
        "std_component_area"  : float(np.std(areas)),
        "max_component_area"  : float(np.max(areas)),
        "edge_density"        : float(np.sum(edges > 0) / plate_area),
        "brightness_mean"     : float(np.mean(enh[plate_mask > 0])),
        "brightness_std"      : float(np.std(enh[plate_mask > 0])),
    }


def decide_countability(features):
    if features["coverage_ratio"]    > 0.45:   return "not_countable"
    if features["max_component_area"] > 10000:  return "not_countable"
    if features["component_count"]   < 3:       return "not_countable"
    if features["component_count"]   > 400:     return "not_countable"
    return "countable"


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    return buf.read()


# ── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown('<p class="section-head">Colony size filter (px²)</p>', unsafe_allow_html=True)
    min_area = st.slider("Min area", 5, 100, 20, 5)
    max_area = st.slider("Max area", 500, 5000, 2000, 100)

    st.markdown('<p class="section-head">Watershed spacing</p>', unsafe_allow_html=True)
    min_dist = st.slider("Min distance between centres (px)", 5, 30, 10, 1)

    st.markdown('<p class="section-head">CFU estimation</p>', unsafe_allow_html=True)
    dilution   = st.number_input("Dilution factor (e.g. 1e5)", value=1e5, format="%e")
    vol_ml     = st.number_input("Volume plated (mL)", value=0.1, step=0.01, format="%.3f")

    st.markdown("---")
    st.markdown("**AVIT Faculty Hackathon 2026**  \nProject 07 — CSE × Biotechnology")
    st.markdown("*Streamlit demo — no data leaves your browser*")


# ── Hero ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-icon">🧫</div>
  <div>
    <h1>Automated Bacterial Colony Counter</h1>
    <p>Upload a petri-dish photo → get an instant colony count, countability decision, and CFU estimate</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ── File uploader ───────────────────────────────────────────────────────────────

col_up, col_tip = st.columns([2, 1])
with col_up:
    uploaded_files = st.file_uploader(
        "Upload one or more petri-dish images",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
with col_tip:
    st.info("**Tip:** For best results, photograph plates on a plain white background with even lighting.")


if not uploaded_files:
    st.markdown("### 👆 Upload plate images to get started")
    st.markdown("""
    The app will automatically:
    - Detect and isolate the inner agar area
    - Count colonies using **baseline** (simple thresholding) and **improved** (watershed) methods
    - Flag overcrowded or contaminated plates for manual review
    - Estimate **CFU/mL** using your dilution factor
    """)
    st.stop()


# ── Process each image ──────────────────────────────────────────────────────────

all_results = []

progress = st.progress(0, text="Processing plates…")

for file_idx, uploaded_file in enumerate(uploaded_files):
    pil_img = Image.open(uploaded_file)
    img     = pil_to_cv(pil_img)

    feats    = extract_plate_features(img)
    decision = decide_countability(feats)

    b_count, b_out, b_bin  = baseline_threshold_counter(img, min_area, max_area)
    w_count, w_out, w_bin  = watershed_counter(img, min_area, max_area, min_dist)

    cfu = (w_count * dilution / vol_ml) if decision == "countable" else None

    all_results.append({
        "filename"            : uploaded_file.name,
        "decision"            : decision,
        "baseline_count"      : b_count,
        "watershed_count"     : w_count,
        "estimated_CFU_per_mL": cfu,
        "coverage_ratio"      : round(feats["coverage_ratio"], 3),
        "component_count"     : feats["component_count"],
        "max_component_area"  : round(feats["max_component_area"], 1),
        "_img"                : img,
        "_b_out"              : b_out,
        "_w_out"              : w_out,
        "_b_bin"              : b_bin,
    })

    progress.progress((file_idx + 1) / len(uploaded_files),
                       text=f"Processed {file_idx+1}/{len(uploaded_files)} plates")

progress.empty()


# ── Summary row ─────────────────────────────────────────────────────────────────

n_total      = len(all_results)
n_countable  = sum(1 for r in all_results if r["decision"] == "countable")
n_review     = n_total - n_countable
avg_wc       = np.mean([r["watershed_count"] for r in all_results if r["decision"] == "countable"]) if n_countable else 0

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card info">
    <div class="val">{n_total}</div>
    <div class="lbl">Plates uploaded</div>
  </div>
  <div class="metric-card good">
    <div class="val">{n_countable}</div>
    <div class="lbl">Countable</div>
  </div>
  <div class="metric-card warn">
    <div class="val">{n_review}</div>
    <div class="lbl">Manual review</div>
  </div>
  <div class="metric-card">
    <div class="val">{avg_wc:.0f}</div>
    <div class="lbl">Avg colony count</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Per-image detail tabs ────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 🔬 Per-Plate Results")

for r in all_results:
    badge = (
        '<span class="badge-countable">✅ Countable</span>'
        if r["decision"] == "countable"
        else '<span class="badge-not-countable">⚠️ Manual Review</span>'
    )
    with st.expander(f"**{r['filename']}**  {badge}", expanded=(len(all_results) == 1)):
        st.markdown(badge, unsafe_allow_html=True)

        # Metric strip
        cfu_str = f"{r['estimated_CFU_per_mL']:.2e}" if r["estimated_CFU_per_mL"] else "—"
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card"><div class="val">{r['baseline_count']}</div><div class="lbl">Baseline count</div></div>
          <div class="metric-card good"><div class="val">{r['watershed_count']}</div><div class="lbl">Watershed count</div></div>
          <div class="metric-card info"><div class="val">{cfu_str}</div><div class="lbl">Est. CFU/mL</div></div>
          <div class="metric-card"><div class="val">{r['coverage_ratio']:.1%}</div><div class="lbl">Plate coverage</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Image panels
        tab1, tab2, tab3 = st.tabs(["📷 Original", "🔵 Baseline", "🟢 Watershed"])

        with tab1:
            st.image(r["_img"], use_container_width=True)
        with tab2:
            c1, c2 = st.columns(2)
            c1.image(r["_b_bin"], caption="Binary mask", use_container_width=True, clamp=True)
            c2.image(r["_b_out"], caption=f"Detected: {r['baseline_count']} colonies",
                     use_container_width=True)
        with tab3:
            c1, c2 = st.columns(2)
            c1.image(r["_w_bin"], caption="Binary mask", use_container_width=True, clamp=True)
            c2.image(r["_w_out"], caption=f"Detected: {r['watershed_count']} colonies",
                     use_container_width=True)

        # Feature breakdown
        with st.expander("📊 Plate feature breakdown", expanded=False):
            feat_df = pd.DataFrame([{
                "Feature": k, "Value": v
            } for k, v in {
                "Coverage ratio"       : f"{r['coverage_ratio']:.3f}",
                "Component count"      : r["component_count"],
                "Max component area"   : r["max_component_area"],
                "Countability decision": r["decision"],
            }.items()])
            st.dataframe(feat_df, hide_index=True, use_container_width=True)


# ── Batch results table ──────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 📋 Batch Results Table")

display_df = pd.DataFrame([{
    "Filename"        : r["filename"],
    "Decision"        : r["decision"],
    "Baseline Count"  : r["baseline_count"],
    "Watershed Count" : r["watershed_count"],
    "Est. CFU/mL"     : f"{r['estimated_CFU_per_mL']:.2e}" if r["estimated_CFU_per_mL"] else "N/A",
    "Coverage"        : f"{r['coverage_ratio']:.1%}",
} for r in all_results])

st.dataframe(display_df, hide_index=True, use_container_width=True)

csv_bytes = display_df.to_csv(index=False).encode()
st.download_button(
    "⬇️ Download results CSV",
    data=csv_bytes,
    file_name="colony_counting_results.csv",
    mime="text/csv",
)

# ── Footer ───────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
  AVIT Faculty Hackathon 2026 · Project 07 · Automated Bacterial Colony Counter<br>
  CSE Team (CV + ML) supporting Biotechnology · Built with OpenCV, scikit-image, Streamlit
</div>
""", unsafe_allow_html=True)
