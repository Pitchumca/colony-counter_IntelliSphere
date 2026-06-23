# 🧫 Automated Bacterial Colony Counter

**AVIT Faculty Hackathon 2026 — Project 07**  
CSE × Biotechnology Interdisciplinary Team

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## What it does

Upload a petri-dish photograph and the app instantly:

- Detects and isolates the inner agar area using Hough Circle transform
- Counts colonies via two methods:
  - **Baseline** — Otsu threshold + connected component labelling
  - **Improved** — marker-controlled watershed (separates touching colonies)
- Flags overcrowded or contaminated plates for **manual review**
- Estimates **CFU/mL** using your dilution factor and plated volume
- Lets you download the full results as a CSV

---

## Run locally

```bash
# 1. Clone
git clone https://github.com/<your-username>/colony-counter.git
cd colony-counter

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Deploy to Streamlit Community Cloud (free)

1. **Push this repo to GitHub** (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repository, branch `main`, and set **Main file path** to `app.py`
4. Click **Deploy** — live in ~2 minutes

---

## Project structure

```
colony-counter/
├── app.py              ← Streamlit application (all logic here)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## Parameters (sidebar)

| Parameter | Default | Description |
|---|---|---|
| Min colony area | 20 px² | Smallest object counted as a colony |
| Max colony area | 2000 px² | Largest object counted as a colony |
| Min distance | 10 px | Minimum centre-to-centre distance (watershed) |
| Dilution factor | 1×10⁵ | Experimental dilution (e.g. 10⁻⁵ = 1e5) |
| Volume plated | 0.1 mL | Volume spread per plate |

---

## Methods

### Baseline — Simple Thresholding + CCL
Otsu global threshold → morphological opening → connected component labelling with area filter.  
**Limitation:** merges touching colonies into one blob.

### Improved — Marker-Controlled Watershed
Same preprocessing → distance transform → `peak_local_max` seeds one marker per colony → watershed on the negative distance map.  
**Advantage:** separates touching colonies that the baseline merges.

### Countability Classifier (rule-based)
A plate is flagged `not_countable` (manual review) if:
- Coverage ratio > 45 % (likely confluent)
- Largest component > 10 000 px² (smear / contamination)
- Fewer than 3 components (empty plate)
- More than 400 components (TNTC)

---

## Dependencies

`opencv-python-headless · scikit-image · scikit-learn · scipy · numpy · pandas · matplotlib · Pillow · streamlit`

---

## Team

**Biotechnology** — sample preparation, manual counts, lab coordination  
**CSE (Data Science)** — CV pipeline, ML classifier, web app  
Aarupadai Veedu Institute of Technology (AVIT), VMRF Deemed University
