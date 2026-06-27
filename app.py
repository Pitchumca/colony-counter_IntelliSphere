import os
import re
import glob
import zipfile
import tempfile
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log
from skimage.segmentation import watershed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Automated Bacterial Colony Counter",
    layout="wide",
    page_icon="🧫"
)

st.title("🧫 Automated Bacterial Colony Counter - AVIT FACULTY HACKATHON 2026 BY iTECH PVT. LTD.")
st.caption("Classical Computer Vision Colony Counter + Countable/Manual-Review Random Forest Classifier")

st.markdown("""
### System Design
- **Counting:** deterministic classical computer vision only
- **Core CV:** petri masking, reflection/background removal, CLAHE, thresholding, marker-controlled watershed
- **Post-processing:** area, circularity, solidity, aspect-ratio and duplicate filtering
- **Validation:** count accuracy calculated only on countable plates: **25–250 colonies**
- **ML:** Random Forest only for plate-level countable/not-countable decision
- **Large data:** upload images, upload ZIP, or process a large ZIP from local/Colab path
""")

# ============================================================
# CONSTANTS
# ============================================================

VALID_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
VALID_ZIP_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

IMAGE_EXTENSIONS = [
    "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff",
    "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF"
]

FEATURE_COLUMNS = [
    "coverage_ratio",
    "component_count",
    "mean_component_area",
    "std_component_area",
    "max_component_area",
    "brightness_mean",
    "brightness_std",
    "edge_density",
    "auto_count",
    "candidate_count",
    "accepted_fraction",
]

# ============================================================
# FILE HELPERS
# ============================================================

def find_all_images(folder):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(list(set(paths)))


@st.cache_data(show_spinner=False)
def read_image_cached(path, max_width=800):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return img


def read_image(path, max_width=800):
    return read_image_cached(path, max_width)


def read_uploaded_image(uploaded_file, max_width=800):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Cannot read uploaded image: {uploaded_file.name}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return img


def list_zip_images(zip_path, max_images):
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [
            n for n in z.namelist()
            if not n.endswith("/") and n.lower().endswith(VALID_ZIP_IMAGE_EXT)
        ]

    items = []
    for name in sorted(names)[:max_images]:
        items.append({
            "name": os.path.basename(name),
            "zip_path": zip_path,
            "zip_member": name,
            "type": "zip_member",
        })
    return items


def read_zip_image(zip_path, member_name, max_width=800):
    with zipfile.ZipFile(zip_path, "r") as z:
        data = z.read(member_name)

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Cannot read image inside ZIP: {member_name}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return img


def load_item_image(item):
    if item["type"] == "array":
        return item["image"]
    if item["type"] == "path":
        return read_image(item["path"])
    if item["type"] == "zip_member":
        return read_zip_image(item["zip_path"], item["zip_member"])
    raise ValueError("Unsupported image item type.")


def normalize_filename(filename):
    return os.path.basename(str(filename)).strip()


def extract_manual_count_from_filename(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def get_manual_count(filename, lab_df=None):
    filename_norm = normalize_filename(filename)

    if lab_df is not None and "filename" in lab_df.columns and "manual_count" in lab_df.columns:
        temp = lab_df.copy()
        temp["filename_norm"] = temp["filename"].apply(normalize_filename)
        matched = temp[temp["filename_norm"] == filename_norm]
        if len(matched) > 0:
            value = matched.iloc[0]["manual_count"]
            if pd.notna(value):
                return int(value)

    return extract_manual_count_from_filename(filename_norm)


def microbiology_countability_from_manual(manual_count):
    if manual_count is None or pd.isna(manual_count):
        return None
    manual_count = int(manual_count)
    return "countable" if 25 <= manual_count <= 250 else "not_countable"


def get_lab_label(filename, lab_df=None, manual_count=None):
    filename_norm = normalize_filename(filename)

    if lab_df is not None and "filename" in lab_df.columns and "countability_label" in lab_df.columns:
        temp = lab_df.copy()
        temp["filename_norm"] = temp["filename"].apply(normalize_filename)
        matched = temp[temp["filename_norm"] == filename_norm]

        if len(matched) > 0:
            label = str(matched.iloc[0]["countability_label"]).strip().lower()
            label = label.replace(" ", "_").replace("-", "_")

            if label in ["countable", "not_countable"]:
                return label
            if label in ["manual_review", "uncountable", "tntc", "tftc"]:
                return "not_countable"

    return microbiology_countability_from_manual(manual_count)


def safe_percentage_error(manual, auto):
    if manual is None or pd.isna(manual) or manual == 0:
        return None
    return abs(manual - auto) / manual * 100


def safe_r2(y_true, y_pred):
    try:
        if len(y_true) < 2:
            return None
        return r2_score(y_true, y_pred)
    except Exception:
        return None


def countable_validation_df(df):
    if len(df) == 0:
        return df
    return df[
        (df["manual_count"].notna()) &
        (df["automatic_count"].notna()) &
        (df["manual_count"] >= 25) &
        (df["manual_count"] <= 250)
    ].copy()


def all_manual_df(df):
    if len(df) == 0:
        return df
    return df[(df["manual_count"].notna()) & (df["automatic_count"].notna())].copy()

# ============================================================
# ROBUST CLASSICAL CV PIPELINE
# ============================================================


def detect_petri_dish(image):
    """
    Robust petri dish detection.

    Priority:
    1. Use Hough circle detection on the plate rim.
    2. If Hough fails, use largest bright circular contour.
    3. If both fail, use conservative center fallback.

    This fixes the wrong shifted/yellow circle issue caused by using only
    a fixed centered circle.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Optional fast centered fallback only if user explicitly enables it.
    if st.session_state.get("fast_centered_mask", False):
        x, y = w // 2, h // 2
        r = int(min(h, w) * st.session_state.get("mask_radius_factor", 0.43))
        inner_r = int(r * st.session_state.get("inner_mask_factor", 0.82))
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (x, y), inner_r, 255, -1)
        return x, y, inner_r, mask

    # Enhance rim contrast.
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    eq = cv2.equalizeHist(blur)

    min_r = int(min(h, w) * 0.28)
    max_r = int(min(h, w) * 0.52)

    circles = cv2.HoughCircles(
        eq,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(h, w) * 0.45),
        param1=80,
        param2=28,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Choose the largest detected circle close to the image center.
        candidates = []
        cx0, cy0 = w / 2, h / 2
        for c in circles[0, :]:
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            center_dist = np.sqrt((x - cx0) ** 2 + (y - cy0) ** 2)
            score = r - 0.15 * center_dist
            candidates.append((score, x, y, r))
        _, x, y, r = max(candidates, key=lambda v: v[0])

        inner_r = int(r * st.session_state.get("inner_mask_factor", 0.88))
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (x, y), inner_r, 255, -1)
        return int(x), int(y), int(inner_r), mask

    # Fallback: find largest circular/rim contour.
    edges = cv2.Canny(eq, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (min(h, w) ** 2) * 0.05:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        r = float(r)

        if r < min_r or r > max_r:
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = 0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < 0.25:
            continue

        center_dist = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
        score = area * circularity - center_dist * 10

        if score > best_score:
            best_score = score
            best = (int(x), int(y), int(r))

    if best is not None:
        x, y, r = best
        inner_r = int(r * st.session_state.get("inner_mask_factor", 0.88))
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (x, y), inner_r, 255, -1)
        return int(x), int(y), int(inner_r), mask

    # Last fallback.
    x, y = w // 2, h // 2
    r = int(min(h, w) * st.session_state.get("mask_radius_factor", 0.43))
    inner_r = int(r * st.session_state.get("inner_mask_factor", 0.88))
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (x, y), inner_r, 255, -1)
    return int(x), int(y), int(inner_r), mask


def remove_reflections_and_background(gray, plate_mask):
    masked = cv2.bitwise_and(gray, gray, mask=plate_mask)

    # Median background subtraction removes slow illumination variations and plastic glare.
    bg_kernel = int(st.session_state.get("background_kernel", 51))
    if bg_kernel % 2 == 0:
        bg_kernel += 1

    bg_kernel = max(31, bg_kernel)
    background = cv2.medianBlur(masked, bg_kernel)
    corrected = cv2.subtract(masked, background)

    # Top-hat enhances small bright colonies over uneven agar.
    top_kernel = int(st.session_state.get("tophat_kernel", 15))
    if top_kernel % 2 == 0:
        top_kernel += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (top_kernel, top_kernel))
    tophat = cv2.morphologyEx(masked, cv2.MORPH_TOPHAT, kernel)

    combined = cv2.addWeighted(corrected, 0.65, tophat, 0.35, 0)
    combined = cv2.bitwise_and(combined, combined, mask=plate_mask)

    return combined


def enhance_image(image, plate_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    corrected = remove_reflections_and_background(gray, plate_mask)

    clahe = cv2.createCLAHE(
        clipLimit=float(st.session_state.get("clahe_clip", 2.5)),
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(corrected)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=plate_mask)

    return enhanced


def threshold_colonies(enhanced, plate_mask):
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    mode = st.session_state.get("threshold_mode", "Otsu")

    if mode == "Adaptive":
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int(st.session_state.get("adaptive_block", 41)) | 1,
            float(st.session_state.get("adaptive_c", -3)),
        )
    else:
        _, binary = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    if st.session_state.get("invert_binary", False):
        binary = cv2.bitwise_not(binary)

    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return binary


def contour_features(component_mask):
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "area": 0,
            "perimeter": 0,
            "circularity": 0,
            "aspect_ratio": 0,
            "solidity": 0,
            "extent": 0,
            "cx": None,
            "cy": None,
            "radius": 0,
        }

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))

    circularity = 0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 0
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0

    moments = cv2.moments(cnt)
    cx = None
    cy = None
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

    (_, _), radius = cv2.minEnclosingCircle(cnt)

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "extent": extent,
        "cx": cx,
        "cy": cy,
        "radius": float(radius),
    }


def is_valid_colony(feat, min_area, max_area):
    if feat["cx"] is None or feat["cy"] is None:
        return False

    if feat["area"] < min_area or feat["area"] > max_area:
        return False

    if feat["circularity"] < float(st.session_state.get("min_circularity", 0.25)):
        return False

    if feat["solidity"] < float(st.session_state.get("min_solidity", 0.45)):
        return False

    if feat["extent"] < float(st.session_state.get("min_extent", 0.20)):
        return False

    if feat["aspect_ratio"] < float(st.session_state.get("min_aspect", 0.25)):
        return False

    if feat["aspect_ratio"] > float(st.session_state.get("max_aspect", 4.0)):
        return False

    return True


def suppress_duplicates(points, min_dist=6):
    accepted = []

    for p in points:
        x, y, r = p

        duplicate = False
        for ax, ay, ar in accepted:
            d = np.sqrt((x - ax) ** 2 + (y - ay) ** 2)
            if d < max(min_dist, 0.6 * (r + ar)):
                duplicate = True
                break

        if not duplicate:
            accepted.append(p)

    return accepted


def connected_component_counter(image, min_area, max_area):
    output = image.copy()

    px, py, pr, plate_mask = detect_petri_dish(image)
    enhanced = enhance_image(image, plate_mask)
    binary = threshold_colonies(enhanced, plate_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    candidates = 0
    points = []

    cv2.circle(output, (px, py), pr, (255, 255, 0), 2)

    for i in range(1, num_labels):
        component_mask = np.zeros(binary.shape, dtype=np.uint8)
        component_mask[labels == i] = 255

        feat = contour_features(component_mask)

        if feat["area"] >= min_area:
            candidates += 1

        if not is_valid_colony(feat, min_area, max_area):
            continue

        cx, cy = feat["cx"], feat["cy"]

        if 0 <= cy < plate_mask.shape[0] and 0 <= cx < plate_mask.shape[1]:
            if plate_mask[cy, cx] == 255:
                points.append((cx, cy, max(3, feat["radius"])))

    points = suppress_duplicates(points, min_dist=int(st.session_state.get("duplicate_distance", 6)))

    for x, y, r in points:
        cv2.circle(output, (int(x), int(y)), int(max(3, min(r, 10))), (0, 255, 0), 2)

    return len(points), output, binary, plate_mask, candidates


def robust_watershed_counter(image, min_area, max_area, min_distance):
    output = image.copy()

    px, py, pr, plate_mask = detect_petri_dish(image)
    enhanced = enhance_image(image, plate_mask)
    binary = threshold_colonies(enhanced, plate_mask)

    distance = ndi.distance_transform_edt(binary)

    coords = peak_local_max(
        distance,
        min_distance=int(min_distance),
        threshold_rel=float(st.session_state.get("marker_threshold_rel", 0.25)),
        labels=binary,
        exclude_border=False,
    )

    markers = np.zeros(distance.shape, dtype=np.int32)

    for idx, (y, x) in enumerate(coords, start=1):
        markers[y, x] = idx

    markers = ndi.label(markers > 0)[0]
    labels = watershed(-distance, markers, mask=binary)

    candidates = 0
    points = []

    cv2.circle(output, (px, py), pr, (255, 255, 0), 2)

    for label in np.unique(labels):
        if label == 0:
            continue

        component_mask = np.zeros(binary.shape, dtype=np.uint8)
        component_mask[labels == label] = 255

        feat = contour_features(component_mask)

        if feat["area"] >= min_area:
            candidates += 1

        if not is_valid_colony(feat, min_area, max_area):
            continue

        cx, cy = feat["cx"], feat["cy"]

        if 0 <= cy < plate_mask.shape[0] and 0 <= cx < plate_mask.shape[1]:
            if plate_mask[cy, cx] == 255:
                points.append((cx, cy, max(3, feat["radius"])))

    points = suppress_duplicates(points, min_dist=int(st.session_state.get("duplicate_distance", 6)))

    for x, y, r in points:
        cv2.circle(output, (int(x), int(y)), int(max(3, min(r, 10))), (0, 255, 0), 2)

    return len(points), output, binary, plate_mask, candidates


def log_blob_counter(image, min_area, max_area):
    output = image.copy()

    px, py, pr, plate_mask = detect_petri_dish(image)
    enhanced = enhance_image(image, plate_mask)

    blobs = blob_log(
        enhanced,
        min_sigma=float(st.session_state.get("log_min_sigma", 2)),
        max_sigma=float(st.session_state.get("log_max_sigma", 9)),
        num_sigma=8,
        threshold=float(st.session_state.get("log_threshold", 0.09)),
    )

    points = []

    cv2.circle(output, (px, py), pr, (255, 255, 0), 2)

    for blob in blobs:
        y, x, sigma = blob
        x, y = int(x), int(y)
        r = float(sigma * np.sqrt(2))

        area_est = np.pi * r * r

        if area_est < min_area or area_est > max_area:
            continue

        if y >= plate_mask.shape[0] or x >= plate_mask.shape[1]:
            continue

        if plate_mask[y, x] == 0:
            continue

        points.append((x, y, max(3, r)))

    points = suppress_duplicates(points, min_dist=int(st.session_state.get("duplicate_distance", 6)))

    for x, y, r in points:
        cv2.circle(output, (int(x), int(y)), int(max(3, min(r, 10))), (0, 255, 0), 2)

    return len(points), output, enhanced, plate_mask, len(blobs)


def density_adjustment(binary, raw_count, plate_mask):
    """
    Estimates colonies hidden inside dense/merged regions.
    Conservative adjustment only for high coverage countable-looking regions.
    """
    if not st.session_state.get("enable_density_adjustment", True):
        return raw_count

    plate_area = np.sum(plate_mask > 0)
    fg_area = np.sum(binary > 0)
    coverage = fg_area / plate_area if plate_area else 0

    if coverage < float(st.session_state.get("density_trigger", 0.18)):
        return raw_count

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    valid_areas = []
    large_areas = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 10:
            continue
        if area <= float(st.session_state.get("typical_colony_max_area", 350)):
            valid_areas.append(area)
        else:
            large_areas.append(area)

    if len(valid_areas) < 5:
        return raw_count

    typical_area = np.median(valid_areas)
    if typical_area <= 0:
        return raw_count

    extra = 0
    for la in large_areas:
        est = int(round(la / typical_area))
        est = max(1, min(est, int(st.session_state.get("max_split_large_region", 20))))
        extra += max(0, est - 1)

    adjusted = raw_count + extra

    max_reasonable = int(st.session_state.get("max_reportable_count", 300))
    return min(adjusted, max_reasonable)


def consensus_counter(image, min_area, max_area, min_distance):
    cc_count, cc_out, cc_binary, plate_mask, cc_cand = connected_component_counter(image, min_area, max_area)
    ws_count, ws_out, ws_binary, _, ws_cand = robust_watershed_counter(image, min_area, max_area, min_distance)
    log_count, _, _, _, log_cand = log_blob_counter(image, min_area, max_area)

    raw_final = int(np.median([cc_count, ws_count, log_count]))
    adjusted = density_adjustment(ws_binary, raw_final, plate_mask)

    return adjusted, ws_out, ws_binary, plate_mask, max(cc_cand, ws_cand, log_cand)


def run_counter(image, method, min_area, max_area, min_distance):
    if method == "Fast Connected Components + Shape Filter":
        count, output, processed, mask, cand = connected_component_counter(image, min_area, max_area)
        return count, output, processed, mask, cand

    if method == "Robust Marker-Controlled Watershed":
        count, output, processed, mask, cand = robust_watershed_counter(image, min_area, max_area, min_distance)
        count = density_adjustment(processed, count, mask)
        return count, output, processed, mask, cand

    if method == "LoG Blob Detector":
        return log_blob_counter(image, min_area, max_area)

    if method == "Recommended Consensus":
        return consensus_counter(image, min_area, max_area, min_distance)

    if method == "Baseline: Simple Thresholding":
        count, output, processed, mask, cand = connected_component_counter(image, min_area, max_area)
        return count, output, processed, mask, cand

    return consensus_counter(image, min_area, max_area, min_distance)


def extract_plate_features(image, auto_count, candidate_count=None):
    px, py, pr, plate_mask = detect_petri_dish(image)
    enhanced = enhance_image(image, plate_mask)
    binary = threshold_colonies(enhanced, plate_mask)

    plate_area = np.sum(plate_mask > 0)
    foreground_area = np.sum(binary > 0)

    coverage_ratio = foreground_area / plate_area if plate_area > 0 else 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    areas = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 10:
            areas.append(area)

    if len(areas) == 0:
        areas = [0]

    areas = np.array(areas)

    edges = cv2.Canny(enhanced, 50, 150)
    edge_density = np.sum(edges > 0) / plate_area if plate_area > 0 else 0

    masked_pixels = enhanced[plate_mask > 0]
    if len(masked_pixels) == 0:
        masked_pixels = np.array([0])

    candidate_count = int(candidate_count) if candidate_count is not None else int(len(areas))
    accepted_fraction = auto_count / candidate_count if candidate_count > 0 else 0

    return {
        "coverage_ratio": float(coverage_ratio),
        "component_count": int(len(areas)),
        "mean_component_area": float(np.mean(areas)),
        "std_component_area": float(np.std(areas)),
        "max_component_area": float(np.max(areas)),
        "brightness_mean": float(np.mean(masked_pixels)),
        "brightness_std": float(np.std(masked_pixels)),
        "edge_density": float(edge_density),
        "auto_count": int(auto_count),
        "candidate_count": int(candidate_count),
        "accepted_fraction": float(accepted_fraction),
    }


def rule_based_countability_label(features, manual_count=None):
    reference_count = manual_count if manual_count is not None else features["auto_count"]

    if reference_count < 25:
        return "not_countable"

    if reference_count > 250:
        return "not_countable"

    if features["coverage_ratio"] > float(st.session_state.get("max_countable_coverage", 0.45)):
        return "not_countable"

    if features["max_component_area"] > float(st.session_state.get("max_countable_component", 10000)):
        return "not_countable"

    if features["brightness_std"] < 4:
        return "not_countable"

    return "countable"


def cfu_status_and_value(auto_count, dilution_factor, plated_volume_ml, manual_count=None):
    reference_count = manual_count if manual_count is not None else auto_count

    if 25 <= reference_count <= 250:
        cfu = (auto_count * dilution_factor) / plated_volume_ml
        return "valid_for_CFU", cfu

    if reference_count < 25:
        return "TFTC_too_few_colonies", None

    return "TNTC_too_many_colonies", None

# ============================================================
# VISUALIZATION
# ============================================================

def plot_manual_vs_auto(df, title):
    if len(df) < 2:
        return

    fig = px.scatter(
        df,
        x="manual_count",
        y="automatic_count",
        hover_name="filename",
        color="rule_decision",
        title=title,
    )

    min_v = float(min(df["manual_count"].min(), df["automatic_count"].min()))
    max_v = float(max(df["manual_count"].max(), df["automatic_count"].max()))

    fig.add_trace(
        go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode="lines",
            name="Ideal agreement",
            line=dict(dash="dash"),
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_count_distribution(results_df):
    fig = px.histogram(results_df, x="automatic_count", nbins=25, title="Automatic Colony Count Distribution")
    st.plotly_chart(fig, use_container_width=True)


def plot_validity_pie(results_df):
    temp = results_df["plate_validity"].fillna("unknown").value_counts().reset_index()
    temp.columns = ["plate_validity", "count"]

    fig = px.pie(temp, names="plate_validity", values="count", title="Plate Validity Split")
    st.plotly_chart(fig, use_container_width=True)


def plot_decision_bar(results_df):
    temp = results_df["rule_decision"].fillna("unknown").value_counts().reset_index()
    temp.columns = ["decision", "count"]

    fig = px.bar(temp, x="decision", y="count", title="Countable vs Manual Review", text="count")
    st.plotly_chart(fig, use_container_width=True)


def plot_error_bar(df, title):
    if len(df) == 0:
        return

    temp = df.sort_values("absolute_error", ascending=False).head(25)
    fig = px.bar(temp, x="filename", y="absolute_error", title=title, text="absolute_error")
    fig.update_layout(xaxis_title="Plate image", yaxis_title="Absolute error")
    st.plotly_chart(fig, use_container_width=True)


def plot_confusion_matrix(cm, labels):
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        title="Classifier Confusion Matrix",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_importance(clf):
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    fig = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Random Forest Feature Importance",
        text_auto=".3f",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def plot_robustness(robustness_df):
    rob_long = robustness_df.melt(
        id_vars="filename",
        value_vars=["dark_deviation", "bright_deviation", "noise_deviation"],
        var_name="Perturbation",
        value_name="Count deviation",
    )

    fig = px.box(rob_long, x="Perturbation", y="Count deviation", title="Robustness: Count Deviation")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Image Source")

source = st.sidebar.radio(
    "Choose image source",
    ["Upload Images", "Upload ZIP within 200 MB"]
)

st.sidebar.header("Classical CV Method")

method = st.sidebar.selectbox(
    "Select counter",
    [
        "Recommended Consensus",
        "Robust Marker-Controlled Watershed",
        "Fast Connected Components + Shape Filter",
        "LoG Blob Detector",
        "Baseline: Simple Thresholding",
    ],
    index=0,
)

max_images = st.sidebar.number_input(
    "Maximum images to process",
    min_value=1,
    max_value=10000,
    value=300,
    step=50,
)

st.sidebar.header("Speed and Mask Settings")

fast_centered_mask = st.sidebar.checkbox("Fast centered petri mask", value=False)
st.session_state["fast_centered_mask"] = fast_centered_mask

mask_radius_factor = st.sidebar.slider("Plate radius factor", 0.35, 0.55, 0.45, 0.01)
inner_mask_factor = st.sidebar.slider("Inner agar mask factor", 0.70, 0.95, 0.88, 0.01)
st.session_state["mask_radius_factor"] = mask_radius_factor
st.session_state["inner_mask_factor"] = inner_mask_factor

skip_preview_processing = st.sidebar.checkbox("Skip preview processing", value=False)

st.sidebar.header("Preprocessing Settings")

threshold_mode = st.sidebar.selectbox("Threshold mode", ["Otsu", "Adaptive"], index=0)
st.session_state["threshold_mode"] = threshold_mode

invert_binary = st.sidebar.checkbox("Invert binary", value=False)
st.session_state["invert_binary"] = invert_binary

clahe_clip = st.sidebar.slider("CLAHE clip limit", 1.0, 5.0, 2.5, 0.1)
background_kernel = st.sidebar.slider("Background median kernel", 31, 101, 51, 2)
tophat_kernel = st.sidebar.slider("Top-hat kernel", 7, 41, 15, 2)

st.session_state["clahe_clip"] = clahe_clip
st.session_state["background_kernel"] = background_kernel
st.session_state["tophat_kernel"] = tophat_kernel

if threshold_mode == "Adaptive":
    adaptive_block = st.sidebar.slider("Adaptive block size", 21, 81, 41, 2)
    adaptive_c = st.sidebar.slider("Adaptive C", -15, 15, -3)
    st.session_state["adaptive_block"] = adaptive_block
    st.session_state["adaptive_c"] = adaptive_c

st.sidebar.header("Colony Filtering")

min_area = st.sidebar.slider("Minimum colony area", 5, 500, 20)
max_area = st.sidebar.slider("Maximum colony area", 100, 10000, 2000)
min_distance = st.sidebar.slider("Watershed marker distance", 5, 40, 16)
marker_threshold_rel = st.sidebar.slider("Marker threshold relative", 0.05, 0.60, 0.25, 0.05)

min_circularity = st.sidebar.slider("Minimum circularity", 0.05, 0.90, 0.25, 0.05)
min_solidity = st.sidebar.slider("Minimum solidity", 0.10, 0.95, 0.45, 0.05)
min_extent = st.sidebar.slider("Minimum extent", 0.05, 0.90, 0.20, 0.05)
duplicate_distance = st.sidebar.slider("Duplicate suppression distance", 2, 20, 6)

st.session_state["marker_threshold_rel"] = marker_threshold_rel
st.session_state["min_circularity"] = min_circularity
st.session_state["min_solidity"] = min_solidity
st.session_state["min_extent"] = min_extent
st.session_state["min_aspect"] = 0.25
st.session_state["max_aspect"] = 4.0
st.session_state["duplicate_distance"] = duplicate_distance

st.sidebar.header("Dense Plate Adjustment")

enable_density_adjustment = st.sidebar.checkbox("Enable density adjustment for merged colonies", value=True)
density_trigger = st.sidebar.slider("Density adjustment trigger", 0.05, 0.50, 0.18, 0.01)
typical_colony_max_area = st.sidebar.slider("Typical colony max area", 80, 1000, 350)
max_split_large_region = st.sidebar.slider("Max split per large region", 2, 40, 20)
max_reportable_count = st.sidebar.slider("Max reportable count", 100, 1000, 300)

st.session_state["enable_density_adjustment"] = enable_density_adjustment
st.session_state["density_trigger"] = density_trigger
st.session_state["typical_colony_max_area"] = typical_colony_max_area
st.session_state["max_split_large_region"] = max_split_large_region
st.session_state["max_reportable_count"] = max_reportable_count

st.sidebar.header("LoG Settings")

log_min_sigma = st.sidebar.slider("LoG min sigma", 1.0, 6.0, 2.0, 0.5)
log_max_sigma = st.sidebar.slider("LoG max sigma", 4.0, 20.0, 9.0, 0.5)
log_threshold = st.sidebar.slider("LoG threshold", 0.01, 0.30, 0.09, 0.01)

st.session_state["log_min_sigma"] = log_min_sigma
st.session_state["log_max_sigma"] = log_max_sigma
st.session_state["log_threshold"] = log_threshold

st.sidebar.header("CFU Settings")

dilution_factor = st.sidebar.number_input("Dilution factor", min_value=1, value=100000)
plated_volume = st.sidebar.number_input("Plated volume in mL", min_value=0.01, value=0.1)

st.sidebar.header("Performance Options")

run_baseline = st.sidebar.checkbox("Run baseline comparison", value=True)
run_classifier = st.sidebar.checkbox("Train ML classifier", value=True)
run_robustness = st.sidebar.checkbox("Run robustness analysis", value=False)

if st.sidebar.button("Clear loaded images"):
    for key in ["image_items", "results_df", "feature_df", "baseline_df", "robustness_df"]:
        st.session_state.pop(key, None)
    st.rerun()

# ============================================================
# LOAD IMAGES
# ============================================================

st.header("1. Load Petri Plate Images")

image_items = []

if source == "Upload Images":
    uploaded_files = st.file_uploader(
        "Upload images",
        type=VALID_IMAGE_TYPES,
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files[:max_images]:
            try:
                image_items.append({
                    "name": file.name,
                    "image": read_uploaded_image(file),
                    "type": "array",
                })
            except Exception as e:
                st.error(f"Could not read {file.name}: {e}")

        st.session_state["image_items"] = image_items
        if len(image_items) > 0:
            st.success(f"{len(image_items)} uploaded images loaded.")

elif source == "Upload ZIP":
    st.warning(
        "For ZIP files above the Streamlit browser limit, use Large ZIP Path instead. "
        "Large ZIP Path reads the ZIP directly from disk/Colab and avoids browser upload limits."
    )

    zip_file = st.file_uploader("Upload ZIP", type=["zip"])

    if zip_file is not None:
        temp_dir = tempfile.mkdtemp(prefix="uploaded_zip_stream_")
        zip_path = os.path.join(temp_dir, "uploaded_dataset.zip")

        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        try:
            image_items = list_zip_images(zip_path, max_images)
            st.session_state["image_items"] = image_items
            if len(image_items) > 0:
                st.success(f"{len(image_items)} images indexed from ZIP without full extraction.")
            else:
                st.warning("ZIP opened, but no supported images were found.")
        except Exception as e:
            st.error(f"Could not process ZIP: {e}")

elif source == "Large ZIP Path":
    st.info(
        "Use this for ZIP files above 1GB. Put the ZIP on the local machine or Colab/Drive first, "
        "then paste the full file path."
    )

    zip_path_input = st.text_input(
        "Large ZIP file path",
        value="/content/drive/MyDrive/Petri_plates.zip"
    )

    if st.button("Index Large ZIP"):
        if not os.path.exists(zip_path_input):
            st.error("ZIP path not found. Check the file path.")
        else:
            try:
                image_items = list_zip_images(zip_path_input, max_images)
                st.session_state["image_items"] = image_items
                if len(image_items) > 0:
                    st.success(f"{len(image_items)} images indexed from large ZIP.")
                else:
                    st.warning("ZIP opened, but no supported image files were found.")
            except Exception as e:
                st.error(f"Could not index ZIP: {e}")

if "image_items" in st.session_state:
    image_items = st.session_state["image_items"]

if len(image_items) == 0:
    st.warning("No images loaded yet.")
    st.stop()

st.success(f"Total loaded images: {len(image_items)}")

# ============================================================
# OPTIONAL VALIDATION CSV
# ============================================================

st.header("2. Optional Lab Validation CSV")

st.markdown("""
Upload a CSV if available. Required columns:

```text
filename, manual_count, countability_label
```

If no CSV is uploaded, manual count is extracted from the last number in the filename.
""")

label_file = st.file_uploader(
    "Upload CSV with columns: filename, manual_count, countability_label",
    type=["csv"]
)

lab_label_df = None

if label_file is not None:
    try:
        lab_label_df = pd.read_csv(label_file)
        st.dataframe(lab_label_df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

# ============================================================
# PREVIEW
# ============================================================

st.header("3. Preview Single Plate")

selected_item = st.selectbox(
    "Preview image",
    options=image_items,
    index=0,
    format_func=lambda x: x["name"] if isinstance(x, dict) and "name" in x else "Unknown image",
)

if selected_item is None:
    st.warning("No image selected.")
    st.stop()

try:
    image = load_item_image(selected_item)
except Exception as e:
    st.error(f"Could not read selected image: {e}")
    st.stop()

manual_count = get_manual_count(selected_item["name"], lab_label_df)

if skip_preview_processing:
    st.image(image, caption=f"Original | Manual: {manual_count}", use_container_width=True)
    st.info("Preview processing skipped for speed. Click Process All Images to run counting.")
else:
    auto_count, output, processed_view, plate_mask, candidate_count = run_counter(
        image,
        method,
        min_area,
        max_area,
        min_distance,
    )

    features = extract_plate_features(image, auto_count, candidate_count)
    rule_decision = rule_based_countability_label(features, manual_count)

    plate_validity, estimated_cfu = cfu_status_and_value(
        auto_count,
        dilution_factor,
        plated_volume,
        manual_count,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Manual count", manual_count if manual_count is not None else "NA")
    m2.metric("Automatic count", auto_count)
    m3.metric("Candidates", candidate_count)
    m4.metric("Decision", rule_decision)
    m5.metric("CFU/mL", f"{estimated_cfu:.2e}" if estimated_cfu is not None else "NA")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.image(image, caption=f"Original | Manual: {manual_count}", use_container_width=True)

    with c2:
        st.image(plate_mask, caption="Petri/Agar Mask", use_container_width=True)

    with c3:
        st.image(processed_view, caption="Processed Binary View", use_container_width=True)

    with c4:
        st.image(output, caption=f"Overlay Count: {auto_count}", use_container_width=True)

    preview_df = pd.DataFrame([{
        "filename": selected_item["name"],
        "manual_count": manual_count,
        "automatic_count": auto_count,
        "candidate_count": candidate_count,
        "absolute_error": abs(manual_count - auto_count) if manual_count is not None else None,
        "percentage_error": safe_percentage_error(manual_count, auto_count),
        "rule_decision": rule_decision,
        "plate_validity": plate_validity,
        "estimated_CFU_per_mL": estimated_cfu,
        "coverage_ratio": features["coverage_ratio"],
        "accepted_fraction": features["accepted_fraction"],
        "method": method,
    }])

    st.dataframe(preview_df, use_container_width=True)

# ============================================================
# BATCH PROCESSING
# ============================================================

st.header("4. Process All Plates")

if st.button("Process All Images", type="primary"):
    results = []
    feature_rows = []
    baseline_rows = []
    robustness_rows = []

    progress = st.progress(0)
    status_box = st.empty()

    for i, item in enumerate(image_items):
        try:
            status_box.info(f"Processing {i + 1}/{len(image_items)}: {item['name']}")

            img = load_item_image(item)
            filename = item["name"]

            manual = get_manual_count(filename, lab_label_df)
            true_label = get_lab_label(filename, lab_label_df, manual)

            auto_count, _, processed, plate_mask, candidate_count = run_counter(
                img,
                method,
                min_area,
                max_area,
                min_distance,
            )

            features = extract_plate_features(img, auto_count, candidate_count)
            rule_label = rule_based_countability_label(features, manual)
            training_label = true_label if true_label is not None else rule_label

            abs_error = abs(manual - auto_count) if manual is not None else None
            pct_error = safe_percentage_error(manual, auto_count)

            plate_validity, estimated_cfu = cfu_status_and_value(
                auto_count,
                dilution_factor,
                plated_volume,
                manual,
            )

            results.append({
                "filename": filename,
                "manual_count": manual,
                "automatic_count": auto_count,
                "candidate_count": candidate_count,
                "absolute_error": abs_error,
                "percentage_error": pct_error,
                "manual_countability_label": true_label,
                "rule_decision": rule_label,
                "training_label_used": training_label,
                "plate_validity": plate_validity,
                "estimated_CFU_per_mL": estimated_cfu,
                "method": method,
            })

            feature_row = features.copy()
            feature_row["filename"] = filename
            feature_row["label"] = training_label
            feature_rows.append(feature_row)

            if run_baseline:
                cc_count, _, _, _, _ = connected_component_counter(img, min_area, max_area)
                ws_count, _, _, _, _ = robust_watershed_counter(img, min_area, max_area, min_distance)
                log_count, _, _, _, _ = log_blob_counter(img, min_area, max_area)
                cons_count, _, _, _, _ = consensus_counter(img, min_area, max_area, min_distance)

                baseline_rows.append({
                    "filename": filename,
                    "manual_count": manual,
                    "connected_components_count": cc_count,
                    "watershed_count": ws_count,
                    "log_count": log_count,
                    "consensus_count": cons_count,
                    "connected_components_absolute_error": abs(manual - cc_count) if manual is not None else None,
                    "watershed_absolute_error": abs(manual - ws_count) if manual is not None else None,
                    "log_absolute_error": abs(manual - log_count) if manual is not None else None,
                    "consensus_absolute_error": abs(manual - cons_count) if manual is not None else None,
                    "connected_components_percentage_error": safe_percentage_error(manual, cc_count),
                    "watershed_percentage_error": safe_percentage_error(manual, ws_count),
                    "log_percentage_error": safe_percentage_error(manual, log_count),
                    "consensus_percentage_error": safe_percentage_error(manual, cons_count),
                })

            if run_robustness:
                normal_count, _, _, _, _ = run_counter(img, method, min_area, max_area, min_distance)

                dark_img = np.clip(img * 0.65, 0, 255).astype(np.uint8)
                bright_img = np.clip(img * 1.25, 0, 255).astype(np.uint8)
                noise = np.random.normal(0, 15, img.shape)
                noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)

                dark_count, _, _, _, _ = run_counter(dark_img, method, min_area, max_area, min_distance)
                bright_count, _, _, _, _ = run_counter(bright_img, method, min_area, max_area, min_distance)
                noisy_count, _, _, _, _ = run_counter(noisy_img, method, min_area, max_area, min_distance)

                robustness_rows.append({
                    "filename": filename,
                    "normal_count": normal_count,
                    "dark_count": dark_count,
                    "bright_count": bright_count,
                    "noisy_count": noisy_count,
                    "dark_deviation": abs(normal_count - dark_count),
                    "bright_deviation": abs(normal_count - bright_count),
                    "noise_deviation": abs(normal_count - noisy_count),
                })

        except Exception as e:
            results.append({
                "filename": item.get("name", "unknown"),
                "error": str(e),
            })

        progress.progress((i + 1) / len(image_items))

    status_box.success("Processing completed.")

    st.session_state["results_df"] = pd.DataFrame(results)
    st.session_state["feature_df"] = pd.DataFrame(feature_rows)
    st.session_state["baseline_df"] = pd.DataFrame(baseline_rows)
    st.session_state["robustness_df"] = pd.DataFrame(robustness_rows)

# ============================================================
# RESULTS DASHBOARD
# ============================================================

if "results_df" in st.session_state:
    results_df = st.session_state["results_df"]
    feature_df = st.session_state.get("feature_df", pd.DataFrame())
    baseline_df = st.session_state.get("baseline_df", pd.DataFrame())
    robustness_df = st.session_state.get("robustness_df", pd.DataFrame())

    st.header("5. Results and Validation Dashboard")

    st.subheader("A. Agreement Table vs Manual Counts")
    st.dataframe(results_df, use_container_width=True)

    all_valid = all_manual_df(results_df)
    countable_valid = countable_validation_df(results_df)

    st.subheader("A1. Visual Dashboard")

    d1, d2, d3 = st.columns(3)

    with d1:
        plot_count_distribution(results_df)

    with d2:
        plot_validity_pie(results_df)

    with d3:
        plot_decision_bar(results_df)

    st.subheader("A2. Correct Count Accuracy: Countable Plates Only")

    st.info(
        "MAE, MAPE, and R² are calculated only for plates with manual count between 25 and 250. "
        "TFTC and TNTC plates are excluded from count-error validation."
    )

    if len(countable_valid) > 1:
        mae = mean_absolute_error(countable_valid["manual_count"], countable_valid["automatic_count"])
        mape = countable_valid["percentage_error"].dropna().mean()
        estimated_accuracy = max(0, 100 - mape)
        r2 = safe_r2(countable_valid["manual_count"], countable_valid["automatic_count"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Countable-Only MAE", round(mae, 2))
        m2.metric("Countable-Only MAPE (%)", round(mape, 2))
        m3.metric("Countable-Only Accuracy (%)", round(estimated_accuracy, 2))
        m4.metric("Countable-Only R²", round(r2, 3) if r2 is not None else "NA")

        st.dataframe(countable_valid, use_container_width=True)
        plot_manual_vs_auto(countable_valid, "Manual vs Automatic Count: Countable Plates Only")
        plot_error_bar(countable_valid, "Top Absolute Errors: Countable Plates Only")
    else:
        st.warning("Not enough countable plates with manual counts between 25 and 250.")

    with st.expander("Show all-plate metrics for reference only"):
        if len(all_valid) > 1:
            mae_all = mean_absolute_error(all_valid["manual_count"], all_valid["automatic_count"])
            mape_all = all_valid["percentage_error"].dropna().mean()
            r2_all = safe_r2(all_valid["manual_count"], all_valid["automatic_count"])

            c1, c2, c3 = st.columns(3)
            c1.metric("All-Plate MAE", round(mae_all, 2))
            c2.metric("All-Plate MAPE (%)", round(mape_all, 2))
            c3.metric("All-Plate R²", round(r2_all, 3) if r2_all is not None else "NA")

            plot_manual_vs_auto(all_valid, "Manual vs Automatic Count: All Plates")

    if run_baseline and len(baseline_df) > 0:
        st.subheader("B. Baseline vs Improved Methods")
        st.dataframe(baseline_df, use_container_width=True)

        baseline_countable = baseline_df[
            (baseline_df["manual_count"].notna()) &
            (baseline_df["manual_count"] >= 25) &
            (baseline_df["manual_count"] <= 250)
        ].copy()

        if len(baseline_countable) > 1:
            comparison_table = pd.DataFrame([
                {
                    "Method": "Connected Components",
                    "MAE": round(mean_absolute_error(
                        baseline_countable["manual_count"],
                        baseline_countable["connected_components_count"],
                    ), 2),
                    "MAPE (%)": round(baseline_countable["connected_components_percentage_error"].dropna().mean(), 2),
                    "R²": round(safe_r2(
                        baseline_countable["manual_count"],
                        baseline_countable["connected_components_count"],
                    ), 3),
                },
                {
                    "Method": "Watershed",
                    "MAE": round(mean_absolute_error(
                        baseline_countable["manual_count"],
                        baseline_countable["watershed_count"],
                    ), 2),
                    "MAPE (%)": round(baseline_countable["watershed_percentage_error"].dropna().mean(), 2),
                    "R²": round(safe_r2(
                        baseline_countable["manual_count"],
                        baseline_countable["watershed_count"],
                    ), 3),
                },
                {
                    "Method": "LoG",
                    "MAE": round(mean_absolute_error(
                        baseline_countable["manual_count"],
                        baseline_countable["log_count"],
                    ), 2),
                    "MAPE (%)": round(baseline_countable["log_percentage_error"].dropna().mean(), 2),
                    "R²": round(safe_r2(
                        baseline_countable["manual_count"],
                        baseline_countable["log_count"],
                    ), 3),
                },
                {
                    "Method": "Consensus",
                    "MAE": round(mean_absolute_error(
                        baseline_countable["manual_count"],
                        baseline_countable["consensus_count"],
                    ), 2),
                    "MAPE (%)": round(baseline_countable["consensus_percentage_error"].dropna().mean(), 2),
                    "R²": round(safe_r2(
                        baseline_countable["manual_count"],
                        baseline_countable["consensus_count"],
                    ), 3),
                },
            ])

            st.dataframe(comparison_table, use_container_width=True)

            comp_long = comparison_table.melt(
                id_vars="Method",
                value_vars=["MAE", "MAPE (%)"],
                var_name="Metric",
                value_name="Value",
            )

            fig_comp = px.bar(
                comp_long,
                x="Method",
                y="Value",
                color="Metric",
                barmode="group",
                title="Method Comparison: Countable-Plate Error",
                text_auto=True,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    if run_classifier:
        st.subheader("C. Small Countability ML Classifier")

        if len(feature_df) > 0 and "label" in feature_df.columns and feature_df["label"].nunique() >= 2:
            X = feature_df[FEATURE_COLUMNS]
            y = feature_df["label"]

            min_class_count = y.value_counts().min()
            stratify_arg = y if min_class_count >= 2 and len(y) >= 4 else None

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.25,
                    random_state=42,
                    stratify=stratify_arg,
                )

                clf = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                )

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, pos_label="countable", zero_division=0)
                recall = recall_score(y_test, y_pred, pos_label="countable", zero_division=0)
                f1 = f1_score(y_test, y_pred, pos_label="countable", zero_division=0)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", round(acc, 3))
                c2.metric("Precision", round(precision, 3))
                c3.metric("Recall", round(recall, 3))
                c4.metric("F1 Score", round(f1, 3))

                labels_sorted = sorted(list(set(y_test) | set(y_pred)))
                cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
                plot_confusion_matrix(cm, labels_sorted)

                st.text(classification_report(y_test, y_pred, zero_division=0))
                plot_feature_importance(clf)

                feature_df["ml_predicted_decision"] = clf.predict(feature_df[FEATURE_COLUMNS])
                st.dataframe(feature_df, use_container_width=True)

                model_path = os.path.join(tempfile.mkdtemp(), "countability_random_forest.pkl")
                joblib.dump(clf, model_path)

                with open(model_path, "rb") as f:
                    st.download_button(
                        "Download Random Forest Classifier",
                        data=f,
                        file_name="countability_random_forest.pkl",
                        mime="application/octet-stream",
                    )

            except Exception as e:
                st.warning(f"Classifier could not be trained: {e}")
        else:
            st.warning("Classifier needs both classes: countable and not_countable.")

    if run_robustness and len(robustness_df) > 0:
        st.subheader("D. Robustness Analysis")
        st.dataframe(robustness_df, use_container_width=True)
        plot_robustness(robustness_df)

    st.subheader("E. Downloads")

    st.download_button(
        "Download Final Results CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="final_colony_counting_results.csv",
        mime="text/csv",
    )

    if len(feature_df) > 0:
        st.download_button(
            "Download Feature Dataset CSV",
            data=feature_df.to_csv(index=False).encode("utf-8"),
            file_name="countability_feature_dataset.csv",
            mime="text/csv",
        )

    if run_baseline and len(baseline_df) > 0:
        st.download_button(
            "Download Baseline Comparison CSV",
            data=baseline_df.to_csv(index=False).encode("utf-8"),
            file_name="baseline_comparison_results.csv",
            mime="text/csv",
        )

    if run_robustness and len(robustness_df) > 0:
        st.download_button(
            "Download Robustness CSV",
            data=robustness_df.to_csv(index=False).encode("utf-8"),
            file_name="robustness_analysis_results.csv",
            mime="text/csv",
        )

