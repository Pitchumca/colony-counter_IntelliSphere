import os, re, glob, zipfile, tempfile, cv2, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log
from skimage.segmentation import watershed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

st.set_page_config(
    page_title="AVIT Colony Counter",
    layout="wide"
)

st.title("Automated Bacterial Colony Counter")
st.caption("AVIT Faculty Hackathon 2026 · Classical CV Counter + Countable/Manual-Review ML Classifier")

st.markdown("""
### Requirement Alignment
- **Counting:** deterministic classical computer vision only  
- **Preprocessing:** resize + petri mask + **CLAHE + Otsu**
- **Baseline:** simple thresholding + connected components  
- **Improved segmentation:** marker-controlled watershed  
- **Best method:** Watershed + LoG + Hough consensus  
- **ML:** Random Forest only for countable/not-countable decision  
- **Output:** colony count, CFU/mL, validation metrics, manual-review flag  
""")

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
    "auto_count"
]


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def find_all_images(folder):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(list(set(paths)))


@st.cache_data(show_spinner=False)
def read_image_cached(path, max_width=650):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    return img


def read_image(path, max_width=650):
    return read_image_cached(path, max_width)


def read_uploaded_image(uploaded_file, max_width=650):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Cannot read uploaded image: {uploaded_file.name}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    return img


def extract_manual_count_from_filename(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def get_manual_count(filename, lab_df=None):
    if lab_df is not None and "filename" in lab_df.columns and "manual_count" in lab_df.columns:
        matched = lab_df[lab_df["filename"] == filename]
        if len(matched) > 0:
            value = matched.iloc[0]["manual_count"]
            if pd.notna(value):
                return int(value)

    return extract_manual_count_from_filename(filename)


def get_lab_label(filename, lab_df=None):
    if lab_df is not None and "filename" in lab_df.columns and "countability_label" in lab_df.columns:
        matched = lab_df[lab_df["filename"] == filename]
        if len(matched) > 0:
            label = str(matched.iloc[0]["countability_label"]).strip()
            if label in ["countable", "not_countable"]:
                return label
    return None


# ============================================================
# PETRI MASK + CLAHE + OTSU
# ============================================================

def detect_petri_dish(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 9)

    h, w = gray.shape

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 2,
        param1=80,
        param2=35,
        minRadius=int(min(h, w) * 0.25),
        maxRadius=int(min(h, w) * 0.50)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
    else:
        x, y = w // 2, h // 2
        r = int(min(h, w) * 0.42)

    inner_r = int(r * 0.78)

    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(inner_r), 255, -1)

    return int(x), int(y), int(inner_r), mask


def clahe_enhance(image, plate_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=plate_mask)

    background = cv2.medianBlur(masked, 51)
    corrected = cv2.subtract(masked, background)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(corrected)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=plate_mask)

    return enhanced


def otsu_binary(enhanced, plate_mask):
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)

    return binary


# ============================================================
# CLASSICAL CV COUNTERS
# ============================================================

def baseline_threshold_counter(image, min_area=20, max_area=2000):
    output = image.copy()

    px, py, pr, plate_mask = detect_petri_dish(image)
    enhanced = clahe_enhance(image, plate_mask)
    binary = otsu_binary(enhanced, plate_mask)

    kernel = np.ones((3, 3), np.uint8)

    clean = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean)

    count = 0
    cv2.circle(output, (px, py), pr, (255, 255, 0), 2)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if min_area <= area <= max_area:
            x, y = centroids[i]
            count += 1
            cv2.circle(output, (int(x), int(y)), 6, (255, 0, 0), 2)

    return count, output, binary, plate_mask


def watershed_counter(image, min_area=20, max_area=2000, min_distance=10):
    output = image.copy()

    px, py, pr, plate_mask = detect_petri_dish(image)
    enhanced = clahe_enhance(image, plate_mask)
    binary = otsu_binary(enhanced, plate_mask)

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    distance = ndi.distance_transform_edt(opening)

    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_abs=2,
        labels=opening
    )

    markers = np.zeros(distance.shape, dtype=np.int32)

    for idx, (y, x) in enumerate(coords, start=1):
        markers[y, x] = idx

    markers = ndi.label(markers > 0)[0]

    labels = watershed(
        -distance,
        markers,
        mask=opening
    )

    count = 0
    cv2.circle(output, (px, py), pr, (255, 255, 0), 2)

    for label in np.unique(labels):
        if label == 0:
            continue

        component_mask = np.zeros(enhanced.shape, dtype=np.uint8)
        component_mask[labels == label] = 255

        area = cv2.countNonZero(component_mask)

        if min_area <= area <= max_area:
            moments = cv2.moments(component_mask)

            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])

                if plate_mask[cY, cX] == 255:
                    count += 1
                    cv2.circle(output, (cX, cY), 6, (0, 255, 0), 2)

    return count, output, binary, plate_mask


def log_blob_counter(image):
    output = image.copy()

    px, py, pr, plate_mask = detect_petri_dish(image)
    enhanced = clahe_enhance(image, plate_mask)

    blobs = blob_log(
        enhanced,
        min_sigma=2,
        max_sigma=10,
        num_sigma=8,
        threshold=0.08
    )

    used_centers = []
    count = 0

    cv2.circle(output, (px, py), pr, (255, 255, 0), 2)

    for blob in blobs:
        y, x, sigma = blob
        x = int(x)
        y = int(y)

        r = int(sigma * np.sqrt(2))
        r = max(3, min(r, 18))

        if y >= plate_mask.shape[0] or x >= plate_mask.shape[1]:
            continue

        if plate_mask[y, x] == 0:
            continue

        duplicate = False

        for ux, uy in used_centers:
            if np.sqrt((x - ux) ** 2 + (y - uy) ** 2) < 8:
                duplicate = True
                break

        if duplicate:
            continue

        used_centers.append((x, y))
        count += 1

        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (255, 0, 0), -1)

    return count, output, enhanced, plate_mask


def hough_circle_counter(image):
    output = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    px, py, pr, plate_mask = detect_petri_dish(image)

    masked = cv2.bitwise_and(gray, gray, mask=plate_mask)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(masked)
    blur = cv2.GaussianBlur(enhanced, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=14,
        param1=60,
        param2=16,
        minRadius=3,
        maxRadius=14
    )

    count = 0
    used_centers = []

    cv2.circle(output, (px, py), pr, (255, 255, 0), 2)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for c in circles[0, :]:
            x, y, r = int(c[0]), int(c[1]), int(c[2])

            if y >= plate_mask.shape[0] or x >= plate_mask.shape[1]:
                continue

            if plate_mask[y, x] == 0:
                continue

            duplicate = False

            for ux, uy in used_centers:
                if np.sqrt((x - ux) ** 2 + (y - uy) ** 2) < 10:
                    duplicate = True
                    break

            if duplicate:
                continue

            used_centers.append((x, y))
            count += 1

            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (255, 0, 0), -1)

    return count, output, enhanced, plate_mask


def recommended_counter(image):
    water_count, water_out, processed, plate_mask = watershed_counter(
        image,
        min_area,
        max_area,
        min_distance
    )

    log_count, _, _, _ = log_blob_counter(image)
    hough_count, _, _, _ = hough_circle_counter(image)

    final_count = int(np.median([water_count, log_count, hough_count]))

    return final_count, water_out, processed, plate_mask


def run_counter(image, method):
    if method == "Fast Recommended: Watershed Only":
        return watershed_counter(image, min_area, max_area, min_distance)

    if method == "Recommended: Watershed + LoG + Hough Consensus":
        return recommended_counter(image)

    if method == "LoG Blob Detector":
        return log_blob_counter(image)

    if method == "Hough Circle Detector":
        return hough_circle_counter(image)

    if method == "Baseline: Simple Thresholding":
        return baseline_threshold_counter(image, min_area, max_area)

    return watershed_counter(image, min_area, max_area, min_distance)


# ============================================================
# FEATURES, DECISION, CFU
# ============================================================

def extract_plate_features(image, auto_count):
    _, _, _, plate_mask = detect_petri_dish(image)
    enhanced = clahe_enhance(image, plate_mask)
    binary = otsu_binary(enhanced, plate_mask)

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

    return {
        "coverage_ratio": float(coverage_ratio),
        "component_count": int(len(areas)),
        "mean_component_area": float(np.mean(areas)),
        "std_component_area": float(np.std(areas)),
        "max_component_area": float(np.max(areas)),
        "brightness_mean": float(np.mean(enhanced[plate_mask > 0])),
        "brightness_std": float(np.std(enhanced[plate_mask > 0])),
        "edge_density": float(edge_density),
        "auto_count": int(auto_count)
    }


def rule_based_countability_label(features, manual_count=None):
    reference_count = manual_count if manual_count is not None else features["auto_count"]

    if reference_count < 25:
        return "not_countable"

    if reference_count > 250:
        return "not_countable"

    if features["coverage_ratio"] > 0.40:
        return "not_countable"

    if features["max_component_area"] > 8000:
        return "not_countable"

    if features["brightness_std"] < 5:
        return "not_countable"

    if features["edge_density"] > 0.35:
        return "not_countable"

    return "countable"


def cfu_status_and_value(auto_count, dilution_factor, plated_volume_ml, manual_count=None):
    reference_count = manual_count if manual_count is not None else auto_count

    if 25 <= reference_count <= 250:
        cfu = (auto_count * dilution_factor) / plated_volume_ml
        return "valid_for_CFU", cfu

    if reference_count < 25:
        return "too_few_colonies_unreliable", None

    return "TNTC_too_many_colonies", None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Image Source")

source = st.sidebar.radio(
    "Choose image source",
    ["Google Drive Folder", "Upload Images", "Upload ZIP"]
)

st.sidebar.header("Classical CV Method")

method = st.sidebar.selectbox(
    "Select counter",
    [
        "Fast Recommended: Watershed Only",
        "Recommended: Watershed + LoG + Hough Consensus",
        "LoG Blob Detector",
        "Hough Circle Detector",
        "Baseline: Simple Thresholding"
    ]
)

max_images = st.sidebar.number_input(
    "Maximum images to process",
    min_value=1,
    max_value=10000,
    value=300,
    step=50
)

st.sidebar.header("Segmentation Parameters")

min_area = st.sidebar.slider("Minimum colony area", 5, 500, 20)
max_area = st.sidebar.slider("Maximum colony area", 100, 10000, 2000)
min_distance = st.sidebar.slider("Watershed minimum distance", 3, 40, 10)

st.sidebar.header("CFU Settings")

dilution_factor = st.sidebar.number_input("Dilution factor", min_value=1, value=100000)
plated_volume = st.sidebar.number_input("Plated volume in mL", min_value=0.01, value=0.1)

st.sidebar.header("Performance Options")

run_baseline = st.sidebar.checkbox("Run baseline comparison", value=True)
run_classifier = st.sidebar.checkbox("Train ML classifier", value=True)
run_robustness = st.sidebar.checkbox("Run robustness analysis", value=False)


# ============================================================
# IMAGE LOADING
# ============================================================

st.header("1. Load Petri Plate Images")

image_items = []

if source == "Google Drive Folder":
    folder = st.text_input("Google Drive folder path", "/content/drive/MyDrive/Petri_plates")

    if st.button("Load Images from Drive"):
        if os.path.exists(folder):
            image_paths = find_all_images(folder)[:max_images]

            for path in image_paths:
                image_items.append({
                    "name": os.path.basename(path),
                    "path": path,
                    "type": "path"
                })

            st.session_state["image_items"] = image_items
            st.success(f"{len(image_items)} images loaded.")
        else:
            st.error("Folder not found. Mount Google Drive first.")

elif source == "Upload Images":
    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files[:max_images]:
            image_items.append({
                "name": file.name,
                "image": read_uploaded_image(file),
                "type": "array"
            })

        st.session_state["image_items"] = image_items
        st.success(f"{len(image_items)} uploaded images loaded.")

elif source == "Upload ZIP":
    zip_file = st.file_uploader("Upload ZIP", type=["zip"])

    if zip_file is not None:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "uploaded_dataset.zip")

        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(temp_dir)

        image_paths = find_all_images(temp_dir)[:max_images]

        for path in image_paths:
            image_items.append({
                "name": os.path.basename(path),
                "path": path,
                "type": "path"
            })

        st.session_state["image_items"] = image_items
        st.success(f"{len(image_items)} images extracted from ZIP.")

if "image_items" in st.session_state:
    image_items = st.session_state["image_items"]

if len(image_items) == 0:
    st.warning("No images loaded yet.")
    st.stop()

st.success(f"Total loaded images: {len(image_items)}")


# ============================================================
# OPTIONAL LABEL CSV
# ============================================================

st.header("2. Optional Lab Validation CSV")

label_file = st.file_uploader(
    "Upload CSV with columns: filename, manual_count, countability_label",
    type=["csv"]
)

lab_label_df = None

if label_file is not None:
    lab_label_df = pd.read_csv(label_file)
    st.dataframe(lab_label_df, use_container_width=True)


def get_manual_count(filename, lab_df=None):
    if lab_df is not None and "filename" in lab_df.columns and "manual_count" in lab_df.columns:
        matched = lab_df[lab_df["filename"] == filename]
        if len(matched) > 0:
            value = matched.iloc[0]["manual_count"]
            if pd.notna(value):
                return int(value)

    return extract_manual_count_from_filename(filename)


def get_lab_label(filename, lab_df=None):
    if lab_df is not None and "filename" in lab_df.columns and "countability_label" in lab_df.columns:
        matched = lab_df[lab_df["filename"] == filename]
        if len(matched) > 0:
            label = str(matched.iloc[0]["countability_label"]).strip()
            if label in ["countable", "not_countable"]:
                return label
    return None


# ============================================================
# PREVIEW
# ============================================================

st.header("3. Preview Single Plate")

selected_item = st.selectbox(
    "Preview image",
    image_items,
    format_func=lambda x: x["name"]
)

image = read_image(selected_item["path"]) if selected_item["type"] == "path" else selected_item["image"]

manual_count = get_manual_count(selected_item["name"], lab_label_df)

auto_count, output, processed_view, plate_mask = run_counter(image, method)

features = extract_plate_features(image, auto_count)

rule_decision = rule_based_countability_label(features, manual_count)

plate_validity, estimated_cfu = cfu_status_and_value(
    auto_count,
    dilution_factor,
    plated_volume,
    manual_count
)

c1, c2, c3 = st.columns(3)

with c1:
    st.image(image, caption=f"Original | Manual Count: {manual_count}", use_container_width=True)

with c2:
    st.image(plate_mask, caption="Inner Agar Mask", use_container_width=True)

with c3:
    st.image(output, caption=f"Automatic Count: {auto_count} | {rule_decision}", use_container_width=True)

preview_df = pd.DataFrame([{
    "filename": selected_item["name"],
    "manual_count": manual_count,
    "automatic_count": auto_count,
    "rule_decision": rule_decision,
    "plate_validity": plate_validity,
    "estimated_CFU_per_mL": estimated_cfu,
    "coverage_ratio": features["coverage_ratio"],
    "component_count": features["component_count"],
    "method": method
}])

st.dataframe(preview_df, use_container_width=True)


# ============================================================
# BATCH
# ============================================================

st.header("4. Process All Plates")

if st.button("Process All Images"):

    results = []
    feature_rows = []
    baseline_rows = []
    robustness_rows = []

    progress = st.progress(0)

    for i, item in enumerate(image_items):
        try:
            img = read_image(item["path"]) if item["type"] == "path" else item["image"]
            filename = item["name"]

            manual = get_manual_count(filename, lab_label_df)
            true_label = get_lab_label(filename, lab_label_df)

            auto_count, _, _, _ = run_counter(img, method)
            features = extract_plate_features(img, auto_count)

            rule_label = rule_based_countability_label(features, manual)
            training_label = true_label if true_label is not None else rule_label

            abs_error = None
            pct_error = None

            if manual is not None:
                abs_error = abs(manual - auto_count)
                if manual > 0:
                    pct_error = (abs_error / manual) * 100

            plate_validity, estimated_cfu = cfu_status_and_value(
                auto_count,
                dilution_factor,
                plated_volume,
                manual
            )

            results.append({
                "filename": filename,
                "manual_count": manual,
                "automatic_count": auto_count,
                "absolute_error": abs_error,
                "percentage_error": pct_error,
                "rule_decision": rule_label,
                "true_countability_label": true_label,
                "training_label_used": training_label,
                "plate_validity": plate_validity,
                "estimated_CFU_per_mL": estimated_cfu,
                "method": method
            })

            feature_row = features.copy()
            feature_row["filename"] = filename
            feature_row["label"] = training_label
            feature_rows.append(feature_row)

            if run_baseline:
                baseline_count, _, _, _ = baseline_threshold_counter(img, min_area, max_area)
                improved_count, _, _, _ = watershed_counter(img, min_area, max_area, min_distance)

                if method == "Recommended: Watershed + LoG + Hough Consensus":
                    recommended_count, _, _, _ = recommended_counter(img)
                else:
                    recommended_count = auto_count

                baseline_abs = abs(manual - baseline_count) if manual is not None else None
                improved_abs = abs(manual - improved_count) if manual is not None else None
                recommended_abs = abs(manual - recommended_count) if manual is not None else None

                baseline_pct = (baseline_abs / manual) * 100 if manual not in [None, 0] else None
                improved_pct = (improved_abs / manual) * 100 if manual not in [None, 0] else None
                recommended_pct = (recommended_abs / manual) * 100 if manual not in [None, 0] else None

                baseline_rows.append({
                    "filename": filename,
                    "manual_count": manual,
                    "baseline_simple_threshold_count": baseline_count,
                    "improved_watershed_count": improved_count,
                    "recommended_count": recommended_count,
                    "baseline_absolute_error": baseline_abs,
                    "improved_absolute_error": improved_abs,
                    "recommended_absolute_error": recommended_abs,
                    "baseline_percentage_error": baseline_pct,
                    "improved_percentage_error": improved_pct,
                    "recommended_percentage_error": recommended_pct
                })

            if run_robustness:
                normal_count, _, _, _ = run_counter(img, method)

                dark_img = np.clip(img * 0.65, 0, 255).astype(np.uint8)
                bright_img = np.clip(img * 1.25, 0, 255).astype(np.uint8)
                noise = np.random.normal(0, 15, img.shape)
                noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)

                dark_count, _, _, _ = run_counter(dark_img, method)
                bright_count, _, _, _ = run_counter(bright_img, method)
                noisy_count, _, _, _ = run_counter(noisy_img, method)

                robustness_rows.append({
                    "filename": filename,
                    "normal_count": normal_count,
                    "dark_count": dark_count,
                    "bright_count": bright_count,
                    "noisy_count": noisy_count,
                    "dark_deviation": abs(normal_count - dark_count),
                    "bright_deviation": abs(normal_count - bright_count),
                    "noise_deviation": abs(normal_count - noisy_count)
                })

        except Exception as e:
            results.append({
                "filename": item["name"],
                "error": str(e)
            })

        progress.progress((i + 1) / len(image_items))

    results_df = pd.DataFrame(results)
    feature_df = pd.DataFrame(feature_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    robustness_df = pd.DataFrame(robustness_rows)

    st.subheader("A. Agreement Table vs Manual Counts")
    st.dataframe(results_df, use_container_width=True)

    valid_counter_df = results_df.dropna(subset=["manual_count", "automatic_count"])

    if len(valid_counter_df) > 1:
        mae = mean_absolute_error(valid_counter_df["manual_count"], valid_counter_df["automatic_count"])
        r2 = r2_score(valid_counter_df["manual_count"], valid_counter_df["automatic_count"])
        mape = valid_counter_df["percentage_error"].dropna().mean()
        estimated_accuracy = max(0, 100 - mape)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", round(mae, 2))
        m2.metric("MAPE (%)", round(mape, 2))
        m3.metric("Estimated Accuracy (%)", round(estimated_accuracy, 2))
        m4.metric("R² Agreement", round(r2, 3))

        fig = px.scatter(
            valid_counter_df,
            x="manual_count",
            y="automatic_count",
            hover_name="filename",
            title="Manual Count vs Automatic CV Count"
        )
        st.plotly_chart(fig, use_container_width=True)

    if run_baseline and len(baseline_df) > 0:
        st.subheader("B. Baseline vs Improved Segmentation")
        st.dataframe(baseline_df, use_container_width=True)

        baseline_valid = baseline_df.dropna(
            subset=[
                "manual_count",
                "baseline_simple_threshold_count",
                "improved_watershed_count",
                "recommended_count"
            ]
        )

        if len(baseline_valid) > 1:
            comparison_table = pd.DataFrame([
                {
                    "Method": "Baseline: Simple Thresholding",
                    "MAE": round(mean_absolute_error(
                        baseline_valid["manual_count"],
                        baseline_valid["baseline_simple_threshold_count"]
                    ), 2),
                    "MAPE (%)": round(baseline_valid["baseline_percentage_error"].dropna().mean(), 2),
                    "R²": round(r2_score(
                        baseline_valid["manual_count"],
                        baseline_valid["baseline_simple_threshold_count"]
                    ), 3)
                },
                {
                    "Method": "Improved: Marker-Controlled Watershed",
                    "MAE": round(mean_absolute_error(
                        baseline_valid["manual_count"],
                        baseline_valid["improved_watershed_count"]
                    ), 2),
                    "MAPE (%)": round(baseline_valid["improved_percentage_error"].dropna().mean(), 2),
                    "R²": round(r2_score(
                        baseline_valid["manual_count"],
                        baseline_valid["improved_watershed_count"]
                    ), 3)
                },
                {
                    "Method": "Selected/Recommended Method",
                    "MAE": round(mean_absolute_error(
                        baseline_valid["manual_count"],
                        baseline_valid["recommended_count"]
                    ), 2),
                    "MAPE (%)": round(baseline_valid["recommended_percentage_error"].dropna().mean(), 2),
                    "R²": round(r2_score(
                        baseline_valid["manual_count"],
                        baseline_valid["recommended_count"]
                    ), 3)
                }
            ])

            st.dataframe(comparison_table, use_container_width=True)

    if run_classifier:
        st.subheader("C. Small Countability ML Classifier")

        if len(feature_df) > 0 and feature_df["label"].nunique() >= 2:
            X = feature_df[FEATURE_COLUMNS]
            y = feature_df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42,
                stratify=y
            )

            clf = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
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

            st.text(classification_report(y_test, y_pred, zero_division=0))
            st.write(confusion_matrix(y_test, y_pred))

            feature_df["ml_predicted_decision"] = clf.predict(feature_df[FEATURE_COLUMNS])
            st.dataframe(feature_df, use_container_width=True)

            model_path = os.path.join(tempfile.mkdtemp(), "countability_random_forest.pkl")
            joblib.dump(clf, model_path)

            with open(model_path, "rb") as f:
                st.download_button(
                    "Download Random Forest Classifier",
                    data=f,
                    file_name="countability_random_forest.pkl",
                    mime="application/octet-stream"
                )
        else:
            st.warning("Classifier needs both classes: countable and not_countable.")

    if run_robustness and len(robustness_df) > 0:
        st.subheader("D. Robustness Analysis")
        st.dataframe(robustness_df, use_container_width=True)

    st.subheader("E. Submission Report")

    st.markdown("""
### Approach
The system uses classical CV for colony counting: image resize, petri masking, CLAHE enhancement, Otsu thresholding, segmentation, and count filtering.

### Baseline
Simple Thresholding + Connected Components.

### Improved Segmentation
Marker-Controlled Watershed separates touching colonies using distance transform and local maxima.

### Recommended Model
For best accuracy: Watershed + LoG + Hough Consensus.  
For 300-image live demo: Fast Recommended Watershed Only.

### ML Classifier
Random Forest is used only for countable/manual-review decision.

### CFU Rule
Only plates with 25–250 colonies are valid for CFU/mL estimation.

### Decision Output
The app provides automatic count, CFU/mL, validity status, and manual-review flag.
""")

    st.subheader("F. Downloads")

    st.download_button(
        "Download Final Results CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="final_colony_counting_results.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Feature Dataset CSV",
        data=feature_df.to_csv(index=False).encode("utf-8"),
        file_name="countability_feature_dataset.csv",
        mime="text/csv"
    )

    if run_baseline and len(baseline_df) > 0:
        st.download_button(
            "Download Baseline Comparison CSV",
            data=baseline_df.to_csv(index=False).encode("utf-8"),
            file_name="baseline_comparison_results.csv",
            mime="text/csv"
        )

    if run_robustness and len(robustness_df) > 0:
        st.download_button(
            "Download Robustness CSV",
            data=robustness_df.to_csv(index=False).encode("utf-8"),
            file_name="robustness_analysis_results.csv",
            mime="text/csv"
        )
