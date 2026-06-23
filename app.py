import os, re, glob, cv2, zipfile, tempfile
import numpy as np
import pandas as pd
import streamlit as st
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px

st.set_page_config(page_title="Automated Bacterial Colony Counter", layout="wide")

st.title("Automated Bacterial Colony Counter")
st.write("Google Drive / Image Upload / ZIP Upload | Best Hough Counter | Baseline Comparison | Robustness | Submission Report")

IMAGE_EXTENSIONS = [
    "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff",
    "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF"
]

# ============================================================
# BASIC FUNCTIONS
# ============================================================

def find_all_images(folder):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(list(set(paths)))


def read_image(path, max_width=1000):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    return img


def read_uploaded_image(uploaded_file, max_width=1000):
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


def extract_manual_count(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


# ============================================================
# IMAGE PROCESSING
# ============================================================

def detect_petri_dish(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 9)
    h, w = gray.shape

    circles = cv2.HoughCircles(
        gray_blur,
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


def illumination_correct(image, plate_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=plate_mask)

    background = cv2.medianBlur(masked, 51)
    corrected = cv2.subtract(masked, background)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=plate_mask)

    return enhanced


def best_hough_petri_counter(image):
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    px, py, pr, plate_mask = detect_petri_dish(image)
    masked = cv2.bitwise_and(gray, gray, mask=plate_mask)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked)

    blur = cv2.GaussianBlur(enhanced, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=18,
        param1=60,
        param2=13,
        minRadius=4,
        maxRadius=18
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


def hough_counter(image, min_radius=4, max_radius=18, min_dist=18, param2=13):
    output = image.copy()
    px, py, pr, plate_mask = detect_petri_dish(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=plate_mask)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked)

    blur = cv2.GaussianBlur(enhanced, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=60,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
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


def simple_threshold_counter(image, min_area=20, max_area=2000):
    output = image.copy()
    px, py, pr, plate_mask = detect_petri_dish(image)

    enhanced = illumination_correct(image, plate_mask)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    _, binary = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean)

    count = 0
    cv2.circle(output, (px, py), pr, (255, 255, 0), 2)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if min_area <= area <= max_area:
            x, y = centroids[i]
            count += 1
            cv2.circle(output, (int(x), int(y)), 7, (255, 0, 0), 2)

    return count, output, binary, plate_mask


def watershed_counter(image, min_area=20, max_area=2000, min_distance=10):
    output = image.copy()
    px, py, pr, plate_mask = detect_petri_dish(image)

    enhanced = illumination_correct(image, plate_mask)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    _, binary = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

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
    labels = watershed(-distance, markers, mask=opening)

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
                    cv2.circle(output, (cX, cY), 7, (0, 255, 0), 2)

    return count, output, binary, plate_mask


def plate_features(image):
    _, _, _, plate_mask = detect_petri_dish(image)
    enhanced = illumination_correct(image, plate_mask)

    _, binary = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)

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

    return {
        "coverage_ratio": float(coverage_ratio),
        "component_count": int(len(areas)),
        "mean_component_area": float(np.mean(areas)),
        "max_component_area": float(np.max(areas)),
        "brightness_mean": float(np.mean(enhanced[plate_mask > 0])),
        "brightness_std": float(np.std(enhanced[plate_mask > 0]))
    }


def countability_decision(features, count, percentage_error=None):
    if count < 1:
        return "manual_review"

    if count > 350:
        return "manual_review"

    if percentage_error is not None and percentage_error > 60:
        return "manual_review"

    if features["coverage_ratio"] > 0.45:
        return "manual_review"

    if features["max_component_area"] > 10000:
        return "manual_review"

    if features["component_count"] < 3:
        return "manual_review"

    return "countable"


def run_counter(
    image,
    method,
    min_area,
    max_area,
    min_distance,
    hough_min_radius,
    hough_max_radius,
    hough_min_dist,
    hough_param2
):
    if method == "Best Hough - Petri Plates":
        return best_hough_petri_counter(image)

    if method == "Watershed":
        return watershed_counter(image, min_area, max_area, min_distance)

    if method == "Hough Circle":
        return hough_counter(
            image,
            min_radius=hough_min_radius,
            max_radius=hough_max_radius,
            min_dist=hough_min_dist,
            param2=hough_param2
        )

    return simple_threshold_counter(image, min_area, max_area)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Image Source")

source = st.sidebar.radio(
    "Choose image source",
    ["Google Drive Folder", "Upload Images", "Upload ZIP"]
)

st.sidebar.header("Counting Settings")

method = st.sidebar.selectbox(
    "Counting Method",
    [
        "Best Hough - Petri Plates",
        "Watershed",
        "Hough Circle",
        "Simple Threshold"
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

st.sidebar.header("Hough Parameters")
hough_min_radius = st.sidebar.slider("Hough minimum radius", 2, 30, 4)
hough_max_radius = st.sidebar.slider("Hough maximum radius", 5, 60, 18)
hough_min_dist = st.sidebar.slider("Hough minimum distance", 5, 60, 18)
hough_param2 = st.sidebar.slider("Hough sensitivity param2", 5, 40, 13)

st.sidebar.header("CFU Settings")
dilution_factor = st.sidebar.number_input("Dilution factor", min_value=1, value=100000)
plated_volume = st.sidebar.number_input("Plated volume in mL", min_value=0.01, value=0.1)

# ============================================================
# IMAGE LOADING
# ============================================================

image_items = []

if source == "Google Drive Folder":
    folder = st.sidebar.text_input(
        "Google Drive Folder Path",
        "/content/drive/MyDrive/Petri_plates"
    )

    if os.path.exists(folder):
        image_paths = find_all_images(folder)[:max_images]

        for path in image_paths:
            image_items.append({
                "name": os.path.basename(path),
                "path": path,
                "type": "path"
            })
    else:
        st.warning("Google Drive folder not found.")

elif source == "Upload Images":
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files[:max_images]:
            try:
                image_items.append({
                    "name": file.name,
                    "image": read_uploaded_image(file),
                    "type": "array"
                })
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")

elif source == "Upload ZIP":
    zip_file = st.sidebar.file_uploader(
        "Upload ZIP file containing images",
        type=["zip"]
    )

    if zip_file:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "uploaded_dataset.zip")

        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(temp_dir)

            image_paths = find_all_images(temp_dir)[:max_images]

            for path in image_paths:
                image_items.append({
                    "name": os.path.basename(path),
                    "path": path,
                    "type": "path"
                })

            st.success(f"{len(image_items)} images extracted from ZIP.")

        except Exception as e:
            st.error(f"ZIP extraction failed: {e}")


# ============================================================
# MAIN PREVIEW
# ============================================================

st.subheader("Dataset Status")

if len(image_items) == 0:
    st.warning("No images loaded yet.")
    st.stop()

st.success(f"Total loaded images: {len(image_items)}")

selected_item = st.selectbox(
    "Preview Image",
    image_items,
    format_func=lambda x: x["name"]
)

if selected_item["type"] == "path":
    image = read_image(selected_item["path"])
else:
    image = selected_item["image"]

manual_count = extract_manual_count(selected_item["name"])

count, output, processed_view, plate_mask = run_counter(
    image,
    method,
    min_area,
    max_area,
    min_distance,
    hough_min_radius,
    hough_max_radius,
    hough_min_dist,
    hough_param2
)

features = plate_features(image)

absolute_error = None
percentage_error = None

if manual_count is not None:
    absolute_error = abs(manual_count - count)
    if manual_count > 0:
        percentage_error = (absolute_error / manual_count) * 100

decision = countability_decision(features, count, percentage_error)

c1, c2, c3 = st.columns(3)

with c1:
    st.image(image, caption=f"Original | Manual: {manual_count}", use_container_width=True)

with c2:
    st.image(plate_mask, caption="Inner Agar Mask", use_container_width=True)

with c3:
    st.image(output, caption=f"{method} Count: {count} | {decision}", use_container_width=True)

preview_df = pd.DataFrame([{
    "filename": selected_item["name"],
    "manual_count": manual_count,
    "automatic_count": count,
    "absolute_error": absolute_error,
    "percentage_error": percentage_error,
    "decision": decision,
    "coverage_ratio": features["coverage_ratio"],
    "component_count": features["component_count"],
    "max_component_area": features["max_component_area"],
    "method": method
}])

st.subheader("Preview Result")
st.dataframe(preview_df, use_container_width=True)

st.divider()

# ============================================================
# BATCH PROCESSING
# ============================================================

if st.button("Process All Images"):

    results = []
    progress = st.progress(0)

    for i, item in enumerate(image_items):

        try:
            if item["type"] == "path":
                img = read_image(item["path"])
            else:
                img = item["image"]

            filename = item["name"]
            manual = extract_manual_count(filename)

            auto_count, output, processed_view, plate_mask = run_counter(
                img,
                method,
                min_area,
                max_area,
                min_distance,
                hough_min_radius,
                hough_max_radius,
                hough_min_dist,
                hough_param2
            )

            features = plate_features(img)

            absolute_error = None
            percentage_error = None

            if manual is not None:
                absolute_error = abs(manual - auto_count)
                if manual > 0:
                    percentage_error = (absolute_error / manual) * 100

            decision = countability_decision(features, auto_count, percentage_error)

            estimated_cfu = None
            if decision == "countable":
                estimated_cfu = (auto_count * dilution_factor) / plated_volume

            results.append({
                "filename": filename,
                "manual_count": manual,
                "automatic_count": auto_count,
                "absolute_error": absolute_error,
                "percentage_error": percentage_error,
                "decision": decision,
                "estimated_CFU_per_mL": estimated_cfu,
                "coverage_ratio": features["coverage_ratio"],
                "component_count": features["component_count"],
                "mean_component_area": features["mean_component_area"],
                "max_component_area": features["max_component_area"],
                "brightness_mean": features["brightness_mean"],
                "brightness_std": features["brightness_std"],
                "method": method
            })

        except Exception as e:
            results.append({
                "filename": item["name"],
                "error": str(e),
                "decision": "error"
            })

        progress.progress((i + 1) / len(image_items))

    results_df = pd.DataFrame(results)

    st.subheader("Batch Colony Counting Results")
    st.dataframe(results_df, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Total Images", len(results_df))
    m2.metric("Countable Plates", (results_df["decision"] == "countable").sum())
    m3.metric("Manual Review", (results_df["decision"] == "manual_review").sum())
    m4.metric("Errors", (results_df["decision"] == "error").sum())

    if "automatic_count" in results_df.columns:
        st.metric("Average Automatic Count", round(results_df["automatic_count"].dropna().mean(), 2))

    valid_df = results_df.dropna(subset=["manual_count", "automatic_count"])

    if len(valid_df) > 1:
        mae = mean_absolute_error(valid_df["manual_count"], valid_df["automatic_count"])
        r2 = r2_score(valid_df["manual_count"], valid_df["automatic_count"])
        mape = valid_df["percentage_error"].dropna().mean()
        estimated_accuracy = max(0, 100 - mape)

        st.subheader("Validation Metrics")

        v1, v2, v3, v4 = st.columns(4)

        v1.metric("MAE", round(mae, 2))
        v2.metric("MAPE (%)", round(mape, 2))
        v3.metric("Estimated Accuracy (%)", round(estimated_accuracy, 2))
        v4.metric("R² Score", round(r2, 3))

        fig = px.scatter(
            valid_df,
            x="manual_count",
            y="automatic_count",
            hover_name="filename",
            title="Manual Count vs Automatic Count"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # BASELINE COMPARISON
    # ============================================================

    st.divider()
    st.header("Baseline Comparison")

    baseline_rows = []
    baseline_progress = st.progress(0)

    for i, item in enumerate(image_items):
        try:
            if item["type"] == "path":
                img = read_image(item["path"])
            else:
                img = item["image"]

            filename = item["name"]
            manual = extract_manual_count(filename)

            baseline_count, _, _, _ = simple_threshold_counter(img, min_area, max_area)
            proposed_count, _, _, _ = best_hough_petri_counter(img)

            baseline_error = abs(manual - baseline_count) if manual is not None else None
            proposed_error = abs(manual - proposed_count) if manual is not None else None

            baseline_percent_error = None
            proposed_percent_error = None

            if manual not in [None, 0]:
                baseline_percent_error = (baseline_error / manual) * 100
                proposed_percent_error = (proposed_error / manual) * 100

            baseline_rows.append({
                "filename": filename,
                "manual_count": manual,
                "baseline_simple_threshold_count": baseline_count,
                "proposed_best_hough_count": proposed_count,
                "baseline_absolute_error": baseline_error,
                "proposed_absolute_error": proposed_error,
                "baseline_percentage_error": baseline_percent_error,
                "proposed_percentage_error": proposed_percent_error
            })

        except Exception as e:
            baseline_rows.append({
                "filename": item["name"],
                "error": str(e)
            })

        baseline_progress.progress((i + 1) / len(image_items))

    baseline_df = pd.DataFrame(baseline_rows)
    st.dataframe(baseline_df, use_container_width=True)

    baseline_valid = baseline_df.dropna(
        subset=[
            "manual_count",
            "baseline_simple_threshold_count",
            "proposed_best_hough_count"
        ]
    )

    if len(baseline_valid) > 1:
        baseline_mae = mean_absolute_error(
            baseline_valid["manual_count"],
            baseline_valid["baseline_simple_threshold_count"]
        )

        proposed_mae = mean_absolute_error(
            baseline_valid["manual_count"],
            baseline_valid["proposed_best_hough_count"]
        )

        baseline_mape = baseline_valid["baseline_percentage_error"].dropna().mean()
        proposed_mape = baseline_valid["proposed_percentage_error"].dropna().mean()

        baseline_r2 = r2_score(
            baseline_valid["manual_count"],
            baseline_valid["baseline_simple_threshold_count"]
        )

        proposed_r2 = r2_score(
            baseline_valid["manual_count"],
            baseline_valid["proposed_best_hough_count"]
        )

        comparison_table = pd.DataFrame([
            {
                "Method": "Baseline: Simple Threshold + Connected Components",
                "MAE": round(baseline_mae, 2),
                "MAPE (%)": round(baseline_mape, 2),
                "Estimated Accuracy (%)": round(max(0, 100 - baseline_mape), 2),
                "R² Score": round(baseline_r2, 3)
            },
            {
                "Method": "Proposed: Best Hough - Petri Plates",
                "MAE": round(proposed_mae, 2),
                "MAPE (%)": round(proposed_mape, 2),
                "Estimated Accuracy (%)": round(max(0, 100 - proposed_mape), 2),
                "R² Score": round(proposed_r2, 3)
            }
        ])

        st.subheader("Measured Results: Accuracy / Error Table")
        st.dataframe(comparison_table, use_container_width=True)

        fig2 = px.bar(
            comparison_table,
            x="Method",
            y=["MAE", "MAPE (%)"],
            barmode="group",
            title="Baseline vs Proposed Error Comparison"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ============================================================
    # ROBUSTNESS ANALYSIS
    # ============================================================

    st.divider()
    st.header("Robustness Analysis")

    robustness_rows = []

    for item in image_items[:min(50, len(image_items))]:
        try:
            if item["type"] == "path":
                img = read_image(item["path"])
            else:
                img = item["image"]

            filename = item["name"]
            manual = extract_manual_count(filename)

            normal_count, _, _, _ = best_hough_petri_counter(img)

            dark_img = np.clip(img * 0.65, 0, 255).astype(np.uint8)
            bright_img = np.clip(img * 1.25, 0, 255).astype(np.uint8)

            noise = np.random.normal(0, 15, img.shape)
            noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)

            dark_count, _, _, _ = best_hough_petri_counter(dark_img)
            bright_count, _, _, _ = best_hough_petri_counter(bright_img)
            noisy_count, _, _, _ = best_hough_petri_counter(noisy_img)

            robustness_rows.append({
                "filename": filename,
                "manual_count": manual,
                "normal_count": normal_count,
                "dark_count": dark_count,
                "bright_count": bright_count,
                "noisy_count": noisy_count,
                "dark_deviation": abs(normal_count - dark_count),
                "bright_deviation": abs(normal_count - bright_count),
                "noise_deviation": abs(normal_count - noisy_count)
            })

        except Exception as e:
            robustness_rows.append({
                "filename": item["name"],
                "error": str(e)
            })

    robustness_df = pd.DataFrame(robustness_rows)

    st.subheader("Robustness Table")
    st.dataframe(robustness_df, use_container_width=True)

    if len(robustness_df.dropna()) > 0:
        robustness_summary = pd.DataFrame([
            {
                "Condition": "Dark Image",
                "Average Count Deviation": round(robustness_df["dark_deviation"].dropna().mean(), 2)
            },
            {
                "Condition": "Bright Image",
                "Average Count Deviation": round(robustness_df["bright_deviation"].dropna().mean(), 2)
            },
            {
                "Condition": "Gaussian Noise",
                "Average Count Deviation": round(robustness_df["noise_deviation"].dropna().mean(), 2)
            }
        ])

        st.subheader("Robustness Summary")
        st.dataframe(robustness_summary, use_container_width=True)

    # ============================================================
    # SUBMISSION REPORT
    # ============================================================

    st.divider()
    st.header("Submission Report")

    st.subheader("1. Approach")
    st.markdown("""
    The system accepts Petri plate images from Google Drive, direct image upload, or ZIP upload.
    The image is resized and converted into grayscale. A Petri dish is detected using Hough Circle
    Transform. An inner agar mask is generated to remove the dish rim and external background.
    CLAHE contrast enhancement and Gaussian smoothing are applied before colony detection.
    The proposed method uses Hough circle detection inside the agar mask with duplicate rejection.
    """)

    st.subheader("2. Baseline")
    st.markdown("""
    The baseline method is Simple Thresholding with Connected Components. It applies grayscale
    conversion, illumination correction, Otsu thresholding, morphological opening, and counts each
    valid connected component as a colony.
    """)

    st.subheader("3. Proposed Method")
    st.markdown("""
    The proposed method is **Best Hough - Petri Plates**. It detects the Petri dish, creates an
    inner agar mask, enhances contrast using CLAHE, applies Gaussian smoothing, detects circular
    colonies using Hough Circle Transform, and removes duplicate detections.
    """)

    st.subheader("4. Measured Results")
    st.markdown("""
    The measured results include manual count from filename, automatic count, absolute error,
    percentage error, estimated accuracy, and R² agreement.
    """)

    if len(baseline_valid) > 1:
        st.dataframe(comparison_table, use_container_width=True)

    st.subheader("5. Robustness")
    st.markdown("""
    Robustness was evaluated by testing the proposed method under three perturbations:
    darkened image, brightened image, and Gaussian noise. The average count deviation was used
    to measure stability.
    """)

    if "robustness_summary" in locals():
        st.dataframe(robustness_summary, use_container_width=True)

    st.subheader("6. Limitations")
    st.markdown("""
    1. Very dense or confluent colonies may not be counted accurately.
    2. Overlapping colonies can be under-counted.
    3. Strong reflection or poor illumination can affect detection.
    4. Manual counts extracted from filenames may contain labeling errors.
    5. The current method assumes a near-circular Petri dish.
    6. Very small or irregularly shaped colonies may be missed by Hough detection.
    """)

    st.subheader("7. Conclusion")
    st.markdown("""
    The proposed system provides a complete automated bacterial colony counting pipeline with
    baseline comparison, measured error analysis, robustness evaluation, countability decision,
    and downloadable reports. The Best Hough - Petri Plates method is recommended for this dataset.
    """)

    # ============================================================
    # DOWNLOADS
    # ============================================================

    st.divider()
    st.header("Downloads")

    csv = results_df.to_csv(index=False).encode("utf-8")
    baseline_csv = baseline_df.to_csv(index=False).encode("utf-8")
    robustness_csv = robustness_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Final Results CSV",
        data=csv,
        file_name="final_colony_counting_results.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Baseline Comparison CSV",
        data=baseline_csv,
        file_name="baseline_comparison_results.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Robustness Analysis CSV",
        data=robustness_csv,
        file_name="robustness_analysis_results.csv",
        mime="text/csv"
    )
