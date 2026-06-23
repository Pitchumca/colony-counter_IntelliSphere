import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

st.set_page_config(
    page_title="Automated Bacterial Colony Counter",
    layout="wide"
)

st.title("Automated Bacterial Colony Counter")
st.write("Live batch prototype: upload multiple petri-dish images and get colony count, decision, and CFU estimate.")

# ============================================================
# PETRI DISH DETECTION
# ============================================================

def detect_petri_dish(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    x, y = w // 2, h // 2
    r = int(min(h, w) * 0.42)
    inner_r = int(r * 0.80)

    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (x, y), inner_r, 255, -1)

    return x, y, inner_r, mask


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(masked)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return blur


# ============================================================
# BASELINE SIMPLE THRESHOLDING COUNTER
# ============================================================

def baseline_counter(image):
    output = image.copy()

    px, py, pr, mask = detect_petri_dish(image)
    blur = preprocess(image, mask)

    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    binary = cv2.bitwise_and(binary, binary, mask=mask)

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

        if 20 <= area <= 2500:
            x, y = centroids[i]
            count += 1
            cv2.circle(output, (int(x), int(y)), 8, (255, 0, 0), 2)

    return count, output, binary


# ============================================================
# IMPROVED WATERSHED COUNTER
# ============================================================

def watershed_counter(image):
    output = image.copy()

    px, py, pr, mask = detect_petri_dish(image)
    blur = preprocess(image, mask)

    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    binary = cv2.bitwise_and(binary, binary, mask=mask)

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    coords = peak_local_max(
        distance,
        min_distance=10,
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

        component = np.zeros(binary.shape, dtype=np.uint8)
        component[labels == label] = 255

        area = cv2.countNonZero(component)

        if 20 <= area <= 2500:
            moments = cv2.moments(component)

            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])

                if mask[cY, cX] == 255:
                    count += 1
                    cv2.circle(output, (cX, cY), 8, (0, 255, 0), 2)

    return count, output, binary


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def plate_features(image):
    _, _, _, mask = detect_petri_dish(image)
    blur = preprocess(image, mask)

    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    binary = cv2.bitwise_and(binary, binary, mask=mask)

    plate_area = np.sum(mask > 0)
    colony_area = np.sum(binary > 0)

    coverage_ratio = colony_area / plate_area if plate_area > 0 else 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    areas = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > 10:
            areas.append(area)

    if len(areas) == 0:
        areas = [0]

    return {
        "coverage_ratio": float(coverage_ratio),
        "component_count": int(len(areas)),
        "max_component_area": int(max(areas))
    }


# ============================================================
# COUNTABLE / MANUAL REVIEW DECISION
# ============================================================

def countability_decision(features, count):
    if features["coverage_ratio"] > 0.45:
        return "Manual Review"

    if features["max_component_area"] > 10000:
        return "Manual Review"

    if count < 5:
        return "Manual Review"

    if count > 350:
        return "Manual Review"

    return "Countable"


# ============================================================
# STREAMLIT SIDEBAR
# ============================================================

st.sidebar.header("CFU Settings")

dilution_factor = st.sidebar.number_input(
    "Dilution factor",
    value=100000,
    step=1000
)

plated_volume = st.sidebar.number_input(
    "Plated volume in mL",
    value=0.1,
    step=0.1
)

st.sidebar.info(
    "CFU/mL = Colony Count × Dilution Factor / Plated Volume"
)


# ============================================================
# MULTI-IMAGE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload one or more petri-dish images",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

if uploaded_files:

    all_results = []

    st.subheader("Image-wise Results")

    for uploaded_file in uploaded_files:

        pil_img = Image.open(uploaded_file).convert("RGB")
        image = np.array(pil_img)

        baseline_count, baseline_output, baseline_binary = baseline_counter(image)
        watershed_count, watershed_output, watershed_binary = watershed_counter(image)

        features = plate_features(image)

        decision = countability_decision(
            features,
            watershed_count
        )

        if decision == "Countable":
            estimated_cfu = (watershed_count * dilution_factor) / plated_volume
        else:
            estimated_cfu = None

        all_results.append({
            "Filename": uploaded_file.name,
            "Baseline Count": baseline_count,
            "Watershed Count": watershed_count,
            "Decision": decision,
            "Coverage Ratio": round(features["coverage_ratio"], 4),
            "Component Count": features["component_count"],
            "Max Component Area": features["max_component_area"],
            "Estimated CFU/mL": estimated_cfu
        })

        with st.expander(f"Result: {uploaded_file.name}"):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    image,
                    caption="Original Image",
                    use_container_width=True
                )

            with col2:
                st.image(
                    baseline_output,
                    caption=f"Baseline Count: {baseline_count}",
                    use_container_width=True
                )

            with col3:
                st.image(
                    watershed_output,
                    caption=f"Watershed Count: {watershed_count}",
                    use_container_width=True
                )

            if decision == "Countable":
                st.success(
                    f"Decision: Countable | Estimated CFU/mL: {estimated_cfu:,.2f}"
                )
            else:
                st.warning(
                    "Decision: Manual Review Required"
                )

    results_df = pd.DataFrame(all_results)

    st.subheader("Batch Summary Table")
    st.dataframe(
        results_df,
        use_container_width=True
    )

    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Batch Results CSV",
        data=csv,
        file_name="batch_colony_counting_results.csv",
        mime="text/csv"
    )

else:
    st.info("Upload one or more petri-dish images to start live counting.")
