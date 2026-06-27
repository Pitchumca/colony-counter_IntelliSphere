
# 🧫 Automated Bacterial Colony Counter
**AVIT Faculty Hackathon 2026 — Project 07**  

**Classical Computer Vision Colony Counter + Random Forest Countability Classifier**
CSE × Biotechnology Interdisciplinary Team
[![Open in Streamlit]([https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://colony-counterintellisphere-3ftifsae7c8z9urjof27qj.streamlit.app/)


> **Developed for the AVIT Faculty Hackathon 2026**

A research-oriented application developed as part of the **AVIT Faculty Hackathon 2026**, demonstrating the use of **classical computer vision** and **machine learning** for automated bacterial colony counting and laboratory validation. The system performs deterministic colony counting using robust image processing techniques, while a lightweight Random Forest classifier determines whether a plate is suitable for automatic counting or should be referred for manual review.

---
# 🧫 Automated Bacterial Colony Counter

> **Developed for the AVIT Faculty Hackathon 2026**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Classical%20Computer%20Vision-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange)
![License](https://img.shields.io/badge/License-MIT-success)

---

# 🏆 AVIT Faculty Hackathon 2026

## Event Information

| Item                     | Details                                                          |
| ------------------------ | ---------------------------------------------------------------- |
| **Event**                | AVIT Faculty Hackathon 2026                                      |
| **Institution**          | Aarupadai Veedu Institute of Technology (AVIT)                   |
| **University**           | Vinayaka Mission's Research Foundation (Deemed to be University) |
| **Department**           | Department of Computer Science and Engineering                   |
| **Problem Statement**    | Automated Bacterial Colony Counter                               |
| **Application Domain**   | Artificial Intelligence for Biotechnology                        |
| **Technology Stack**     | Classical Computer Vision + Machine Learning                     |
| **Platform**             | Streamlit                                                        |
| **Programming Language** | Python                                                           |

---

# 📖 Project Overview

Manual bacterial colony counting is a routine laboratory procedure used in microbiology, biotechnology, food safety, environmental monitoring, pharmaceutical quality control, and clinical diagnostics. Traditionally, microbiologists visually inspect petri dishes and manually count bacterial colonies. Although simple, this process is:

* Time consuming
* Labour intensive
* Subjective
* Prone to human error
* Difficult to reproduce consistently across different operators

The challenge becomes even greater when colonies overlap, lighting conditions vary, or images contain reflections and background artifacts.

To address these limitations, this project presents an **Automated Bacterial Colony Counter** based primarily on **Classical Computer Vision (CV)**. Unlike many recent approaches that rely entirely on deep learning, this solution performs colony counting using deterministic image processing techniques that require **no training data** for the counting task.

The application further integrates a lightweight **Random Forest Machine Learning classifier** whose sole purpose is to determine whether a plate should be automatically counted or flagged for manual review.

This design satisfies the hackathon requirement of separating deterministic colony counting from machine learning-based plate assessment.

---

# 🎯 Problem Statement

Develop an intelligent application capable of automatically counting bacterial colonies from petri-dish images using **Classical Computer Vision** techniques while simultaneously predicting whether a plate is suitable for automatic counting or requires manual inspection.

The solution must:

* Count bacterial colonies accurately
* Separate touching colonies
* Handle lighting variations
* Work on real laboratory images
* Compare baseline and improved segmentation techniques
* Validate results against manual laboratory counts
* Use Machine Learning only for plate-level decision making
* Produce quantitative evaluation metrics
* Provide an easy-to-use web application

---

# 🚀 Objectives

The major objectives of this project are:

### Primary Objectives

* Automatically detect petri dishes.
* Remove background and unwanted reflections.
* Enhance image quality.
* Segment bacterial colonies accurately.
* Separate touching colonies using Marker-Controlled Watershed.
* Count colonies automatically.
* Estimate CFU values.
* Compare automated counts with laboratory manual counts.

### Secondary Objectives

* Develop a Random Forest classifier for plate-level countability prediction.
* Compare baseline and improved segmentation algorithms.
* Evaluate robustness under different illumination conditions.
* Generate downloadable validation reports.
* Develop an interactive Streamlit application suitable for laboratory use.

---

# 💡 Key Features

## Image Handling

* Single image upload
* Multiple image upload
* ZIP upload
* Large ZIP processing
* Google Colab compatibility
* Automatic ZIP indexing

---

## Classical Computer Vision

* Automatic petri plate localization
* Agar masking
* Reflection removal
* Background subtraction
* CLAHE enhancement
* Adaptive thresholding
* Otsu thresholding
* Morphological filtering
* Distance transform
* Marker-Controlled Watershed
* Connected Components baseline
* Laplacian of Gaussian detector
* Consensus colony counting
* Colony feature extraction
* Duplicate suppression

---

## Machine Learning

Random Forest classifier for

* Countable
* Too Few To Count (TFTC)
* Too Numerous To Count (TNTC)
* Manual Review

---

## Validation

* Manual count comparison
* MAE
* MAPE
* Accuracy
* R² Score
* Absolute Error
* Percentage Error

---

## Baseline Comparison

The application compares four independent counting approaches.

| Method                      | Purpose             |
| --------------------------- | ------------------- |
| Connected Components        | Baseline Method     |
| Marker-Controlled Watershed | Improved Method     |
| Laplacian of Gaussian       | Blob Detection      |
| Consensus Counter           | Combined Prediction |

---

## Robustness Evaluation

The application automatically evaluates the effect of

* Low illumination
* High illumination
* Image noise

and reports deviations in colony counts.

---

# 🔬 System Workflow

```text
Petri Plate Image
        │
        ▼
Petri Plate Detection
        │
        ▼
Agar Mask Generation
        │
        ▼
Background Removal
        │
        ▼
Contrast Enhancement (CLAHE)
        │
        ▼
Thresholding
        │
        ▼
Morphological Cleaning
        │
        ▼
Distance Transform
        │
        ▼
Marker Generation
        │
        ▼
Marker-Controlled Watershed
        │
        ▼
Colony Filtering
        │
        ▼
Duplicate Removal
        │
        ▼
Automatic Colony Count
        │
        ▼
Validation Against Manual Count
        │
        ├────────► Performance Metrics
        │
        └────────► Random Forest
                    Countable /
                    Manual Review
```

---

# 📊 Performance Evaluation

The system provides comprehensive evaluation including:

### Colony Counting

* Mean Absolute Error (MAE)
* Mean Absolute Percentage Error (MAPE)
* Count Accuracy
* R² Score

### Classification

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

### Robustness

* Dark image analysis
* Bright image analysis
* Noisy image analysis

---

# 🌟 Novel Contributions

* Deterministic colony counting without deep learning.
* Marker-Controlled Watershed segmentation for separating touching colonies.
* Random Forest used exclusively for plate-level decision making.
* Automatic validation against laboratory counts.
* Baseline comparison using multiple counting algorithms.
* Robustness evaluation under varying imaging conditions.
* Interactive Streamlit dashboard with parameter tuning.
* Large-scale batch processing of laboratory datasets.
* Downloadable validation reports for research and publication.

---

# 🎯 Expected Applications

* Microbiology laboratories
* Biotechnology research
* Food microbiology
* Pharmaceutical quality control
* Clinical microbiology
* Environmental monitoring
* Academic research laboratories
* Educational demonstrations
* AI-assisted laboratory automation

---

# 👩‍💻 Developed For

**AVIT Faculty Hackathon 2026**
## 👥 Team Members and Responsibilities

| Team Member                                                                                      | Role                                               | Responsibilities                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------ | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dr. S. Pitchumani Angayarkanni**<br>Professor – Department of Computer Science and Engineering | **Team Leader, AI & Computer Vision Architect**    | • Conceived the project and designed the overall system architecture.<br>• Developed the complete classical computer vision pipeline for automated bacterial colony counting.<br>• Designed and implemented Petri plate detection, image preprocessing, marker-controlled watershed segmentation, colony filtering, and consensus counting algorithms.<br>• Developed the Random Forest-based countable vs. manual-review classifier and validation framework.<br>• Led experimental design, performance evaluation, robustness analysis, technical documentation, GitHub repository preparation, and project presentation. |
| **Mr. V. Dhaneshkumar**<br>Assistant Professor – Department of Computer Science and Engineering  | **Software Development & Deployment Lead**         | • Developed the Streamlit-based interactive web application.<br>• Implemented image upload, ZIP processing, and large dataset handling modules.<br>• Integrated computer vision algorithms with the user interface.<br>• Designed visualization dashboards, CSV report generation, parameter tuning interface, and deployment workflow.<br>• Assisted in software testing, optimization, debugging, and cloud deployment.                                                                                                                                                                                                   |
| **Mr. P. Nayan**<br>Assistant Professor – Department of Computer Science and Engineering         | **Data Engineering & Performance Evaluation Lead** | • Prepared and organized laboratory image datasets for experimentation.<br>• Developed validation datasets and managed manual count comparisons.<br>• Performed baseline comparisons between Connected Components, Marker-Controlled Watershed, Laplacian of Gaussian, and Consensus counting methods.<br>• Generated quantitative evaluation metrics including MAE, MAPE, Accuracy, R², Precision, Recall, and F1-score.<br>• Contributed to robustness analysis under varying illumination and noisy image conditions.                                                                                                    |
| **Dr. B. S. Seshadri**<br>Professor & Head – Department of Computer Science and Engineering      | **Technical Mentor & Research Advisor**            | • Provided technical guidance on system architecture, computer vision methodology, and machine learning integration.<br>• Reviewed the research methodology, experimental design, validation strategy, and statistical evaluation framework.<br>• Advised on algorithm optimization, software engineering practices, and research documentation.<br>• Reviewed the technical report, GitHub documentation, and overall project quality to ensure alignment with hackathon objectives.                                                                                                                                       |

---

## 🧫 Domain Expert Validation

| Domain Expert                                                                                                       | Role                                        | Responsibilities                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Dr. S. Kalaivani Priyadarshini**<br>Associate Professor – Department of Biotechnology, Lady Doak College, Madurai | **Domain Expert & Microbiology Validation** | • Validated the microbiological relevance of the proposed colony counting workflow.<br>• Reviewed Petri plate image interpretation, colony identification, and laboratory counting methodology.<br>• Verified the countable (25–250 CFU) and non-countable (TFTC/TNTC) classification criteria based on standard microbiological practices.<br>• Evaluated the validation methodology, performance metrics, and practical applicability of the system for microbiology and biotechnology laboratories.<br>• Provided expert feedback to improve the biological accuracy and laboratory usability of the developed application. |


**Aarupadai Veedu Institute of Technology (AVIT)**

**Vinayaka Mission's Research Foundation (Deemed to be University)**

**Department of Computer Science and Engineering**
