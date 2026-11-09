# Experimental Run Consolidation & Registry

This document establishes the canonical cross-reference for all physical experimental specimens, 3D scans, and toolpath files in the repository.

---

## 1. Why Run Numbers Were Colliding

There were **two distinct experimental campaigns** conducted at different times, on different sheet geometries, which independently used overlapping labels ("Run 1" and "Run 3"):

1. **Calibration Campaign (`Cal-Run 1`, `Cal-Run 2`, `Cal-Run 3`) — $200 \times 300 \times 0.77\text{ mm}$ Sheets:**
   - Conducted on 0.77 mm thick aluminum sheets.
   - Machine log recorded in `calibration/data/raw/Test Ran.xlsx` as Machine Runs 6, 7, 9, 10.
   - For publication and Bayesian calibration, `calibration/case_labels.py` aliased:
     - Machine Run 6 $\to$ **`Cal-Run 1`** (Primary calibration case)
     - Machine Run 9 $\to$ **`Cal-Run 2`** (Transfer validation case)
     - Machine Run 10 $\to$ **`Cal-Run 3`** (Joint calibration case)
     - Machine Run 7 was discarded/removed from repository (twisted specimen, unused in calibration).

2. **Validation Campaign (`Val-Run 1`, `Val-Run 2`) — $220 \times 124 \times 0.61\text{ mm}$ Sheets:**
   - Fabricated by collaborator (Hyunwoo) on 0.61 mm thick sheets for blind forward prediction.
   - Formerly called "Hyunwoo Run 1" and "Run 3":
     - **`Val-Run 1`**: 24 passes across 3 regions, front + back (turned over, saddle shape).
     - **`Val-Run 2`**: 30 passes on central half-width patch, front face only (trough/dimple shape).

---

## 2. Complete Inventory & Consolidation Matrix

| Canonical Label | Campaign | Sheet Size & Thickness | Raw Scan File | Toolpath File | Toolpath Specs | Physical Process & Role | Historical / Existing Aliases |
|---|---|---|---|---|---|---|---|
| **`Cal-Run 1`** | Calibration | $200 \times 300 \times 0.77\text{ mm}$ | `calibration/data/raw/run6.stl` | `calibration/data/raw/run6_zigzag_path.csv` | 29 passes, $\alpha = 1.0^\circ$, $lv = 130\text{ mm}$, 68 strips | Primary Attempt 209 calibration case | `run6`, Public "Run 1", `Run_6.stl` |
| **`Cal-Run 2`** | Calibration | $200 \times 300 \times 0.77\text{ mm}$ | `calibration/data/raw/Run_9.stl` | `calibration/data/raw/run9_zigzag_path.csv` | 33 passes, $\alpha = 1.0^\circ$, $lv = 187.5\text{ mm}$, 32 strips | Transfer & Bayesian consistency validation | `run9`, Public "Run 2", `Run_9.stl` |
| **`Cal-Run 3`** | Calibration | $200 \times 300 \times 0.77\text{ mm}$ | `calibration/data/raw/Run_10.stl` | `calibration/data/raw/run10_zigzag_path.csv` | 50 passes, $\alpha = 1.5^\circ$, $lv = 175\text{ mm}$, 24 strips | Joint Bayesian calibration & cross-transfer | `run10`, Public "Run 3", `Run_10.stl` |
| **`Val-Run 1`** | Validation | $220 \times 124 \times 0.61\text{ mm}$ | `shape_design/data/Hyunwoo_Run_1_24_Passes.ply` | `shape_design/data/Wheeling_Toolpath_24_Passes_Consolidated.csv` | 24 passes, $\alpha = 2.75^\circ$, $lv = 104\text{ mm}$, 3 regions, **front + back (turned over)** | Blind forward prediction of double-sided wheeling (saddle shape, 0.37 mm RMS) | `hw_run1`, "Hyunwoo Run 1", Card `REPLICATE-HW1.md` |
| **`Val-Run 2`** | Validation | $220 \times 124 \times 0.61\text{ mm}$ | `shape_design/data/Run_3.ply` | `shape_design/data/Run3_toolpath.csv` | 30 passes, $\alpha = 5.00^\circ$, $lv = 55\text{ mm}$, central half-width patch, **front face only** | Blind forward prediction of localized single-sided wheeling (trough/dimple shape, 0.53 mm RMS) | `hw_run3`, "Run 3", `Run_3.ply`, Card `REPLICATE-HW3.md` |

---

## 3. Official Policy (Researcher Decision 2026-11-09)

1. **Calibration Cases:**
   - Refer to the 0.77 mm calibration series as **`Cal-Run 1`**, **`Cal-Run 2`**, **`Cal-Run 3`** in text, figures, and discussions.
   - In code and configuration files, immutable internal keys `run6`, `run9`, `run10` remain untouched to preserve dataset hashes and evaluation caches.
2. **Validation Cases:**
   - Refer to the 0.61 mm collaborator validation series as **`Val-Run 1`** (formerly Hyunwoo Run 1) and **`Val-Run 2`** (formerly Run 3).
   - In directories and scripts, `hw_run1` and `hw_run3` serve as the back-end folder keys (`run/shape_design/hw_run1/` and `run/shape_design/hw_run3/`).
