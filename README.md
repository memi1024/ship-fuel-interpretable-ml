# Ship Fuel Interpretable ML

Implementation of the data preparation, predictive validation, physical-plausibility audit, CII-proxy analysis, and paired ERA5 harmonisation control used in the study.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Workflow

### 1. Prepare ERA5

```bash
python 01_prepare_era5.py \
  --input /path/to/containership.xlsx \
  --output-dir /path/to/era5/raw
```

### 2. Interpolate ERA5 to containership observations

```bash
python 02_interpolate_era5.py \
  --input /path/to/containership.xlsx \
  --era5-dir /path/to/era5/raw \
  --output /path/to/containership_with_era5.xlsx
```

### 3. Clean and construct Fixed31

```bash
python 03_clean_data.py \
  --container-interpolated /path/to/containership_with_era5.xlsx \
  --bulk /path/to/bulk.csv \
  --tanker /path/to/tanker.csv \
  --work-root /path/to/work \
  --paperdata-root /path/to/paperdata
```

### 4. Run the manuscript analysis

```bash
python 04_reproduce_paper.py \
  --cleaning-root /path/to/paperdata/05_clean23/all_phase_equal_cleaning \
  --output-dir /path/to/analysis_output
```

Stage 4 runs:

- seven-model record-level comparison and feature ablation;
- operating-regime and ship-type robustness;
- L1 record-level interpolation, L2 temporal extrapolation, L3 LOVO transfer, and L4 target-vessel adaptation;
- SHAP, GAM/LOWESS, local perturbation, rudder, and explanation-stability analyses;
- observation-period main-engine CII-proxy aggregation;
- fixed-time and explicit fixed-distance 5/10/15% speed-reduction analyses;
- H1 vessel-cluster bootstrap and paired sign-flip inference;
- H3 vessel-bootstrap endpoint inference, ship-type contrasts, and interaction tests;
- full-support wave localisation including head/cross/following and tanker × head-sea analyses;
- pooled and vessel-balanced rudder estimands;
- manuscript figures.

The model workflow uses the fixed hyperparameter registry in `src/paper_analysis/02_best_hyperparameters.csv`.

### 5. Run the paired bulk-carrier ERA5 harmonisation control

```bash
python 05_bulk_era5_control.py \
  --fixed31-cruise /path/to/final_fixed31_cruise.csv \
  --bulk-harmonized-csv /path/to/bulk_model_10min_features_harmonized_era5.csv \
  --main-output /path/to/analysis_output \
  --output-dir /path/to/harmonisation_output
```

## Repository layout

```text
.
├── 01_prepare_era5.py
├── 02_interpolate_era5.py
├── 03_clean_data.py
├── 04_reproduce_paper.py
├── 05_bulk_era5_control.py
├── validate_installation.py
├── src/
│   ├── cleaning/
│   ├── environment/
│   ├── paper_analysis/
│   └── plotting/
├── tests/
├── requirements.txt
└── .gitignore
```

Input datasets are supplied through the command-line paths shown above.
