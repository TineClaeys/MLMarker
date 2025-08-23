# MLMarker

MLMarker is a Python package for tissue-specific proteomics prediction using machine learning, with integrated SHAP-based explainability features.

## Key Features

- **Dual Model Support**: Binary and quantitative tissue prediction models
- **SHAP-Based Predictions**: Uses SHAP values for more interpretable predictions
- **Feature Penalty System**: Adjustable penalty for absent features using `penalty_factor`
- **Visualization Tools**: Force plots, radar charts, and custom visualizations
- **Protein Analysis**: Integrated tools for NSAF calculation and protein information retrieval
- **Data Validation**: Automatic handling of missing features

## Installation

```bash
pip install mlmarker
```

## Quick Start

```python
import pandas as pd
from mlmarker import MLMarker

# Load your data
df = pd.read_csv("your_sample.csv")

# Initialize model (binary=False for quantitative model)
model = MLMarker(binary=False, penalty_factor=1)

# Load and validate your sample
model.load_sample(df)

# Get predictions
predictions = model.predict_top_tissues_shap(n_preds=5)
```

## Core Features

### 1. Model Initialization

```python
# Binary model
binary_model = MLMarker(binary=True)

# Quantitative model with penalty for absent features
quant_model = MLMarker(binary=False, penalty_factor=1)
```

### 2. SHAP-Based Predictions

```python
# Get predictions with SHAP explanations
predictions = model.predict_top_tissues_shap(n_preds=5)

# Visualize SHAP force plot
model.shap_force_plot(n_preds=3)

# Generate radar chart of predictions
model.radar_chart()
```

### 3. SHAP Value Analysis

```python
# Get raw SHAP values
shap_values = model.explainability.calculate_shap()

# Get processed SHAP values with optional penalty
shap_df = model.explainability.get_shap_values(n_preds=5)
```

### 4. Feature Handling

```python
# Get model features
features = model.get_model_features()

# Load sample with feature validation
added_features = model.load_sample(df, output_added_features=True)
```

### 5. NSAF Calculations

```python
# Calculate NSAF scores for proteins
nsaf_df = model.explainability.calculate_NSAF(protein_df, lengths_df)
```

## Advanced Usage

### Penalty Factor

The `penalty_factor` parameter controls how absent features influence predictions:
- `0`: No penalty (default)
- `1`: Full penalty for absent features
- Values between 0-1: Partial penalty

```python
# Model with full penalty for absent features
model = MLMarker(penalty_factor=1)
```

### Custom SHAP Visualization

```python
# Visualize specific tissue
model.shap_force_plot(tissue_name="Liver")

# Visualize top N predictions
model.shap_force_plot(n_preds=3)
```

## Additional Utilities

```python
from mlmarker.utils import (
    get_protein_info,
    get_hpa_info,
    get_go_enrichment,
    visualise_custom_plot
)

# Get protein information
protein_info = get_protein_info("P12345")

# Get Human Protein Atlas information
hpa_info = get_hpa_info("P12345")

# Perform GO enrichment analysis
enrichment = get_go_enrichment(protein_list)
```

## Requirements

- Python ≥ 3.8
- numpy==1.23.5
- pandas
- scikit-learn
- shap==0.42.0
- plotly
- bioservices
- gprofiler-official

