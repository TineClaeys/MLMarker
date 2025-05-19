
# MLMarker

**MLMarker** is a machine learning framework designed for tissue-specific proteomics classification. It leverages state-of-the-art machine learning models to predict tissue origins based on protein expression data, with built-in explainability features for interpreting model predictions using SHAP values.

## Features

- **Model Training & Prediction**: Trains machine learning models to predict tissue types based on protein expression data.
- **SHAP-Based Explainability**: Explains model predictions with SHAP values and provides visualizations like force plots and radar charts.
- **Data Validation & Preprocessing**: Validates input data and ensures it's properly formatted for model input.
- **Scalable**: Designed to handle large datasets with multiple features (proteins) and tissues.
- **Adjustable Interpretability**: Penalize absent features when explaining predictions using the `penalty_factor`.

## Installation

To install **MLMarker**, use the following command:

```bash
pip install mlmarker
```

To install development requirements:

```bash
git clone https://github.com/your-repository/mlmarker.git
cd mlmarker
pip install -r requirements.txt
```

## Usage

### 1. Import and Initialize

```python
from mlmarker.model import MLMarker
import pandas as pd

df = pd.read_csv("sample_input.csv")
ml = MLMarker()
ml.load_sample(df)
```

> Set `binary=True` for binary models, and use `penalty_factor=0` or `1` to control interpretability of absent features.

### 2. Inspect Model Info

```python
ml.get_model_features()
ml.get_model_classes()
```

### 3. Predict Tissue Types

```python
predictions = ml.predict_top_tissues(n_preds=5)
for tissue, prob in predictions:
    print(f"{tissue}: {prob}")
```

### 4. SHAP Explanations

```python
shap_values = ml.calculate_shap()
```

### 5. Visualizations
#### Radar Chart

```python
ml.radar_chart()
```

### 6. Adjusted SHAP Explanations

```python
shap_df = ml.explainability.adjusted_absent_shap_values_df(n_preds=5)
```

This function penalizes SHAP values for proteins with zero intensity in the sample but that still contribute to classification. Use the `penalty_factor` parameter (recommended: `0` or `1`) to determine how strongly to penalize those features:
- `0`: No penalty — treat absent features equally.
- `1`: Full penalty — reduce importance of absent features based on reference values.

---

## Advanced Utilities

MLMarker includes utilities for:
- GO enrichment: `get_go_enrichment()`
- Protein metadata from UniProt or HPA: `get_protein_info()`, `get_hpa_info()`
- Custom SHAP visualizations

---

## Contributing
We welcome contributions to the **MLMarker** project! If you have improvements, bug fixes, or additional features, feel free to submit a pull request.
    1. Fork the repository.
    2. Clone your fork locally.
    3. Create a new branch for your changes.
    4. Make your changes and commit them.
    5. Push the changes to your fork and submit a pull request.