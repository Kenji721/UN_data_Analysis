# Life Expectancy Prediction Model

## 🎯 Model Overview
This directory contains the best-performing machine learning model for predicting life expectancy at birth based on socioeconomic, health, and environmental indicators.

## 📊 Model Performance
- **Algorithm**: Extra Trees Regressor (Ensemble Method)
- **Test R² Score**: 0.9557 (explains 95.57% of variance)
- **Test RMSE**: 1.56 years
- **Test MAE**: 1.18 years
- **Cross-Validation R²**: 0.9555 ± 0.0206
- **Overfitting**: Minimal (0.0449)

## 📁 Files Description

### 1. `best_life_expectancy_model.pkl` (Recommended)
- **Size**: Full model package with metadata
- **Contents**: Model + Scaler + Metadata + Performance metrics
- **Format**: Python Pickle
- **Best for**: Production deployment with full documentation

### 2. `best_life_expectancy_model.joblib` (Fastest Loading)
- **Size**: Full model package (compressed)
- **Contents**: Same as pickle but optimized for sklearn models
- **Format**: Joblib (3x compression)
- **Best for**: High-performance applications

### 3. `extra_trees_model_only.pkl` (Lightweight)
- **Size**: Model object only
- **Contents**: Just the trained Extra Trees model
- **Format**: Python Pickle
- **Best for**: Memory-constrained environments

### 4. `model_loading_example.py`
- **Contents**: Code examples for loading and using the model
- **Best for**: Quick integration reference

## 🔧 How to Use

### Quick Start
```python
import pickle
import pandas as pd

# Load the model
with open('best_life_expectancy_model.pkl', 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
feature_names = model_package['feature_names']

# Make predictions
# your_data should be a DataFrame with the 35 required features
predictions = model.predict(your_data)
```

### Required Features (35 total)
**Socioeconomic Indicators:**
- Under five mortality rate (per 1000 live births)
- Fertility rate, total (live births per woman)
- Population age distribution - 60+ years (%)
- Employment in services (% employed)
- GDP per capita (current US$)
- Economy: Agriculture (% of Gross Value Added)
- Economy: Services and other activity (% of GVA)
- Population growth rate (average annual %)

**Infrastructure & Environment:**
- Individuals using the Internet (per 100 inhabitants)
- CO2 emission estimates - Per capita (tons per capita)
- Energy supply per capita (Gigajoules)
- Tourist/visitor arrivals at national borders (000)
- Energy production, primary (Petajoules)

**Regional Dummy Variables (22 regions):**
- Caribbean, Central America, Central Asia, Eastern Africa, etc.
- Only one should be 1, others should be 0

## 🎯 Model Interpretation

### Top 5 Most Important Features:
1. **Under five mortality rate** (43.1% importance) - Strong negative predictor
2. **Internet usage** (14.5% importance) - Strong positive predictor
3. **Fertility rate** (12.3% importance) - Negative predictor
4. **GDP per capita** (7.8% importance) - Positive predictor
5. **Population 60+ years** (6.0% importance) - Positive predictor

### Performance Interpretation:
- **Excellent Accuracy**: 95.6% of life expectancy variation explained
- **Clinical Precision**: Average error of only 1.18 years
- **Reliable Predictions**: 68% of predictions within ±1.56 years
- **Stable Model**: Consistent performance across different data samples

## 🚀 Production Deployment

### Requirements:
```python
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
joblib>=1.0.0
```

### Memory Usage:
- Full package: ~2-5 MB
- Model only: ~500KB-1MB
- Prediction time: <1ms per sample

### Scaling Notes:
- **No scaling required** for new data (Extra Trees is scale-invariant)
- Scaler included in package for completeness
- Handle missing values before prediction

## 📈 Use Cases

### 1. Policy Planning
- Predict impact of health interventions
- Resource allocation for development programs
- International aid targeting

### 2. Research Applications
- Comparative health system analysis
- Socioeconomic impact studies
- Demographic transition modeling

### 3. Health Monitoring
- Country health performance benchmarking
- Early warning systems for health crises
- Progress tracking for SDG goals

## ⚠️ Important Notes

### Data Requirements:
- All 35 features must be provided
- Regional dummy variables correctly encoded
- No missing values allowed
- Data should be in similar scale/format as training data

### Model Limitations:
- Trained on 2024 country-level data
- May not generalize to sub-national levels
- Performance may degrade with significantly different time periods
- Extrapolation beyond training data range not recommended

### Retraining Recommendations:
- Update model annually with new data
- Monitor prediction accuracy in production
- Retrain if performance drops below R² = 0.90

## 📞 Support
For questions about model usage, performance, or integration, refer to the original analysis notebook: `5.MODEL_regression.ipynb`

---
**Model Exported**: 2025-09-30  
**Training Data**: UN Country Data 2024  
**Algorithm**: Extra Trees Regressor (Scikit-learn)  
**Performance**: R² = 0.9557, RMSE = 1.56 years