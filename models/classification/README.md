# GDP Classification Models

## 🎯 Model Overview
This directory contains trained machine learning models for classifying countries into GDP per capita categories (quartiles).

## 📊 Model Performance Summary
- **Best Model**: Logistic Regression (Optimized)
- **Test Accuracy**: 88.62%
- **Cross-Validation**: 85.06% ± 4.12%
- **Classes**: 4 GDP quartiles (0=Lowest, 3=Highest)

## 📁 Files Description

### Main Models
- `best_classification_model.joblib` - **Recommended for production**
  - Complete package with best performing model
  - Includes metadata, performance metrics, and class labels
  - Ready for deployment

### Default Models (Baseline)
- `logistic_regression_default.joblib` - Logistic Regression with default parameters
- `decision_tree_default.joblib` - Decision Tree with default parameters  
- `random_forest_default.joblib` - Random Forest with default parameters
- `gradient_boosting_default.joblib` - Gradient Boosting with default parameters
- `svm_default.joblib` - Support Vector Machine with scaling pipeline
- `knn_default.joblib` - K-Nearest Neighbors with scaling pipeline

### Optimized Models (Hyperparameter Tuned)
- `logistic_regression_optimized.joblib` - **Best performer** (88.62% accuracy)
- `random_forest_optimized.joblib` - Tuned Random Forest
- `gradient_boosting_optimized.joblib` - Tuned Gradient Boosting

## 🔧 How to Use

### Quick Start
```python
import joblib
import pandas as pd

# Load the best model
model_package = joblib.load('models/classification/best_classification_model.joblib')

# Extract components
model = model_package['model']
feature_names = model_package['feature_names']
class_labels = model_package['class_labels']

# Make predictions
predictions = model.predict(your_data)
probabilities = model.predict_proba(your_data)
```

### Required Features (28 total)
**Numerical Features (6):**
- Population age distribution - 0-14 years (%)
- Life expectancy at birth - average
- Employment in agriculture (% of employed)
- Individuals using the Internet (per 100 inhabitants)
- CO2 emission estimates - Per capita (tons per capita)
- Economy: Services and other activity (% of GVA)

**Regional Dummy Variables (22):**
- One-hot encoded regional indicators
- Only one should be 1, others should be 0
- Regions: Caribbean, Central America, Central Asia, Eastern Africa, etc.

## 🎯 Model Interpretation

### GDP Classes:
- **Class 0**: Lowest GDP quartile (poorest countries)
- **Class 1**: Low-Medium GDP quartile
- **Class 2**: Medium-High GDP quartile  
- **Class 3**: Highest GDP quartile (richest countries)

### Performance by Class:
- **Class 0**: 98% precision, 94% recall - Excellent
- **Class 1**: 81% precision, 87% recall - Good
- **Class 2**: 79% precision, 82% recall - Good
- **Class 3**: 97% precision, 92% recall - Excellent

### Key Insights:
- Model excels at identifying extreme GDP categories (0 and 3)
- Middle categories (1 and 2) have slightly lower precision
- Regional features are important predictors
- Economic and demographic indicators most influential

## 🚀 Production Deployment

### System Requirements:
```python
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
joblib>=1.0.0
```

### Model Package Contents:
Each model file includes:
- Trained model object
- Feature names and order
- Target class labels
- Performance metrics
- Export timestamp
- Dataset information
- Best parameters (for optimized models)

### Memory Usage:
- Individual models: 100KB - 2MB each
- Best model package: ~500KB
- Prediction time: <1ms per sample

## 📈 Use Cases

### 1. Economic Analysis
- Country GDP classification for research
- Economic development assessment
- International aid targeting

### 2. Policy Planning
- Resource allocation decisions
- Development program design
- Economic benchmarking

### 3. Business Intelligence
- Market segmentation by economic development
- Investment risk assessment
- Market entry strategies

## ⚠️ Important Notes

### Data Requirements:
- All 28 features must be provided
- Regional dummy variables correctly encoded (one-hot)
- No missing values allowed
- Features should be in similar scale/format as training data

### Model Limitations:
- Trained on 2024 country-level data (630 complete observations)
- May not generalize to sub-national analysis
- Performance may vary with significantly different time periods
- GDP classifications are based on quartiles from training data

### Retraining Recommendations:
- Update annually with new economic data
- Monitor prediction accuracy in production
- Retrain if accuracy drops below 85%
- Consider rebalancing classes if data distribution changes

## 📞 Support
For questions about model usage, performance, or integration, refer to the original analysis notebook: `4.MODEL_classification.ipynb`

---
**Models Exported**: 2025-09-30  
**Training Data**: UN Country Data 2024  
**Best Algorithm**: Logistic Regression (Optimized)  
**Performance**: 88.62% Test Accuracy