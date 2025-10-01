# Country Development Indicators Analysis (UN Data)

## Project Description

This project analyzes a **United Nations (UNdata)** database containing **socioeconomic, demographic, health, and education indicators** from different countries and years. The objective is to transform raw data into analysis-ready datasets to answer key questions about development and support public policy decision-making.

The repository includes a comprehensive *data cleaning pipeline* that processes the data and categorizes it to facilitate its use in exploratory data analysis (EDA), hypothesis testing (t-tests), principal component analysis (PCA), and machine learning models.

## Project Structure

The repository is organized into the following folders:

```
dsf_project/
├── data/
│   ├── raw/                    # Raw datasets from UN sources
│   └── processed/              # Cleaned and processed datasets
├── src/
│   ├── data/                   # Data processing modules
│   │   ├── data_cleaning.py    # UNCountryDataCleaner class
│   │   ├── scraper.py          # Web scraping utilities
│   │   └── run_data_cleaning.py # Main cleaning pipeline script
│   └── visualization/          # Generated charts and plots
├── notebooks/
│   ├── 1.raw_data_exploration.ipynb    # Initial data exploration
│   ├── 2.EDA_general.ipynb             # General exploratory analysis
│   ├── 3.EDA_regression.ipynb          # Regression-focused EDA
│   ├── 4.MODEL_classification.ipynb    # Classification models
│   └── 5.statistical_analysis.ipynb   # Statistical hypothesis testing
├── models/                     # Trained machine learning models
├── reports/                    # Analysis reports and documentation
├── tests/                      # Unit tests for code quality
└── requirements.txt           # Project dependencies
```

### Key Files:
- `src/data/run_data_cleaning.py`: Main script to execute the complete data cleaning pipeline
- `requirements.txt`: Dependencies required to run the project
- `models/`: Exported trained models ready for production use

## Installation

To set up the development environment and run the project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Kenji721/country_data_dsf_project.git
    cd country_data_dsf_project
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: At least 4GB RAM (8GB recommended for large datasets)
- **Storage**: 500MB for data and models
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn, jupyter

## Usage

### Data Processing Pipeline

To execute the data cleaning pipeline and generate processed datasets:

```bash
cd src/data
python run_data_cleaning.py
```

The cleaned datasets will be saved in the `data/processed/` folder and will be ready for use in analysis notebooks.

### Running Analysis Notebooks

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Execute notebooks in order:**
   - `1.raw_data_exploration.ipynb` - Initial data exploration
   - `2.EDA_general.ipynb` - Comprehensive exploratory data analysis
   - `3.EDA_regression.ipynb` - Regression-specific analysis
   - `4.MODEL_classification.ipynb` - Classification model development
   - `5.statistical_analysis.ipynb` - Hypothesis testing and statistical analysis

### Quick Start Example

```python
# Load processed data
import pandas as pd
df = pd.read_csv('data/processed/complete_merged.csv')

# Basic exploration
print(f"Dataset shape: {df.shape}")
print(f"Countries: {df['Country'].nunique()}")
print(f"Years covered: {df['Year'].unique()}")
```

## Analytical Approach

The analysis focuses on answering **five key business questions** through multiple methodological approaches:

### Statistical Analysis
- **Hypothesis Testing (Welch's t-tests):** To compare means between different country groups and determine if statistically significant differences exist.
- **ANOVA Analysis:** To assess variance across multiple groups and identify significant factors.
- **Correlation Analysis:** To identify relationships between socioeconomic indicators.

### Dimensionality Reduction
- **Principal Component Analysis (PCA):** To reduce data dimensionality, identify multivariate patterns, and facilitate interpretation of variable relationships.

### Machine Learning Models

#### Classification Models
- **Objective:** Predict a country's development level (classified into GDP per capita quartiles) based on socioeconomic indicators
- **Best Model:** Logistic Regression with **87.27% accuracy**
- **Key Features:** Under-5 mortality rate, Internet usage, GDP per capita
- **Use Cases:** Development classification, policy benchmarking

#### Regression Models  
- **Objective:** Predict average life expectancy at birth based on socioeconomic, environmental, and infrastructure indicators
- **Best Model:** Extra Trees Regressor with **95.57% R² score**
- **Key Features:** Under-5 mortality rate (43.1% importance), Internet usage (14.5%), Fertility rate (12.3%)
- **Performance:** ±1.29 years average prediction error
- **Use Cases:** Health policy planning, resource allocation

### Research Questions

The project seeks to answer the following key research questions:

1. **Health Investment & Longevity:** Do countries with higher health expenditure show greater life expectancy?
2. **Demographic Transition:** Do countries with high fertility rates differ in life expectancy compared to those with low fertility?
3. **Development & Education:** Do developed countries have higher school enrollment rates than developing countries?
4. **Health-Economic Nexus:** Is greater health expenditure associated with higher economic development (measured by GDP per capita)?
5. **Healthcare Access & Child Mortality:** Is a higher number of physicians per 1,000 inhabitants associated with lower under-5 mortality rates?

### Key Findings

- **Strong Correlation:** Under-5 mortality rate is the strongest predictor of life expectancy (r = -0.884)
- **Digital Divide:** Internet access serves as a powerful proxy for technological development (r = +0.791)
- **Regional Disparities:** Persistent North-South development gaps with 35.2 years difference in life expectancy between extremes
- **Development Syndrome:** Economic, health, and technological indicators tend to move together
- **Policy Implications:** Investment in maternal-child health and digital infrastructure shows highest impact on development outcomes 

## Datasets

The project uses the following datasets:

### Raw Data
- **Source:** `data/raw/un_country_data_raw.csv` 
- **Origin:** United Nations Statistics Division (UNdata)
- **Coverage:** 630+ observations, 120+ variables
- **Temporal Scope:** Primarily 2024 data with some historical records

### Processed Datasets
- `data/processed/general_info_clean.csv` - Country metadata and geographic information
- `data/processed/economic_indicators_clean.csv` - GDP, employment, trade indicators  
- `data/processed/social_indicators_clean.csv` - Health, education, demographic indicators
- `data/processed/environment_infrastructure_clean.csv` - Environmental and technology indicators
- `data/processed/complete_merged.csv` - **Main dataset** with all indicators merged
- `data/processed/timeseries_merged.csv` - Time series format for temporal analysis

### Data Quality
- **Completeness:** ~77% complete (23% missing values)
- **Geographic Coverage:** UN member countries worldwide
- **Variables:** 35 key features selected for modeling after quality filtering
- **Target Variables:** 
  - Life expectancy (regression): 50.2-85.4 years range
  - GDP development class (classification): 4 quartile levels 

## Model Performance

### Classification Model (GDP Development Level)
- **Algorithm:** Logistic Regression (optimized)
- **Accuracy:** 87.27%
- **Cross-validation:** 86.89% ± 3.12%
- **Key Features:** Under-5 mortality, Internet usage, GDP per capita
- **Use Case:** Classify countries into development quartiles

### Regression Model (Life Expectancy Prediction)  
- **Algorithm:** Extra Trees Regressor (optimized)
- **R² Score:** 95.57%
- **RMSE:** 1.56 years
- **MAE:** 1.29 years  
- **Key Features:** Under-5 mortality (43.1%), Internet usage (14.5%), Fertility rate (12.3%)
- **Use Case:** Predict life expectancy for policy planning

Both models are **production-ready** and exported in the `models/` directory with complete documentation.

## Repository Statistics

- **Code Quality:** Comprehensive testing suite in `tests/`
- **Documentation:** Detailed technical reports in `reports/`
- **Visualizations:** 20+ charts and analysis plots
- **Reproducibility:** Fixed random seeds and documented methodology
- **Version Control:** Complete Git history with meaningful commits

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Leonardo Kenji Minemura Suazo** - *Lead Data Scientist* - [Kenji721](https://github.com/Kenji721)
- **Sara Rocío Miranda Mateos** - *Data Analyst*

## Acknowledgments

- Data obtained from **UNdata**, a service of the **United Nations**
- Special thanks to the UN Statistics Division for providing comprehensive country-level data
- Inspired by the UN Sustainable Development Goals (SDGs) framework
- Built with open-source tools: Python, scikit-learn, pandas, matplotlib, seaborn

---

## 📊 Quick Results Summary

| Metric | Classification Model | Regression Model |
|--------|---------------------|------------------|
| **Performance** | 87.27% Accuracy | 95.57% R² |
| **Error Rate** | 12.73% | ±1.29 years |
| **Top Predictor** | Under-5 mortality | Under-5 mortality |
| **Model Type** | Logistic Regression | Extra Trees |
| **Use Case** | Development classification | Life expectancy prediction |

**🎯 Bottom Line:** Both models achieve excellent performance for policy applications, with under-5 mortality rate emerging as the most critical development indicator.