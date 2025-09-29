# UN Country Data Processing Project

This project contains tools and scripts for cleaning and processing UN country data from raw format into analysis-ready datasets.

## 🏗️ Project Structure

```
dsf_project/
├── data/
│   ├── raw/                     # Raw data files
│   │   └── un_country_data_raw.csv
│   └── processed/               # Cleaned data files (generated)
│       ├── general_info_clean.csv
│       ├── economic_indicators_clean.csv
│       ├── social_indicators_clean.csv
│       └── environment_infrastructure_clean.csv
├── src/
│   └── data/
│       ├── data_cleaning.py     # Main data cleaning module
│       └── scraper.py
├── notebooks/
│   └── raw_data_exploration.ipynb  # Original exploration notebook
├── config.py                   # Configuration settings
├── run_data_cleaning.py        # Main execution script
├── test_cleaner.py            # Test and demo script
└── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Your Data
Ensure your raw UN country data CSV file is located at:
```
data/raw/un_country_data_raw.csv
```

### 3. Run the Cleaning Pipeline
```bash
python run_data_cleaning.py
```

### 4. Test the Module
```bash
python test_cleaner.py
```

## 📚 Using the Data Cleaner Module

### Simple Usage
```python
from src.data.data_cleaning import UNCountryDataCleaner

# Process all data in one go
cleaner = UNCountryDataCleaner()
cleaned_datasets = cleaner.process_all_data(
    raw_data_path="data/raw/un_country_data_raw.csv",
    output_dir="data/processed"
)

# Access cleaned datasets
general_info = cleaned_datasets['general_info']
economic_data = cleaned_datasets['economic_indicators']
social_data = cleaned_datasets['social_indicators']
env_infra_data = cleaned_datasets['environment_infrastructure']
```

### Step-by-Step Usage
```python
from src.data.data_cleaning import UNCountryDataCleaner

# Initialize
cleaner = UNCountryDataCleaner("data/raw/un_country_data_raw.csv")

# Load and split data
df = cleaner.load_data()
general, economic, social, env_infra = cleaner.split_by_categories()

# Clean individual datasets
clean_general = cleaner.clean_general_info(general)
clean_economic = cleaner.clean_economic_indicators(economic)
clean_social = cleaner.clean_social_indicators(social)
clean_env_infra = cleaner.clean_env_infra_indicators(env_infra)

# Export results
cleaner.export_cleaned_data("data/processed")
```

## 🔧 Key Features

### Data Processing
- **Automatic column typo handling**: Uses similarity matching to merge columns with typos
- **Ratio column splitting**: Automatically splits columns like "female/male" into separate columns
- **Numerical data cleaning**: Removes non-numeric characters and converts to proper data types
- **Missing data handling**: Identifies and reports missing values
- **Date parsing**: Converts date columns to proper datetime format

### Data Categories
The cleaner processes four main categories of UN country data:

1. **General Information**
   - Population data, surface area, capital city information
   - Exchange rates, population density
   - UN membership dates

2. **Economic Indicators**
   - GDP data, trade balances, employment statistics
   - Agricultural and industrial production indices
   - Consumer price indices

3. **Social Indicators**
   - Education enrollment ratios, health expenditure
   - Life expectancy, fertility rates
   - Urban population statistics

4. **Environment & Infrastructure**
   - CO2 emissions, energy production
   - Internet usage, biodiversity protection
   - Water and sanitation access

## 📊 Output Data Structure

All cleaned datasets are saved as CSV files with:
- Consistent column names
- Proper data types (numeric, datetime, categorical)
- Split ratio columns for better analysis
- Merged duplicate/typo columns

## ⚙️ Configuration

The `config.py` file contains all configuration settings including:
- File paths and directory structures
- Column definitions for each dataset
- Processing parameters (similarity thresholds, etc.)
- Column splitting and merging rules

## 🔍 Data Quality Features

- **Typo Detection**: Automatically identifies and merges similar column names
- **Data Validation**: Reports missing values and data quality issues
- **Consistent Formatting**: Standardizes column names and data formats
- **Error Handling**: Graceful handling of missing or malformed data

## 📋 Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.21.0
- Standard library modules: re, difflib, os, typing

## 🤝 Contributing

To add new cleaning functionality:

1. Add new methods to the `UNCountryDataCleaner` class
2. Update configuration in `config.py` if needed
3. Add tests to `test_cleaner.py`
4. Update this README

## 📝 Notes

- The original notebook (`notebooks/raw_data_exploration.ipynb`) is preserved for reference
- All cleaning logic has been refactored into the `UNCountryDataCleaner` class
- The module is designed to be extensible and maintainable
- Error handling and logging help with debugging and monitoring

## 🎯 Benefits of the Refactored Code

1. **Maintainability**: Object-oriented design makes it easy to modify and extend
2. **Reusability**: Can be imported and used in other projects
3. **Testability**: Each method can be tested independently
4. **Documentation**: Clear docstrings and type hints
5. **Configuration**: Centralized settings in `config.py`
6. **Error Handling**: Robust error handling and user feedback