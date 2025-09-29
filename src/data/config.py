"""
Configuration file for UN Country Data Processing

This file contains configuration settings for data paths, column mappings,
and processing parameters.
"""

import os

# =============================================================================
# FILE PATHS
# =============================================================================

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Input file
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "un_country_data_raw.csv")

# Output files
OUTPUT_FILES = {
    'general_info': os.path.join(PROCESSED_DATA_DIR, 'general_info_clean.csv'),
    'economic_indicators': os.path.join(PROCESSED_DATA_DIR, 'economic_indicators_clean.csv'),
    'social_indicators': os.path.join(PROCESSED_DATA_DIR, 'social_indicators_clean.csv'),
    'environment_infrastructure': os.path.join(PROCESSED_DATA_DIR, 'environment_infrastructure_clean.csv')
}

# =============================================================================
# DATA PROCESSING PARAMETERS
# =============================================================================

# Similarity threshold for combining typo columns
SIMILARITY_THRESHOLD = 0.8

# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================

# General Information numerical columns
GENERAL_INFO_NUMERICAL_COLS = [
    "Capital city pop. (000)",
    "Capital city pop. (000, 2024)",
    "Exchange rate (per US$)",
    "Pop. density (per km2, 2024)",
    "Population (000, 2024)",
    "Sex ratio (m per 100 f)",
    "Surface area (km2)"
]

# Economic Indicators root forms for typo combining
ECONOMIC_ROOT_FORMS = [
    'Country',
    'Year',
    'Agricultural production index (2014-2016=100)',
    'Balance of payments, current account (million US$)',
    'CPI: Consumer Price Index (2010=100)',
    'Economy: Agriculture (% of Gross Value Added)',
    'Economy: Industry (% of Gross Value Added)',
    'Economy: Services and other activity (% of GVA)',
    'Employment in agriculture (% of employed)',
    'Employment in industry (% of employed)',
    'Employment in services (% employed)',
    'GDP growth rate (annual %, const. 2015 prices)',
    'GDP per capita (current US$)',
    'GDP: Gross domestic product (million current US$)',
    'International trade: balance (million current US$)',
    'International trade: exports (million current US$)',
    'International trade: imports (million current US$)',
    'Labour force participation rate (female/male pop. %)',
    'Unemployment (% of labour force)'
]

# Economic Indicators numerical columns
ECONOMIC_NUMERICAL_COLS = [
    'Agricultural production index (2014-2016=100)',
    'Balance of payments, current account (million US$)',
    'CPI: Consumer Price Index (2010=100)',
    'Economy: Agriculture (% of Gross Value Added)',
    'Economy: Industry (% of Gross Value Added)',
    'Economy: Services and other activity (% of GVA)',
    'Employment in agriculture (% of employed)',
    'Employment in industry (% of employed)',
    'Employment in services (% employed)',
    'GDP growth rate (annual %, const. 2015 prices)',
    'GDP per capita (current US$)',
    'GDP: Gross domestic product (million current US$)',
    'International trade: balance (million current US$)',
    'International trade: exports (million current US$)',
    'International trade: imports (million current US$)',
    'Unemployment (% of labour force)'
]

# Social Indicators root forms for typo combining
SOCIAL_ROOT_FORMS = [
    'Country',
    'Year',
    'Education: Government expenditure (% of GDP)',
    'Education: Lowr. sec. gross enrol. ratio (f/m per 100 pop.)',
    'Education: Primary gross enrol. ratio (f/m per 100 pop.)',
    'Education: Upr. Sec. gross enrol. ratio (f/m per 100 pop.)',
    'Fertility rate, total (live births per woman)',
    'Health: Current expenditure (% of GDP)',
    'Health: Physicians (per 1 000 pop.)',
    'Intentional homicide rate (per 100 000 pop.)',
    'International migrant stock (000/% of total pop.)',
    'Life expectancy at birth (females/males, years)',
    'Population age distributiona,h (0-14/60+ years old, %)',
    'Population growth rate (average annual %)',
    'Refugees and others of concern to UNHCR (000)',
    'Seats held by women in national parliaments (%)',
    'Under five mortality rate (per 1000 live births)',
    'Urban population (% of total population)',
    'Urban population growth rate (average annual %)'
]

# Social Indicators numerical columns
SOCIAL_NUMERICAL_COLS = [
    "Education: Government expenditure (% of GDP)",
    "Fertility rate, total (live births per woman)",
    "Health: Current expenditure (% of GDP)",
    "Health: Physicians (per 1 000 pop.)",
    "Intentional homicide rate (per 100 000 pop.)",
    "Population growth rate (average annual %)",
    "Refugees and others of concern to UNHCR (000)",
    "Seats held by women in national parliaments (%)",
    "Under five mortality rate (per 1000 live births)",
    "Urban population (% of total population)",
    "Urban population growth rate (average annual %)"
]

# Social Indicators columns to split
SOCIAL_COLUMNS_TO_SPLIT = [
    {
        "column": "Education: Primary gross enrol. ratio (f/m per 100 pop.)",
        "col1": "Education: Primary gross enrol. ratio - Female (per 100 pop.)",
        "col2": "Education: Primary gross enrol. ratio - Male (per 100 pop.)"
    },
    {
        "column": "Education: Upr. Sec. gross enrol. ratio (f/m per 100 pop.)",
        "col1": "Education: Upper Sec. gross enrol. ratio - Female (per 100 pop.)",
        "col2": "Education: Upper Sec. gross enrol. ratio - Male (per 100 pop.)"
    },
    {
        "column": "Education: Lowr. sec. gross enrol. ratio (f/m per 100 pop.)",
        "col1": "Education: Lower Sec. gross enrol. ratio - Female (per 100 pop.)",
        "col2": "Education: Lower Sec. gross enrol. ratio - Male (per 100 pop.)"
    },
    {
        "column": "International migrant stock (000/% of total pop.)",
        "col1": "International migrant stock (000)",
        "col2": "International migrant stock (% of total pop.)"
    },
    {
        "column": "Life expectancy at birth (females/males, years)",
        "col1": "Life expectancy at birth - Female (years)",
        "col2": "Life expectancy at birth - Male (years)"
    },
    {
        "column": "Population age distribution (0-14/60+ years old, %)",
        "col1": "Population age distribution - 0-14 years (%)",
        "col2": "Population age distribution - 60+ years (%)"
    }
]

# Environment & Infrastructure root forms for typo combining
ENV_INFRA_ROOT_FORMS = [
    'CO2 emission estimates (million tons/tons per capita)',
    'Country',
    'Energy production, primary (Petajoules)',
    'Energy supply per capita (Gigajoules)',
    'Forested area (% of land area)',
    'Important sites for terrestrial biodiversity protected (%)',
    'Individuals using the Internet (per 100 inhabitants)',
    'Net Official Development Assist. disbursed (% of GNI)',
    'Net Official Development Assist. received (% of GNI)',
    'Pop. using safely managed drinking water (urban/rural, %)',
    'Pop. using safely managed sanitation (urban/rural, %)',
    'Research & Development expenditure (% of GDP)',
    'Threatened species (number)',
    'Tourist/visitor arrivals at national borders (000)',
    'Year'
]

# Environment & Infrastructure numerical columns
ENV_INFRA_NUMERICAL_COLS = [
    "Energy production, primary (Petajoules)",
    "Energy supply per capita (Gigajoules)",
    "Forested area (% of land area)",
    "Important sites for terrestrial biodiversity protected (%)",
    "Individuals using the Internet (per 100 inhabitants)",
    "Net Official Development Assist. disbursed (% of GNI)",
    "Net Official Development Assist. received (% of GNI)",
    "Research & Development expenditure (% of GDP)",
    "Threatened species (number)",
    "Tourist/visitor arrivals at national borders (000)"
]

# Environment & Infrastructure columns to split
ENV_INFRA_COLUMNS_TO_SPLIT = [
    {
        "column": "CO2 emission estimates (million tons/tons per capita)",
        "col1": "CO2 emission estimates - Total (million tons)",
        "col2": "CO2 emission estimates - Per capita (tons per capita)"
    },
    {
        "column": "Pop. using safely managed drinking water (urban/rural, %)",
        "col1": "Pop. using safely managed drinking water - Urban (%)",
        "col2": "Pop. using safely managed drinking water - Rural (%)"
    },
    {
        "column": "Pop. using safely managed sanitation (urban/rural, %)",
        "col1": "Pop. using safely managed sanitation - Urban (%)",
        "col2": "Pop. using safely managed sanitation - Rural (%)"
    }
]