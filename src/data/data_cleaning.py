import pandas as pd
import re
import difflib
import os
from typing import List, Dict, Tuple, Optional


class UNCountryDataCleaner:
    """
    A comprehensive data cleaning class for UN country data.
    Handles data loading, splitting by categories, cleaning, and exporting.
    """
    
    def __init__(self, raw_data_path: str = None):
        """
        Initialize the data cleaner.
        
        Args:
            raw_data_path: Path to the raw UN country data CSV file
        """
        self.raw_data_path = raw_data_path
        self.df = None
        self.df_general_info_flat = None
        self.df_economic_indicators_flat = None
        self.df_social_indicators_flat = None
        self.df_env_infra_indicators = None
        
        # Set pandas display options for better visualization
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load the raw UN country data from CSV file.
        
        Args:
            file_path: Path to the CSV file. If None, uses self.raw_data_path
            
        Returns:
            pd.DataFrame: Loaded raw data
        """
        if file_path:
            self.raw_data_path = file_path
        
        if not self.raw_data_path:
            raise ValueError("No data path provided. Please specify file_path or set raw_data_path during initialization.")
        
        self.df = pd.read_csv(self.raw_data_path)
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        print(f"Number of unique countries: {len(self.df['Country'].unique())}")
        return self.df
    
    def split_by_categories(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the main dataframe by categories.
        
        Returns:
            Tuple of DataFrames: (general_info, economic_indicators, social_indicators, env_infra_indicators)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        df_general_info = self.df[self.df["Category"] == "General Information"]
        df_economic_indicators = self.df[self.df["Category"] == "Economic indicators"]
        df_social_indicators = self.df[self.df["Category"] == "Social indicators"]
        df_env_infra_indicators = self.df[self.df["Category"] == "Environment and infrastructure indicators"]
        
        print(f"General Information: {len(df_general_info['Country'].unique())} countries")
        print(f"Economic Indicators: {len(df_economic_indicators['Country'].unique())} countries")
        print(f"Social Indicators: {len(df_social_indicators['Country'].unique())} countries")
        print(f"Environment & Infrastructure: {len(df_env_infra_indicators['Country'].unique())} countries")
        
        return df_general_info, df_economic_indicators, df_social_indicators, df_env_infra_indicators
    
    @staticmethod
    def pivot_df(df: pd.DataFrame, idx_cols: List[str]) -> pd.DataFrame:
        """
        Pivot dataframe from long to wide format.
        
        Args:
            df: DataFrame to pivot
            idx_cols: List of columns to use as index
            
        Returns:
            pd.DataFrame: Pivoted dataframe
        """
        pivot_df = df.pivot(index=idx_cols, columns="Indicator", values="Value")
        flat_df = pivot_df.reset_index()
        flat_df.columns.name = None
        return flat_df
    
    @staticmethod
    def clean_numerical_cols(df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """
        Clean numerical columns by removing non-numeric characters and converting to numeric.
        
        Args:
            df: DataFrame to clean
            numerical_cols: List of column names to clean
            
        Returns:
            pd.DataFrame: DataFrame with cleaned numerical columns
        """
        df_cleaned = df.copy()

        for col in numerical_cols:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].str.replace(r'[^\d.]', '', regex=True)
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            else:
                print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
                
        return df_cleaned
    
    @staticmethod
    def smart_combine_typo_columns(df: pd.DataFrame, root_forms: List[str]) -> pd.DataFrame:
        """
        Merge typo columns into their root forms using similarity matching with keyword-based filtering.
        
        Args:
            df: DataFrame to clean
            root_forms: List of root column names to merge similar columns into
            
        Returns:
            pd.DataFrame: DataFrame with combined columns
        """
        df_fixed = df.copy()
        
        # Define keywords that must match for specific column types
        def extract_key_keywords(column_name: str) -> set:
            """Extract key distinguishing keywords from column name"""
            keywords = set()
            column_lower = column_name.lower()
            
            # Trade-related keywords
            if 'export' in column_lower:
                keywords.add('export')
            if 'import' in column_lower:
                keywords.add('import')
            if 'balance' in column_lower:
                keywords.add('balance')
                
            # Education level keywords
            if 'primary' in column_lower:
                keywords.add('primary')
            if 'upr' in column_lower or 'upper' in column_lower:
                keywords.add('upper')
            if 'lowr' in column_lower or 'lower' in column_lower:
                keywords.add('lower')
            if 'sec.' in column_lower or 'sec' in column_lower or 'secondary' in column_lower:
                keywords.add('secondary')
                
            # Other distinguishing keywords
            if 'male' in column_lower and 'female' not in column_lower:
                keywords.add('male')
            if 'female' in column_lower and 'male' not in column_lower:
                keywords.add('female')
            if 'urban' in column_lower:
                keywords.add('urban')
            if 'rural' in column_lower:
                keywords.add('rural')
            if 'disbursed' in column_lower:
                keywords.add('disbursed')
            if 'received' in column_lower:
                keywords.add('received')
                
            return keywords
        
        def keywords_compatible(root_keywords: set, candidate_keywords: set) -> bool:
            """Check if keywords are compatible for merging"""
            # If either has no distinguishing keywords, they're compatible
            if not root_keywords or not candidate_keywords:
                return True
            
            # Check for conflicting keywords
            conflicting_pairs = [
                {'export', 'import'},
                {'export', 'balance'},
                {'import', 'balance'},
                {'primary', 'upper'},
                {'primary', 'lower'},
                {'upper', 'lower'},
                {'male', 'female'},
                {'urban', 'rural'},
                {'disbursed', 'received'}
            ]
            
            for conflict_pair in conflicting_pairs:
                if (conflict_pair & root_keywords) and (conflict_pair & candidate_keywords):
                    # Check if they have conflicting keywords from the same pair
                    root_conflict = conflict_pair & root_keywords
                    candidate_conflict = conflict_pair & candidate_keywords
                    if root_conflict != candidate_conflict:
                        return False
            
            return True
        
        # For each root form, find and merge similar columns
        for root_col in root_forms:
            if root_col in df_fixed.columns:
                similar_columns = []
                root_keywords = extract_key_keywords(root_col)
                
                # Check all columns for similarity to this root form
                for col in df_fixed.columns:
                    if col != root_col and col not in root_forms:  # Don't compare with itself or other root forms
                        similarity = difflib.SequenceMatcher(None, root_col.lower(), col.lower()).ratio()
                        
                        if similarity > 0.92:  # High similarity threshold
                            candidate_keywords = extract_key_keywords(col)
                            
                            # Only add if keywords are compatible
                            if keywords_compatible(root_keywords, candidate_keywords):
                                similar_columns.append((col, similarity))
                            else:
                                print(f"  Skipping '{col}' -> '{root_col}' due to keyword conflict")
                                print(f"    Root keywords: {root_keywords}")
                                print(f"    Candidate keywords: {candidate_keywords}")
                
                # Merge similar columns into the root column
                if similar_columns:
                    print(f"\nMerging into '{root_col}':")
                    print(f"  Found {len(similar_columns)} compatible similar columns")
                    
                    for similar_col, sim_score in similar_columns:
                        print(f"  Merging: '{similar_col}' -> '{root_col}' (similarity: {sim_score:.3f})")
                        # Fill missing values in root column with values from similar column
                        df_fixed[root_col] = df_fixed[root_col].fillna(df_fixed[similar_col])
                        # Drop the similar column
                        df_fixed = df_fixed.drop(columns=[similar_col])
        
        return df_fixed
    
    @staticmethod
    def split_ratio_column(df: pd.DataFrame, column_to_split: str, new_col1_name: str, new_col2_name: str) -> pd.DataFrame:
        """
        Split a column with format "value1 / value2" into separate columns.
        
        Args:
            df: DataFrame to modify
            column_to_split: Name of the column to split (should contain values like "19.1 / 75.3i")
            new_col1_name: Name for the first value column
            new_col2_name: Name for the second value column
            
        Returns:
            pd.DataFrame: DataFrame with new columns added and ratio calculated
        """
        df_result = df.copy()
        
        if column_to_split in df_result.columns:
            # Split the column on '/'
            split_data = df_result[column_to_split].str.split('/', expand=True)
            
            if split_data.shape[1] >= 2:
                # Clean and convert first value
                df_result[new_col1_name] = split_data[0].str.strip().str.replace(r'[^\d.]', '', regex=True)
                df_result[new_col1_name] = pd.to_numeric(df_result[new_col1_name], errors='coerce')
                
                # Clean and convert second value
                df_result[new_col2_name] = split_data[1].str.strip().str.replace(r'[^\d.]', '', regex=True)
                df_result[new_col2_name] = pd.to_numeric(df_result[new_col2_name], errors='coerce')
                
                # Calculate ratio
                ratio_col_name = f"{new_col1_name.split(' (')[0]}/{new_col2_name.split(' (')[0]} Ratio"
                df_result[ratio_col_name] = (df_result[new_col1_name] / df_result[new_col2_name]) * 100
                
                print(f"Successfully split '{column_to_split}' into:")
                print(f"  - {new_col1_name}")
                print(f"  - {new_col2_name}")
                print(f"  - {ratio_col_name}")
            else:
                print(f"Warning: Could not split '{column_to_split}' - unexpected format")
        else:
            print(f"Warning: Column '{column_to_split}' not found in DataFrame")
        
        return df_result
    
    def clean_general_info(self, df_general_info: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the general information dataset.
        
        Args:
            df_general_info: Raw general information DataFrame
            
        Returns:
            pd.DataFrame: Cleaned general information DataFrame
        """
        print("Cleaning General Information dataset...")
        
        # Drop unnecessary columns
        df_general_info = df_general_info.drop(columns=["Category", "Year"], axis=1)
        
        # Pivot the dataframe
        df_general_info_flat = self.pivot_df(df_general_info, idx_cols=["Country"])
        
        # Define numerical columns to clean
        numerical_cols = [
            "Capital city pop. (000)",
            "Capital city pop. (000, 2024)",
            "Exchange rate (per US$)",
            "Pop. density (per km2, 2024)",
            "Population (000, 2024)",
            "Sex ratio (m per 100 f)",
            "Surface area (km2)"
        ]
        
        # Clean numerical columns
        df_general_info_flat = self.clean_numerical_cols(df_general_info_flat, numerical_cols)
        
        # Convert UN membership date to datetime
        df_general_info_flat["UN membership date"] = pd.to_datetime(
            df_general_info_flat["UN membership date"], errors="coerce"
        )
        
        # Fill missing values
        df_general_info_flat["Capital_city_pop"] = df_general_info_flat["Capital city pop. (000, 2024)"].fillna(df_general_info_flat["Capital city pop. (000)"])
        df_general_info_flat = df_general_info_flat.drop(columns=["Capital city pop. (000)", "Capital city pop. (000, 2024)"])

        #Filling conversion rate with 1 for countries using USD 
        df_general_info_flat.loc[df_general_info_flat["Country"] != "State of Palestine", "Exchange rate (per US$)"] = df_general_info_flat.loc[df_general_info_flat["Country"]!= "State of Palestine", "Exchange rate (per US$)"].fillna(1)
        
        self.df_general_info_flat = df_general_info_flat
        print(f"General Info cleaned. Shape: {df_general_info_flat.shape}")
        print(f"Missing values: {df_general_info_flat.isnull().sum().sum()}")
        
        return df_general_info_flat
    
    def clean_economic_indicators(self, df_economic_indicators: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the economic indicators dataset.
        
        Args:
            df_economic_indicators: Raw economic indicators DataFrame
            
        Returns:
            pd.DataFrame: Cleaned economic indicators DataFrame
        """
        print("Cleaning Economic Indicators dataset...")
        
        # Drop unnecessary columns
        df_economic_indicators = df_economic_indicators.drop(columns=["Category"], axis=1)
        
        # Pivot the dataframe
        df_economic_indicators_flat = self.pivot_df(df_economic_indicators, idx_cols=["Country", "Year"])
        
        # Define root forms for combining typo columns
        root_forms = [
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
        
        # Combine typo columns
        df_economic_indicators_flat = self.smart_combine_typo_columns(df_economic_indicators_flat, root_forms)
        
        # Define numerical columns to clean
        numerical_cols = [
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
        
        # Clean numerical columns
        df_economic_indicators_flat = self.clean_numerical_cols(df_economic_indicators_flat, numerical_cols)
        
        # Split labor force participation rate column
        if 'Labour force participation rate (female/male pop. %)' in df_economic_indicators_flat.columns:
            df_economic_indicators_flat = self.split_ratio_column(
                df_economic_indicators_flat,
                "Labour force participation rate (female/male pop. %)",
                "Labour force participation rate - Female (per 100 pop.)",
                "Labour force participation rate - Male (per 100 pop.)"
            )
            # Drop the original column after splitting
            df_economic_indicators_flat = df_economic_indicators_flat.drop(
                columns=["Labour force participation rate (female/male pop. %)"], axis=1
            )
        
        self.df_economic_indicators_flat = df_economic_indicators_flat
        print(f"Economic Indicators cleaned. Shape: {df_economic_indicators_flat.shape}")
        print(f"Missing values: {df_economic_indicators_flat.isnull().sum().sum()}")
        
        return df_economic_indicators_flat
    
    def clean_social_indicators(self, df_social_indicators: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the social indicators dataset.
        
        Args:
            df_social_indicators: Raw social indicators DataFrame
            
        Returns:
            pd.DataFrame: Cleaned social indicators DataFrame
        """
        print("Cleaning Social Indicators dataset...")
        
        # Drop unnecessary columns
        df_social_indicators = df_social_indicators.drop(columns=["Category"], axis=1)
        
        # Pivot the dataframe
        df_social_indicators_flat = self.pivot_df(df_social_indicators, idx_cols=["Country", "Year"])
        
        # Define root forms for combining typo columns
        root_column_names = [
            'Country',
            'Year',
            'Education: Government expenditure (% of GDP)',
            'Education: Lowr. sec. gross enrol. ratio  (f/m per 100 pop.)',
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
        
        # Combine typo columns
        df_social_indicators_flat = self.smart_combine_typo_columns(df_social_indicators_flat, root_column_names)
        
        # Rename population age distribution column
        df_social_indicators_flat.rename(
            columns={"Population age distributiona,h (0-14/60+ years old, %)": "Population age distribution (0-14/60+ years old, %)"},
            inplace=True
        )
        
        # Define numerical columns to clean
        numerical_cols = [
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
        
        # Clean numerical columns
        df_social_indicators_flat = self.clean_numerical_cols(df_social_indicators_flat, numerical_cols)
        
        # Define columns to split and their new column names
        columns_to_split = [
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
                "column": "Education: Lowr. sec. gross enrol. ratio  (f/m per 100 pop.)",
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
        
        # Apply the split function to all columns
        for split_info in columns_to_split:
            if split_info["column"] in df_social_indicators_flat.columns:
                print(f"\nProcessing: {split_info['column']}")
                df_social_indicators_flat = self.split_ratio_column(
                    df=df_social_indicators_flat,
                    column_to_split=split_info["column"],
                    new_col1_name=split_info["col1"],
                    new_col2_name=split_info["col2"]
                )
                # Drop the original column after splitting
                df_social_indicators_flat = df_social_indicators_flat.drop(columns=[split_info["column"]], axis=1)
            else:
                print(f"Warning: Column '{split_info['column']}' not found in dataframe")
        
        self.df_social_indicators_flat = df_social_indicators_flat
        print(f"Social Indicators cleaned. Shape: {df_social_indicators_flat.shape}")
        print(f"Missing values: {df_social_indicators_flat.isnull().sum().sum()}")
        
        return df_social_indicators_flat
    
    def clean_env_infra_indicators(self, df_env_infra_indicators: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the environment and infrastructure indicators dataset.
        
        Args:
            df_env_infra_indicators: Raw environment and infrastructure indicators DataFrame
            
        Returns:
            pd.DataFrame: Cleaned environment and infrastructure indicators DataFrame
        """
        print("Cleaning Environment and Infrastructure Indicators dataset...")
        
        # Drop unnecessary columns
        df_env_infra_indicators = df_env_infra_indicators.drop(columns=["Category"], axis=1)
        
        # Pivot the dataframe
        df_env_infra_indicators = self.pivot_df(df_env_infra_indicators, idx_cols=["Country", "Year"])
        
        # Define root forms for combining typo columns
        root_column_names = [
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
        
        # Combine typo columns
        df_env_infra_indicators = self.smart_combine_typo_columns(df_env_infra_indicators, root_column_names)
        
        # Define numerical columns to clean
        numerical_cols = [
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
        
        # Clean numerical columns
        df_env_infra_indicators = self.clean_numerical_cols(df_env_infra_indicators, numerical_cols)
        
        # Define columns to split and their new column names for environment & infrastructure indicators
        columns_to_split = [
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
        
        # Apply the split function to all columns
        for split_info in columns_to_split:
            if split_info["column"] in df_env_infra_indicators.columns:
                print(f"\nProcessing: {split_info['column']}")
                df_env_infra_indicators = self.split_ratio_column(
                    df=df_env_infra_indicators,
                    column_to_split=split_info["column"],
                    new_col1_name=split_info["col1"],
                    new_col2_name=split_info["col2"]
                )
                # Drop the original column after splitting
                df_env_infra_indicators = df_env_infra_indicators.drop(columns=[split_info["column"]], axis=1)
            else:
                print(f"Warning: Column '{split_info['column']}' not found in dataframe")
        
        self.df_env_infra_indicators = df_env_infra_indicators
        print(f"Environment & Infrastructure cleaned. Shape: {df_env_infra_indicators.shape}")
        print(f"Missing values: {df_env_infra_indicators.isnull().sum().sum()}")
        
        return df_env_infra_indicators
    
    def export_cleaned_data(self, output_dir: str = "data/processed") -> None:
        """
        Export all cleaned datasets to CSV files.
        
        Args:
            output_dir: Directory to save the cleaned CSV files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("Exporting cleaned datasets...")
        
        # Export all cleaned dataframes to CSV
        if self.df_general_info_flat is not None:
            filepath = os.path.join(output_dir, 'general_info_clean.csv')
            self.df_general_info_flat.to_csv(filepath, index=False)
            print(f"✓ General Info: {self.df_general_info_flat.shape} -> {filepath}")
        
        if self.df_economic_indicators_flat is not None:
            filepath = os.path.join(output_dir, 'economic_indicators_clean.csv')
            self.df_economic_indicators_flat.to_csv(filepath, index=False)
            print(f"✓ Economic Indicators: {self.df_economic_indicators_flat.shape} -> {filepath}")
        
        if self.df_social_indicators_flat is not None:
            filepath = os.path.join(output_dir, 'social_indicators_clean.csv')
            self.df_social_indicators_flat.to_csv(filepath, index=False)
            print(f"✓ Social Indicators: {self.df_social_indicators_flat.shape} -> {filepath}")
        
        if self.df_env_infra_indicators is not None:
            filepath = os.path.join(output_dir, 'environment_infrastructure_clean.csv')
            self.df_env_infra_indicators.to_csv(filepath, index=False)
            print(f"✓ Environment & Infrastructure: {self.df_env_infra_indicators.shape} -> {filepath}")
        
        print(f"\n🎉 All datasets exported successfully to {output_dir}/ directory!")
    
    def merge_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge all cleaned datasets according to the specified logic.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_timeseries, df_merged)
                - df_timeseries: Economic, social, and environment/infrastructure indicators merged by Country and Year
                - df_merged: Timeseries data merged with general info by Country only
        """
        if (self.df_economic_indicators_flat is None or 
            self.df_social_indicators_flat is None or 
            self.df_env_infra_indicators is None or 
            self.df_general_info_flat is None):
            raise ValueError("All datasets must be cleaned before merging. Please run the cleaning methods first.")
        
        print("Merging datasets...")
        
        # Merge timeseries data (economic, social, environment/infrastructure)
        df_timeseries = self.df_economic_indicators_flat.merge(
            self.df_social_indicators_flat, 
            on=['Country', 'Year'], 
            how='outer'
        ).merge(
            self.df_env_infra_indicators, 
            on=['Country', 'Year'], 
            how='outer'
        )
        
        print(f"Timeseries data merged. Shape: {df_timeseries.shape}")
        
        # Merge timeseries with general info
        df_merged = pd.merge(df_timeseries, self.df_general_info_flat, on='Country', how='left')
        
        print(f"Final merged dataset. Shape: {df_merged.shape}")
        print(f"Countries in merged data: {len(df_merged['Country'].unique())}")
        print(f"Years covered: {sorted(df_merged['Year'].unique())}")
        
        return df_timeseries, df_merged
    
    def export_merged_data(self, df_timeseries: pd.DataFrame, df_merged: pd.DataFrame, 
                          output_dir: str = "data/processed") -> None:
        """
        Export merged datasets to CSV files.
        
        Args:
            df_timeseries: Timeseries merged dataframe
            df_merged: Complete merged dataframe
            output_dir: Directory to save the CSV files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("Exporting merged datasets...")
        
        # Export timeseries data
        timeseries_filepath = os.path.join(output_dir, 'timeseries_merged.csv')
        df_timeseries.to_csv(timeseries_filepath, index=False)
        print(f"✓ Timeseries data: {df_timeseries.shape} -> {timeseries_filepath}")
        
        # Export complete merged data
        merged_filepath = os.path.join(output_dir, 'complete_merged.csv')
        df_merged.to_csv(merged_filepath, index=False)
        print(f"✓ Complete merged data: {df_merged.shape} -> {merged_filepath}")
        
        print(f"\n🎉 Merged datasets exported successfully to {output_dir}/ directory!")
    
    def load_and_merge_cleaned_data(self, data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load cleaned datasets from CSV files and create merged datasets.
        Useful when you already have cleaned data and just want to create merged versions.
        
        Args:
            data_dir: Directory containing the cleaned CSV files
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_timeseries, df_merged)
        """
        print("Loading cleaned datasets from CSV files...")
        
        # Load cleaned datasets
        try:
            self.df_general_info_flat = pd.read_csv(os.path.join(data_dir, 'general_info_clean.csv'))
            self.df_economic_indicators_flat = pd.read_csv(os.path.join(data_dir, 'economic_indicators_clean.csv'))
            self.df_social_indicators_flat = pd.read_csv(os.path.join(data_dir, 'social_indicators_clean.csv'))
            self.df_env_infra_indicators = pd.read_csv(os.path.join(data_dir, 'environment_infrastructure_clean.csv'))
            
            print("✓ All cleaned datasets loaded successfully")
            
            # Create merged datasets
            df_timeseries, df_merged = self.merge_datasets()
            
            return df_timeseries, df_merged
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find cleaned data files in {data_dir}. "
                                  f"Please run the cleaning process first or check the file paths. Error: {e}")
    
    def process_all_data(self, raw_data_path: str = None, output_dir: str = "data/processed", 
                        include_merged: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Complete data processing pipeline: load, clean, merge, and export all datasets.
        
        Args:
            raw_data_path: Path to the raw UN country data CSV file
            output_dir: Directory to save the cleaned CSV files
            include_merged: Whether to create and export merged datasets
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing all cleaned and merged datasets
        """
        print("Starting complete data processing pipeline...")
        print("=" * 50)
        
        # Load data
        self.load_data(raw_data_path)
        
        # Split by categories
        df_general_info, df_economic_indicators, df_social_indicators, df_env_infra_indicators = self.split_by_categories()
        
        # Clean all datasets
        df_general_info_clean = self.clean_general_info(df_general_info)
        df_economic_indicators_clean = self.clean_economic_indicators(df_economic_indicators)
        df_social_indicators_clean = self.clean_social_indicators(df_social_indicators)
        df_env_infra_indicators_clean = self.clean_env_infra_indicators(df_env_infra_indicators)
        
        # Export cleaned data
        self.export_cleaned_data(output_dir)
        
        # Prepare return dictionary with cleaned datasets
        cleaned_datasets = {
            'general_info': df_general_info_clean,
            'economic_indicators': df_economic_indicators_clean,
            'social_indicators': df_social_indicators_clean,
            'environment_infrastructure': df_env_infra_indicators_clean
        }
        
        # Create merged datasets if requested
        if include_merged:
            print("\n" + "=" * 50)
            df_timeseries, df_merged = self.merge_datasets()
            self.export_merged_data(df_timeseries, df_merged, output_dir)
            
            # Add merged datasets to return dictionary
            cleaned_datasets.update({
                'timeseries_merged': df_timeseries,
                'complete_merged': df_merged
            })
        
        print("=" * 50)
        print("Data processing pipeline completed successfully!")
        
        return cleaned_datasets


def main():
    """
    Main function to demonstrate usage of the UNCountryDataCleaner class.
    """
    # Initialize the cleaner
    cleaner = UNCountryDataCleaner()
    
    # Process all data (adjust the path as needed)
    raw_data_path = "data/raw/un_country_data_raw.csv"  # Update this path
    output_dir = "data/processed"
    
    try:
        # Process all data including merging
        cleaned_datasets = cleaner.process_all_data(raw_data_path, output_dir, include_merged=True)
        
        # Display summary information
        print("\nDataset Summary:")
        print("-" * 50)
        for name, df in cleaned_datasets.items():
            print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")
            if name in ['timeseries_merged', 'complete_merged']:
                print(f"  └─ Countries: {len(df['Country'].unique())}")
                if 'Year' in df.columns:
                    years = sorted(df['Year'].dropna().unique())
                    if len(years) > 0:
                        print(f"  └─ Years: {min(years)} - {max(years)}")
        
        print("\nMerged datasets created successfully!")
        print("- timeseries_merged.csv: Economic, social, and environment/infrastructure data")
        print("- complete_merged.csv: All data including general information")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        print("Please ensure the raw data file path is correct and the file exists.")
if __name__ == "__main__":
    main()



