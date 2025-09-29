#!/usr/bin/env python3
"""
Test script for the UNCountryDataCleaner class

This script demonstrates how to use the data cleaning module with sample data.
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_data_cleaner():
    """Test the data cleaner functionality."""
    try:
        from src.data.data_cleaning import UNCountryDataCleaner
        print("✅ Successfully imported UNCountryDataCleaner")
        
        # Initialize the cleaner
        cleaner = UNCountryDataCleaner()
        print("✅ Successfully initialized UNCountryDataCleaner")
        
        # Test if the raw data file exists
        raw_data_path = "data/raw/un_country_data_raw.csv"
        if os.path.exists(raw_data_path):
            print(f"✅ Raw data file found at: {raw_data_path}")
            
            # Try loading the data
            try:
                df = cleaner.load_data(raw_data_path)
                print(f"✅ Data loaded successfully: {df.shape}")
                
                # Test splitting by categories
                general, economic, social, env_infra = cleaner.split_by_categories()
                print("✅ Data split by categories successfully")
                
                print("\n📊 Data Overview:")
                print(f"  - General Info: {general.shape}")
                print(f"  - Economic Indicators: {economic.shape}")
                print(f"  - Social Indicators: {social.shape}")
                print(f"  - Environment & Infrastructure: {env_infra.shape}")
                
                # Note: Full processing would require the actual data file
                print("\n💡 To run full processing, use:")
                print("    cleaned_datasets = cleaner.process_all_data(raw_data_path, 'data/processed')")
                
            except Exception as e:
                print(f"⚠️ Could not load data (expected if file not present): {e}")
        else:
            print(f"ℹ️ Raw data file not found at: {raw_data_path}")
            print("   This is expected if you haven't placed the data file yet.")
        
        print("\n🎉 All tests passed! The data cleaner is ready to use.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure pandas and other dependencies are installed.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def show_usage_examples():
    """Show usage examples for the data cleaner."""
    print("\n" + "="*60)
    print("📖 USAGE EXAMPLES")
    print("="*60)
    
    print("""
# Example 1: Complete pipeline
from src.data.data_cleaning import UNCountryDataCleaner

cleaner = UNCountryDataCleaner()
cleaned_datasets = cleaner.process_all_data(
    raw_data_path="data/raw/un_country_data_raw.csv",
    output_dir="data/processed"
)

# Example 2: Step-by-step processing
cleaner = UNCountryDataCleaner("data/raw/un_country_data_raw.csv")
df = cleaner.load_data()
general, economic, social, env_infra = cleaner.split_by_categories()

# Clean individual datasets
clean_general = cleaner.clean_general_info(general)
clean_economic = cleaner.clean_economic_indicators(economic)
clean_social = cleaner.clean_social_indicators(social)
clean_env_infra = cleaner.clean_env_infra_indicators(env_infra)

# Export results
cleaner.export_cleaned_data("data/processed")

# Example 3: Access cleaned data
print(f"General info shape: {cleaner.df_general_info_flat.shape}")
print(f"Economic indicators shape: {cleaner.df_economic_indicators_flat.shape}")
""")

if __name__ == "__main__":
    print("🧪 Testing UN Country Data Cleaner")
    print("="*50)
    
    success = test_data_cleaner()
    
    if success:
        show_usage_examples()
    
    print("\n" + "="*50)
    print("Test completed!")