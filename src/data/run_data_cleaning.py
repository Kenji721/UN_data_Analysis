#!/usr/bin/env python3
"""
Data Cleaning Script for UN Country Data

This script uses the UNCountryDataCleaner class to process and clean
the UN country data from raw format to analysis-ready datasets.

Usage:
    python run_data_cleaning.py
"""

import sys
import os

# Add src directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_cleaning import UNCountryDataCleaner


def main():
    """Main function to run the data cleaning pipeline."""
    
    print("🌍 UN Country Data Cleaning Pipeline")
    print("=" * 50)
    
    # Define file paths
    raw_data_path = "data/raw/un_country_data_raw.csv"
    output_dir = "data/processed"
    
    # Check if raw data file exists
    if not os.path.exists(raw_data_path):
        print(f"❌ Error: Raw data file not found at {raw_data_path}")
        print("Please ensure the file exists or update the path in this script.")
        return 1
    
    try:
        # Initialize the cleaner
        cleaner = UNCountryDataCleaner()
        
        # Run the complete processing pipeline
        cleaned_datasets = cleaner.process_all_data(
            raw_data_path=raw_data_path,
            output_dir=output_dir
        )
        
        # Display final summary
        print("\n📊 Final Dataset Summary:")
        print("-" * 40)
        total_rows = 0
        total_cols = 0
        
        for name, df in cleaned_datasets.items():
            rows, cols = df.shape
            total_rows += rows
            total_cols += cols
            print(f"📋 {name.replace('_', ' ').title()}: {rows:,} rows, {cols} columns")
        
        print(f"\n🎯 Total: {total_rows:,} data points across {len(cleaned_datasets)} datasets")
        print(f"📁 All files saved to: {output_dir}/")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please check your file paths and try again.")
        return 1
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("Please check your data and try again.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)