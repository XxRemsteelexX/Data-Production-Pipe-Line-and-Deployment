#!/usr/bin/env python
# coding: utf-8

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def clean_data():
    # load dataset
    print("Loading dataset...")
    data = pd.read_csv('T_ONTIME_REPORTING.csv')
    
    # filter the dataset for the chosen airport ord
    print("Filtering for ORD airport...")
    chosen_airport = 'ORD'
    filtered_data = data[data['ORIGIN'] == chosen_airport]
    
    # check for missing values
    print("Checking for missing values...")
    missing_values = filtered_data.isnull().sum()
    print("Missing values before cleaning:")
    print(missing_values)
    
    # remove missing values
    cleaned_data = filtered_data.dropna()
    missing_values = cleaned_data.isnull().sum()
    print("\nMissing values after cleaning:")
    print(missing_values)
    
    # Rename columns to match the expected format
    print("\nRenaming columns...")
    column_mapping = {
        "YEAR": "YEAR",
        "MONTH": "MONTH",
        "DAY_OF_MONTH": "DAY",
        "DAY_OF_WEEK": "DAY_OF_WEEK",
        "ORIGIN": "ORG_AIRPORT",
        "DEST": "DEST_AIRPORT",
        "CRS_DEP_TIME": "SCHEDULED_DEPARTURE",
        "DEP_TIME": "DEPARTURE_TIME",
        "DEP_DELAY": "DEPARTURE_DELAY",
        "CRS_ARR_TIME": "SCHEDULED_ARRIVAL",
        "ARR_TIME": "ARRIVAL_TIME",
        "ARR_DELAY": "ARRIVAL_DELAY"
    }
    
    # Rename columns
    cleaned_data = cleaned_data.rename(columns=column_mapping)
    
    # Keep required columns
    columns_to_keep = [
        "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "DEPARTURE_TIME",
        "SCHEDULED_ARRIVAL", "ARRIVAL_TIME", "DEPARTURE_DELAY", "ORG_AIRPORT", "DEST_AIRPORT"
    ]
    cleaned_data = cleaned_data[columns_to_keep]
    
    # save the dataset
    print("\nSaving cleaned dataset...")
    cleaned_data.to_csv("cleaned_data.csv", index=False)
    print("✅ Dataset saved successfully as 'cleaned_data.csv'")
    
    return cleaned_data

if __name__ == "__main__":
    print("Starting data cleaning process...")
    cleaned_data = clean_data()
    print("✅ Data cleaning completed successfully!")