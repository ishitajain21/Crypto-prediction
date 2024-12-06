import pandas as pd
import numpy as np

# Read the existing Bitcoin dataset
bitcoin_df = pd.read_stata('/Users/ishitajain/Downloads/bitcoin_merged_dataset.dta')

# Read the new Gold Price dataset
gold_df = pd.read_csv('/Users/ishitajain/Downloads/Gold Price Prediction.csv')

# Ensure the date column is in datetime format
bitcoin_df['date_column'] = pd.to_datetime(bitcoin_df['date_column'])
gold_df['Date'] = pd.to_datetime(gold_df['Date'])

# Rename the date column in gold dataset to match bitcoin dataset
gold_df = gold_df.rename(columns={
    'Date': 'date_column',
    'Std Dev 10': 'Std Dev Ten',
    'Fifty Day Moving Average': 'Fifty Moving Average', 
    '200 Day Moving Average': 'Two Hundred Moving Average',
    'Volume ': 'Volume',
    'Treasury Par Yield Curve Rates (10 Yr)': 'Treasury Par Yield Ten Year'
})

# Merge the datasets
# We want to keep only the rows from the Bitcoin dataset and merge corresponding Gold data
merged_df = bitcoin_df.merge(gold_df, on='date_column', how='left')

# Specify the numerical columns to fill
numerical_columns = [
    'Price 2 Days Prior', 'Price 1 Day Prior', 'Price Today', 'Price Tomorrow', 
    'Price Change Tomorrow', 'Price Change Ten', 'Std Dev Ten', 
    'Twenty Moving Average', 'Fifty Moving Average', 'Two Hundred Moving Average', 
    'Monthly Inflation Rate', 'EFFR Rate', 'Volume', 
    'Treasury Par Yield Month', 'Treasury Par Yield Two Year', 
    'Treasury Par Yield Ten Year', 'DXY', 'SP Open', 'VIX', 'Crude'
]

# Forward-fill and backward-fill for these columns
merged_df[numerical_columns] = merged_df[numerical_columns].fillna(method='ffill').fillna(method='bfill')

# Remove the last row if it has any NaN values after filling
merged_df = merged_df.dropna()

# Save to Stata dataset
merged_df.to_stata('/Users/ishitajain/Downloads/bitcoin_gold_merged_dataset1.dta', write_index=False)

# Print information about the merged dataset
print(merged_df.info())
print("\nFirst few rows:\n", merged_df.head())
print("\nDescriptive statistics of new columns:\n", merged_df[numerical_columns].describe())
