import pandas as pd
import numpy as np

# Read the datasets
tweets_df = pd.read_csv('/Users/ishitajain/Downloads/dataset_52-person-from-2021-02-05_2023-06-12_21-34-17-266_with_sentiment.csv')
price_df = pd.read_csv('/Users/ishitajain/Downloads/Bitcoin Historical Data.csv')

# Data Preprocessing
tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'])
price_df['date_column'] = pd.to_datetime(price_df['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')

# Function to convert K/M notation to numeric
def convert_volume(vol_str):
    if pd.isna(vol_str):
        return np.nan
    vol_str = str(vol_str).replace(',', '')
    if vol_str.endswith('K'):
        return float(vol_str[:-1]) * 1000
    elif vol_str.endswith('M'):
        return float(vol_str[:-1]) * 1000000
    elif vol_str.endswith('B'):
        return float(vol_str[:-1]) * 1000000000
    return float(vol_str)

price_df['price'] = price_df['Price'].str.replace(',', '').astype(float)
price_df['open_price'] = price_df['Open'].str.replace(',', '').astype(float)
price_df['high_price'] = price_df['High'].str.replace(',', '').astype(float)
price_df['low_price'] = price_df['Low'].str.replace(',', '').astype(float)
price_df['volume'] = price_df['Vol.'].apply(convert_volume)
price_df['price_change_pct'] = price_df['Change %'].str.replace('%', '').astype(float)

# Aggregate tweet sentiment and volume per day
daily_tweet_sentiment = tweets_df.groupby(tweets_df['created_at'].dt.date).agg({
    'sentiment_type': [
        ('positive_ratio', lambda x: (x == 'POSITIVE').mean()),
        ('volume', 'count')
    ],
    'favorite_count': 'sum',
    'retweet_count': 'sum'
})

# Flatten the multi-level column index
daily_tweet_sentiment.columns = ['_'.join(col).strip() for col in daily_tweet_sentiment.columns.values]
daily_tweet_sentiment = daily_tweet_sentiment.rename(columns={
    'favorite_count_sum': 'favorite_count',
    'retweet_count_sum': 'retweet_count'
}).reset_index()

daily_tweet_sentiment['Date'] = pd.to_datetime(daily_tweet_sentiment['created_at']).dt.strftime('%m/%d/%Y')

print(daily_tweet_sentiment.head())
print(price_df.tail())
print(daily_tweet_sentiment['Date'])
print(price_df['Date'])
# Merge price data with tweet sentiment using a left join
merged_df = price_df.copy()
merged_df = merged_df.merge(
    daily_tweet_sentiment[
        [ 'Date',
         'sentiment_type_positive_ratio', 
         'favorite_count', 
         'retweet_count', 
         'sentiment_type_volume']
    ], 
    on='Date', 
    how='inner'
)



# Fill missing sentiment-related data with 0s
sentiment_columns = [
    'sentiment_type_positive_ratio', 
    'favorite_count', 
    'retweet_count', 
    'sentiment_type_volume'
]
merged_df[sentiment_columns] = merged_df[sentiment_columns].fillna(0)

# Forward-fill and backward-fill to ensure all price data is complete
price_columns = ['price', 'open_price', 'high_price', 'low_price', 'volume', 'price_change_pct']
merged_df[price_columns] = merged_df[price_columns].fillna(method='ffill').fillna(method='bfill')

# Feature Engineering
merged_df['price_7d_ma'] = merged_df['price'].rolling(window=7).mean().shift(1)
merged_df['price_30d_ma'] = merged_df['price'].rolling(window=30).mean().shift(1)
merged_df['price_7d_volatility'] = merged_df['price'].rolling(window=7).std().shift(1)
merged_df['price_momentum_7d'] = (merged_df['price'] / merged_df['price'].shift(7) - 1).shift(1)

# RSI Calculation
def calculate_rsi(data, periods=14):
    delta = data.diff()
    increase_avg = delta.clip(lower=0).rolling(window=periods).mean()
    decrease_avg = -delta.clip(upper=0).rolling(window=periods).mean()
    relative_strength = increase_avg / decrease_avg
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    return rsi

merged_df['rsi_14d'] = calculate_rsi(merged_df['price']).shift(1)
merged_df['vol_7d_ma'] = merged_df['volume'].rolling(window=7).mean().shift(1)
merged_df['vol_change_ratio'] = (merged_df['volume'] / merged_df['volume'].shift(1) - 1).shift(1)
merged_df['next_day_price_change'] = merged_df['price'].shift(-1) / merged_df['price'] - 1
merged_df['tweet_engagement_ratio'] = (
    (merged_df['favorite_count'] + merged_df['retweet_count']) / 
    merged_df['sentiment_type_volume'].replace(0, np.nan)
).fillna(0)
merged_df['ema_12d'] = merged_df['price'].ewm(span=12).mean().shift(1)
merged_df['ema_26d'] = merged_df['price'].ewm(span=26).mean().shift(1)

# Remove the last row as it won't have a next day price change
merged_df = merged_df.iloc[:-1]

# Save to Stata dataset
merged_df.to_stata('/Users/ishitajain/Downloads/bitcoin_merged_dataset1.dta', write_index=False)

# Print basic information about the merged dataset
print(merged_df.info())
print("\nFirst few rows:\n", merged_df.head())
print("\nDescriptive statistics:\n", merged_df.describe())
