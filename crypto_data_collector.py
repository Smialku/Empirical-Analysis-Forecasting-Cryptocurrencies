# crypto_data_collector.py

import yfinance as yf
import pandas as pd

# List of cryptocurrencies with Yahoo Finance tickers
coins = {
    'bitcoin': 'BTC-USD',
    'ethereum': 'ETH-USD',
    'solana': 'SOL-USD',
    'cardano': 'ADA-USD',
    'ripple': 'XRP-USD'
}

# Date range for historical data
start_date = '2019-01-01'
end_date = '2024-12-31'

dataframes = {}

for name, ticker in coins.items():
    print(f"Fetching data for: {name} ({ticker})")
    df = yf.download(ticker, start=start_date, end=end_date)[['Close']]
    df = df.reset_index()  # Ensure 'Date' is a column
    df.columns = ['Date', f'{name}_price']  # Rename columns properly
    dataframes[name] = df

# Merge dataframes on date
merged = dataframes[list(coins.keys())[0]]
for name in list(coins.keys())[1:]:
    merged = pd.merge(merged, dataframes[name], on='Date', how='outer')

# Sort and save to file
merged = merged.sort_values('Date')
merged = merged.dropna()

merged.to_csv("multi_crypto_prices.csv", index=False)

# Calculate daily returns
daily_returns = merged.set_index('Date').pct_change().dropna()

# Rename columns from e.g. bitcoin_price -> bitcoin_return
daily_returns.columns = [col.replace('_price', '_return') for col in daily_returns.columns]

# Reset index and save
daily_returns = daily_returns.reset_index()
daily_returns.to_csv("multi_crypto_returns.csv", index=False)

print("Done! Data saved as CSV.")
