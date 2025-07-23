# Empirical Analysis and Forecasting of Major Cryptocurrencies Using Historical Data

This repository contains Python scripts and data files for analyzing and forecasting cryptocurrency prices, including models like ARIMA and GARCH.

## Files

- `dashapp.py` - Dash application for interactive visualization
- `analysis.py` - Data analysis scripts
- `arima_model.py` - ARIMA model implementation
- `crypto_data_collector.py` - Script for collecting historical cryptocurrency data
- `multi_crypto_prices.csv` - Historical price data
- `multi_crypto_returns.csv` - Calculated returns data

## Usage


# Clone the repository
git clone https://github.com/Smialku/Empirical-Analysis-Forecasting-Cryptocurrencies.git
cd Empirical-Analysis-Forecasting-Cryptocurrencies

# (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Dash app
python dashapp.py

Open the app in your browser:
Go to http://127.0.0.1:8050

