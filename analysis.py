import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.stats import norm, shapiro
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import io
import base64
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf





# Load price data and daily returns
prices = pd.read_csv("multi_crypto_prices.csv", parse_dates=["Date"])
returns = pd.read_csv("multi_crypto_returns.csv", parse_dates=["Date"])

# Quick check
# print("Prices:")
# print(prices.head())
#
# print("\nReturns:")
# print(returns.head())
#


# Function 1: all prices
def plot_all_prices(prices_df):
    df_long = prices_df.melt(id_vars='Date', var_name='Cryptocurrency', value_name='Price')
    df_long['Cryptocurrency'] = df_long['Cryptocurrency'].str.replace('_price', '')
    fig = px.line(df_long, x='Date', y='Price', color='Cryptocurrency',
                  title='Cryptocurrency Prices Over Time')
    return fig

# Function 2: all returns
def plot_all_returns(returns_df):
    df_long = returns_df.melt(id_vars='Date', var_name='Cryptocurrency', value_name='Return')
    df_long['Cryptocurrency'] = df_long['Cryptocurrency'].str.replace('_return', '')
    fig = px.line(df_long, x='Date', y='Return', color='Cryptocurrency',
                  title='Cryptocurrency Daily Returns Over Time')
    return fig

# Function 3: prices without BTC, ETH, and SOL
def plot_prices_no_btc_eth(prices_df):
    cols = [col for col in prices_df.columns if
            col != 'Date' and not (col.startswith('bitcoin') or col.startswith('ethereum') or col.startswith('solana'))]
    df_filtered = prices_df[['Date'] + cols]
    df_long = df_filtered.melt(id_vars='Date', var_name='Cryptocurrency', value_name='Price')
    df_long['Cryptocurrency'] = df_long['Cryptocurrency'].str.replace('_price', '')
    fig = px.line(df_long, x='Date', y='Price', color='Cryptocurrency',
                  title='Cryptocurrency Prices Over Time (Excluding BTC, ETH, SOL)')
    return fig

def plot_returns_correlation(returns_df):
    # Drop 'Date' column and compute correlation matrix
    corr_matrix = returns_df.drop(columns=['Date']).corr()

    # Optional: rename columns to remove "_return"
    corr_matrix.columns = [col.replace('_return', '') for col in corr_matrix.columns]
    corr_matrix.index = [idx.replace('_return', '') for idx in corr_matrix.index]

    # Create heatmap
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale='Viridis',
        showscale=True,
        hoverinfo="z"
    )
    fig.update_layout(title="Correlation Matrix of Daily Returns")
    return fig

# Compute correlation matrix on prices
price_corr = prices.drop(columns='Date').corr()

# Create a heatmap
fig_price_corr = ff.create_annotated_heatmap(
    z=price_corr.values.round(2),
    x=[col.replace('_price', '') for col in price_corr.columns],
    y=[row.replace('_price', '') for row in price_corr.index],
    colorscale='Viridis',
    showscale=True
)
fig_price_corr.update_layout(title='Correlation Matrix of Cryptocurrency Prices')

# Save the figure to a variable for Dash
price_correlation_fig = fig_price_corr



def plot_return_histograms(returns_df):
    """
    Create histogram plots of daily returns for each cryptocurrency,
    including a fitted normal distribution curve.
    Returns a list of Plotly figures, one per cryptocurrency.
    """
    figures = []

    for col in returns_df.columns:
        if col != 'Date':
            data = returns_df[col].dropna()  # Remove missing values
            fig = go.Figure()

            # Add histogram of returns (normalized to match PDF scale)
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=100,
                marker_color='blue',
                histnorm='probability density'  # Normalize to match normal distribution
            ))

            # Fit and add normal distribution curve
            mean = data.mean()
            std = data.std()
            x_range = np.linspace(data.min(), data.max(), 300)
            y_pdf = norm.pdf(x_range, mean, std)

            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_pdf,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red')
            ))

            # Update layout
            fig.update_layout(
                title=f'Histogram of {col.replace("_return", "").capitalize()} Daily Returns',
                xaxis_title='Daily Return',
                yaxis_title='Density',
                bargap=0.1
            )

            figures.append(fig)

    return figures


def shapiro_test_summary(returns_df):
    """
    Run Shapiro-Wilk normality test for each cryptocurrency's returns.
    Returns a DataFrame with the results.
    """
    results = []

    for col in returns_df.columns:
        if col != 'Date':
            stat, p_value = shapiro(returns_df[col])
            results.append({
                'Cryptocurrency': col.replace('_return', '').capitalize(),
                'W Statistic': round(stat, 4),
                'p-value': round(p_value, 4),
                'Normal (p > 0.05)': 'Yes' if p_value > 0.05 else 'No'
            })

    return pd.DataFrame(results)


def plot_stationarity_tests(prices_df, coin_column):
    """
    Create plots and ADF test result for stationarity check of a single cryptocurrency price series.
    Returns Plotly figures and ADF test summary dict.
    """
    series = prices_df[coin_column].dropna()

    # Calculate rolling statistics
    rolling_mean = series.rolling(window=12).mean()
    rolling_std = series.rolling(window=12).std()

    # Calculate Exponential Weighted Moving Average
    exp_weighted_avg = series.ewm(span=12, adjust=False).mean()

    # Perform Augmented Dickey-Fuller test
    adf_result = adfuller(series)
    adf_output = {
        'ADF Statistic': adf_result[0],
        'p-value': adf_result[1],
        'Used Lag': adf_result[2],
        'Number of Observations': adf_result[3]
    }

    # Plot original series with rolling stats
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name='Original Series'))
    fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, mode='lines', name='Rolling Mean'))
    fig.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std, mode='lines', name='Rolling Std Dev'))
    fig.add_trace(go.Scatter(x=exp_weighted_avg.index, y=exp_weighted_avg, mode='lines', name='Exp Weighted Mean'))

    fig.update_layout(
        title=f'Stationarity Check for {coin_column}',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0, y=1)
    )

    return fig, adf_output



def plot_pacf_figure(series, lags=40):
    """
    Generate a PACF plot as a Plotly figure, suitable for Dash.

    Parameters:
    - series: pandas Series (e.g., returns['bitcoin_return'])
    - lags: number of lags to display

    Returns:
    - plotly.graph_objects.Figure
    """
    series = series.dropna()
    pacf_vals = pacf(series, nlags=lags, method='ywm')

    x = list(range(1, lags + 1))
    y = pacf_vals[1:]  # skip lag 0

    # 95% confidence interval
    conf_int = 1.96 / (len(series) ** 0.5)

    fig = go.Figure()

    # PACF bars
    fig.add_trace(go.Bar(x=x, y=y, name='PACF'))

    # Confidence intervals
    fig.add_shape(type='line', x0=0, x1=lags, y0=conf_int, y1=conf_int,
                  line=dict(color='red', dash='dash'))
    fig.add_shape(type='line', x0=0, x1=lags, y0=-conf_int, y1=-conf_int,
                  line=dict(color='red', dash='dash'))

    fig.update_layout(
        title="Partial Autocorrelation Function (PACF)",
        xaxis_title="Lag",
        yaxis_title="Partial Autocorrelation",
        template="plotly_white"
    )

    return fig

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf


def suggest_p_from_pacf(series, nlags=40, alpha=0.05):
    # Compute PACF values and confidence intervals
    pacf_vals = pacf(series.dropna(), nlags=nlags, alpha=alpha)
    pacf_coef = pacf_vals[0]
    conf_int = pacf_vals[1]

    # Extract upper and lower confidence bounds
    upper_bounds = conf_int[:, 1]
    lower_bounds = conf_int[:, 0]

    # Identify significant lags (those where PACF value lies outside the confidence interval)
    significant_lags = [
        i for i in range(1, nlags + 1)
        if pacf_coef[i] < lower_bounds[i] or pacf_coef[i] > upper_bounds[i]
    ]

    # Suggest p as the smallest significant lag, or 0 if none found
    suggested_p = min(significant_lags) if significant_lags else 0

    # Create interactive PACF plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(pacf_coef))), y=pacf_coef, name='PACF'))
    fig.add_trace(go.Scatter(x=list(range(len(pacf_coef))), y=upper_bounds,
                             line=dict(color='red', dash='dot'), name='Upper CI'))
    fig.add_trace(go.Scatter(x=list(range(len(pacf_coef))), y=lower_bounds,
                             line=dict(color='red', dash='dot'), name='Lower CI'))

    fig.update_layout(title=f'PACF Plot (Suggested p = {suggested_p})',
                      xaxis_title='Lag',
                      yaxis_title='Partial Autocorrelation',
                      template='plotly_white')

    # Interpretation summary
    summary_text = f"""
    Based on the PACF analysis, significant autocorrelation was detected at lag(s): {significant_lags if significant_lags else 'none'}.
    Therefore, the suggested value for parameter p is: {suggested_p}.
    """

    return fig, summary_text






import numpy as np
import plotly.graph_objs as go
from statsmodels.tsa.stattools import acf

def create_acf_figure(series, lags=40):
    n = len(series)
    acf_vals = acf(series, nlags=lags, fft=False)
    conf_level = 1.96 / np.sqrt(n)  # 95% confidence interval
    lags_array = np.arange(len(acf_vals))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=lags_array, y=acf_vals, name='ACF'))
    fig.add_trace(go.Scatter(x=lags_array, y=[conf_level]*len(acf_vals),
                             mode='lines', name='Upper CI', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=lags_array, y=[-conf_level]*len(acf_vals),
                             mode='lines', name='Lower CI', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Autocorrelation Function (ACF)',
                      xaxis_title='Lag',
                      yaxis_title='ACF',
                      bargap=0.1)
    return fig

from arch import arch_model


def fit_garch_model(series, p=1, q=1):
    """
    Fit a GARCH(p, q) model to a return series and return forecast + plots.
    """
    series = series.dropna()

    model = arch_model(series, vol='GARCH', p=p, q=q, rescale=False)
    fitted_model = model.fit(disp='off')

    # Forecast next 30 steps
    forecasts = fitted_model.forecast(horizon=30)
    mean_forecast = forecasts.mean.iloc[-1]
    volatility_forecast = forecasts.variance.iloc[-1] ** 0.5

    # Create a DataFrame for Dash Table
    forecast_df = pd.DataFrame({
        "Step": range(1, 31),
        "Mean Forecast": mean_forecast.values,
        "Volatility Forecast": volatility_forecast.values
    })

    # Plot: Volatility Forecast
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=forecast_df["Step"],
        y=forecast_df["Volatility Forecast"],
        mode='lines+markers',
        name='Volatility Forecast'
    ))
    fig_vol.update_layout(
        title="30-Day Volatility Forecast (GARCH)",
        xaxis_title="Forecast Horizon",
        yaxis_title="Volatility",
        template="plotly_white"
    )

    return forecast_df, fitted_model.summary().as_text(), fig_vol
