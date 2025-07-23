import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def fit_arima_model(returns_series, order=(0, 0, 0), forecast_steps=30):
    """
    Fits an ARIMA model to the provided returns series and generates forecasts.

    Parameters:
    - returns_series (pd.Series): Time series data of returns
    - order (tuple): ARIMA model order (p, d, q)
    - forecast_steps (int): Number of steps to forecast

    Returns:
    - forecast_df (pd.DataFrame): DataFrame with forecast values and confidence intervals
    - model_summary (str): Text summary of model fit
    """
    model = ARIMA(returns_series, order=order)
    fitted_model = model.fit()

    forecast = fitted_model.get_forecast(steps=forecast_steps)
    forecast_df = forecast.summary_frame()

    return forecast_df, fitted_model.summary().as_text()
