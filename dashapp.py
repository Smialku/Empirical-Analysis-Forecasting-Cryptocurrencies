import dash
from dash import html, dcc, dash_table
import pandas as pd
from analysis import plot_all_prices, plot_all_returns, plot_prices_no_btc_eth,plot_returns_correlation,price_correlation_fig,plot_return_histograms,shapiro_test_summary
from analysis import plot_stationarity_tests,plot_pacf_figure,suggest_p_from_pacf,create_acf_figure, fit_garch_model
from arima_model import fit_arima_model


# Load data
prices = pd.read_csv("multi_crypto_prices.csv", parse_dates=["Date"])
returns = pd.read_csv("multi_crypto_returns.csv", parse_dates=["Date"])

# Create static figures using functions from analysis.py
fig_all_prices = plot_all_prices(prices)
fig_all_returns = plot_all_returns(returns)
fig_no_btc_eth = plot_prices_no_btc_eth(prices)
fig_corr_matrix = plot_returns_correlation(returns)
histograms = plot_return_histograms(returns)
shapiro_df = shapiro_test_summary(returns)
# ADF test for Bitcoin returns
fig_stationarity_ret, adf_ret = plot_stationarity_tests(returns, 'bitcoin_return')

# Prepare stationarity plot and ADF test summary for Bitcoin
fig_stationarity_btc, adf_btc = plot_stationarity_tests(prices, 'bitcoin_price')

pacf_fig = plot_pacf_figure(returns['bitcoin_return'])

fig_pacf, pacf_interpretation_text = suggest_p_from_pacf(returns['bitcoin_return'])

# Fit ARIMA model to Bitcoin returns (assuming column: 'bitcoin_return')
btc_returns = returns['bitcoin_return'].dropna()
forecast_df, model_summary = fit_arima_model(btc_returns, order=(0, 0, 0), forecast_steps=30)

# GARCH model
garch_forecast_df, garch_summary, fig_garch_vol = fit_garch_model(btc_returns)





# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Cryptocurrency Dashboard"

# Summary text about histograms
summary_text = """
All histograms show that most values are concentrated around zero, which is expected because returns can be both positive and negative.
The histograms suggest that the data approximately follow a normal distribution, which is helpful but not strictly required for further analysis.
To gain more insight, Shapiro-Wilk tests will be conducted in the next step to statistically assess normality..
"""

# Layout: stacked charts
app.layout = html.Div([
    html.H1("Cryptocurrency Dashboard", style={'textAlign': 'center'}),

    html.Div([




        # Time series plots
        dcc.Graph(figure=fig_all_prices),
        dcc.Graph(figure=fig_no_btc_eth),
        dcc.Graph(figure=fig_all_returns),

        # Correlation heatmaps
        dcc.Graph(figure=fig_corr_matrix),
        dcc.Graph(figure=price_correlation_fig),

        # Histograms of returns
        html.H2("Histograms of Daily Returns", style={'textAlign': 'center'}),
        *[dcc.Graph(figure=fig) for fig in histograms],
        html.Div([
            html.H2("Histogram Summary"),
            html.P(summary_text, style={'fontSize': 16, 'lineHeight': '1.5em', 'maxWidth': '1200px', 'margin': 'auto'})
        ], style={'padding': '20px', 'textAlign': 'center'}),

        html.H2("Shapiro-Wilk Normality Test for Daily Returns", style={'textAlign': 'center'}),
        dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in shapiro_df.columns],
            data=shapiro_df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
        ),

        # Shapiro-Wilk test results summary text
        html.Div([
            html.P(
                "Although the histograms suggested that the data might be approximately "
                "normally distributed, the Shapiro-Wilk test results indicate that none "
                "of the variables fully satisfy the normality assumption. This outcome "
                "suggests deviations from normality that should be taken into account in further analysis."
            )
        ], style={'marginTop': 30, 'marginBottom': 30}),
        html.H2("ARIMA Model Assumptions and Next Steps", style={'textAlign': 'center', 'marginTop': '40px'}),

        html.Div([
            html.P(
                "The ARIMA (AutoRegressive Integrated Moving Average) model requires several key assumptions to be met for reliable forecasting:"
            ),
            html.Ul([
                html.Li(
                    "Stationarity: The time series should be stationary, meaning its statistical properties (mean, variance) do not change over time. Differencing the series can help achieve stationarity."),
                html.Li(
                    "No significant autocorrelation in residuals: After fitting, residuals should behave like white noise (no patterns)."),
                html.Li("Linearity: The relationship between past values and current values is assumed to be linear."),
                html.Li(
                    "Normally distributed residuals: While not strictly required, normally distributed residuals help with inference and confidence intervals."),
            ]),
            html.P(
                "Since the original price data are non-stationary, and returns show deviations from normality, the following steps will be undertaken:"
            ),
            html.Ul([
                html.Li("Apply differencing or transformations to ensure stationarity."),
                html.Li("Use statistical tests (e.g., Augmented Dickey-Fuller) to confirm stationarity."),
                html.Li("Fit ARIMA models on transformed or differenced data."),
                html.Li("Validate residuals for autocorrelation and normality."),
                html.Li("Adjust model parameters as necessary to meet assumptions."),
            ]),
            html.P(
                "These steps will help build a robust forecasting model based on ARIMA methodology."
            ),
        ], style={'maxWidth': '1000px', 'margin': 'auto', 'fontSize': '16px', 'lineHeight': '1.7em', 'padding': '20px'}),





        html.H2("Stationarity Check for Bitcoin Returns", style={'textAlign': 'center'}),

dcc.Graph(figure=fig_stationarity_ret),

html.Div([
    html.P(f"ADF Statistic: {adf_ret['ADF Statistic']:.4f}"),
    html.P(f"p-value: {adf_ret['p-value']:.4f}"),
    html.P(f"Used Lag: {adf_ret['Used Lag']}"),
    html.P(f"Number of Observations: {adf_ret['Number of Observations']}"),
    html.P("Interpretation: " +
           ("The return series is stationary (reject null hypothesis)"
            if adf_ret['p-value'] < 0.05
            else "The return series is non-stationary (fail to reject null hypothesis)"))
], style={'textAlign': 'center', 'marginBottom': '40px'}),


html.Div([
    html.H2("Interpretation of ADF Test for Bitcoin Returns", style={'textAlign': 'center'}),
    html.P(
        "The Augmented Dickey-Fuller (ADF) test returned a very low p-value (0.0000), "
        "which leads to rejecting the null hypothesis of non-stationarity. "
        "This strongly indicates that the Bitcoin daily return series is stationary."
    ),
    html.P(
        "The ADF test statistic was -42.71, which is significantly lower than the critical values "
        "at all standard confidence levels. Such a highly negative value further supports the conclusion "
        "that the return series does not contain a unit root and has stable statistical properties over time."
    )
], style={'textAlign': 'center', 'marginBottom': '40px'}),


html.H2("Stationarity Check for Bitcoin Prices", style={'textAlign': 'center'}),

dcc.Graph(figure=fig_stationarity_btc),

html.Div([
    html.P(f"ADF Statistic: {adf_btc['ADF Statistic']:.4f}"),
    html.P(f"p-value: {adf_btc['p-value']:.4f}"),
    html.P(f"Used Lag: {adf_btc['Used Lag']}"),
    html.P(f"Number of Observations: {adf_btc['Number of Observations']}"),
    html.P("Interpretation: " +
           ("The series is stationary (reject null hypothesis)" if adf_btc['p-value'] < 0.05 else
            "The series is non-stationary (fail to reject null hypothesis)"))
], style={'textAlign': 'center', 'marginBottom': '40px'}),


        html.H2("Interpretation of ADF Test for Bitcoin Prices", style={'textAlign': 'center'}),

html.Div([

    html.P(
        "Interpretation: The p-value is significantly higher than the common significance level of 0.05, "
        "which means we fail to reject the null hypothesis that a unit root is present. "
        "Therefore, the Bitcoin price series is non-stationary. "
        "This is expected for financial price data, and differencing or other transformations will be needed "
        "to make the series stationary before ARIMA modeling."
    )
], style={'textAlign': 'center', 'marginBottom': '40px'}),

html.Div([
    html.P(
        "Stationarity tests were conducted for both price levels and daily returns. "
        "While prices were found to be non-stationary, daily returns passed the ADF test, "
        "indicating stationarity. Therefore, further modeling will be based on returns, "
        "as they better meet the statistical assumptions required for time series forecasting."
    )
], style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '40px'}),

dcc.Graph(figure=pacf_fig),

html.H2("Partial Autocorrelation (PACF) for Bitcoin Returns", style={'textAlign': 'center'}),

dcc.Graph(figure=fig_pacf),

html.Div([
    html.P(pacf_interpretation_text, style={'fontSize': 16, 'textAlign': 'center', 'maxWidth': '1000px', 'margin': 'auto'})
], style={'marginTop': '20px', 'marginBottom': '40px'}),


dcc.Graph(figure=create_acf_figure(returns['bitcoin_return'], lags=40)),
html.H2("ARIMA Parameter Selection Summary", style={'textAlign': 'center'}),

html.Div([
    html.P(
        "After analyzing the stationarity of the Bitcoin return series and examining both the ACF and PACF plots, "
        "the most appropriate ARIMA parameters were determined to be (p=0, d=0, q=0). "
        "This implies that the return series is already stationary (d=0), and no significant autocorrelations were found "
        "at any lag in either the ACF or PACF plots, which suggests that no autoregressive (p=0) or moving average (q=0) components "
        "are necessary for modeling.",
        style={'fontSize': 16, 'lineHeight': '1.7em'}
    ),
    html.P(
        "Such a parameter configuration (ARIMA(0,0,0)) indicates that the data behaves like white noise, "
        "meaning that past values do not help in predicting future values. "
        "In practice, this means that any forecasts generated from such a model would be essentially random, "
        "as the series does not exhibit any predictable patterns.",
        style={'fontSize': 16, 'lineHeight': '1.7em'}
    )
], style={'padding': '20px', 'textAlign': 'center', 'maxWidth': '1000px', 'margin': 'auto'}),

        # Forecast header
        html.H2("ARIMA Forecast (0,0,0)", style={'textAlign': 'center'}),

        # Forecast Table
        dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in forecast_df.reset_index().columns],
            data=forecast_df.reset_index().to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
        ),

        # Model Summary
        html.Div([
            html.H3("ARIMA Model Summary"),
            html.Pre(model_summary, style={'whiteSpace': 'pre-wrap', 'fontSize': '12px'})
        ], style={'padding': '20px'}),
        html.Div([
            html.H2("ARIMA(0,0,0) Model Summary", style={'textAlign': 'center'}),

            html.Div([
                html.P(
                    "Although Bitcoin return data initially appeared promising due to their stationarity and lack of need for transformations, modeling revealed that the returns behave like white noise."
                ),
                html.P(
                    "The ARIMA(0,0,0) model reduces to a simple white noise process with a small non-zero mean: "
                    "rₜ = μ + εₜ, where μ ≈ 0.0020. While the constant is statistically significant (p = 0.009), the model lacks predictive power."
                ),
                html.P("Key statistics from the model analysis:"),
                html.Ul([
                    html.Li("Ljung-Box test (Q = 1.45, p = 0.23): no significant autocorrelation."),
                    html.Li(
                        "Jarque-Bera test (JB = 821.34, p = 0.00): strong non-normality and heavy tails (kurtosis = 6.37)."),
                    html.Li("Heteroskedasticity test: p = 0.00, indicating time-varying volatility."),
                ]),
                html.P(
                    "These results confirm that while the return series is stationary, it lacks meaningful temporal structure. "
                    "The mean is not predictable, and returns resemble pure noise."
                ),
                html.H5("Next Steps", style={'marginTop': '20px'}),
                html.Ul([
                    html.Li("Use GARCH-type models (e.g., GARCH, EGARCH) to model volatility instead of returns."),
                    html.Li("Consider modeling log-prices instead of returns to capture long-term trends."),
                    html.Li(
                        "Apply machine learning methods like LSTM or Random Forests to detect nonlinear relationships."),
                    html.Li(
                        "Introduce exogenous variables (ARIMAX/SARIMAX) such as trading volume or macroeconomic indicators."),
                ]),
                html.P(
                    "In conclusion, although ARIMA cannot forecast Bitcoin returns, advanced models targeting volatility or nonlinear dependencies may provide better insight."
                )
            ], style={'maxWidth': '1000px', 'margin': 'auto', 'fontSize': '16px', 'lineHeight': '1.7em',
                      'padding': '20px'}),

            html.H2("GARCH Volatility Forecast", style={'textAlign': 'center', 'marginTop': '40px'}),

            dcc.Graph(figure=fig_garch_vol),

dash_table.DataTable(
    columns=[{"name": col, "id": col} for col in garch_forecast_df.columns],
    data=garch_forecast_df.to_dict('records'),
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'center'},
    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
),

html.H3("GARCH Model Summary"),
html.Pre(garch_summary, style={'whiteSpace': 'pre-wrap', 'fontSize': '12px'}),
html.H2("GARCH(1,1) Model Summary", style={'textAlign': 'center'}),

html.Div([
    html.P(
        "The GARCH(1,1) model was fitted to daily Bitcoin returns to model volatility. "
        "The model assumes a constant mean and conditional volatility dependent on past observations and variance."
    ),
    html.H4("Key Results:", style={'marginTop': '20px'}),
    html.Ul([
        html.Li("Mean return (μ): 0.00206 – statistically significant (p = 0.003)."),
        html.Li("Parameter ω (omega): 2.04e-5 – baseline for conditional variance."),
        html.Li("Parameter α (alpha): 0.0500 – impact of recent shocks."),
        html.Li("Parameter β (beta): 0.9300 – influence of long-term variance."),
    ]),
    html.P(
        "The high value of β suggests that shocks to volatility have a long-lasting effect, which is typical in financial data. "
        "This model can be used to forecast volatility and assess risk (e.g., Value at Risk)."
    ),
    html.P(
        "The sum α + β = 0.98 indicates strong volatility persistence, "
        "but the process remains stationary since the sum is less than 1."
    ),
    html.H4("Conclusions:"),
    html.P(
        "The GARCH model captures Bitcoin's volatility well. This is important because classical models (e.g., ARIMA) do not account for "
        "time-varying variance. This model will serve as a basis for calculating risk measures such as Value at Risk (VaR)."
    )
], style={'maxWidth': '1000px', 'margin': 'auto', 'fontSize': '16px', 'lineHeight': '1.7em', 'padding': '20px'})


        ])

    ])
])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)