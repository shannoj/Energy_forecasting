import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from statsmodels.graphics.tsaplots import plot_pacf
from utils import seasonal_plot, plot_periodogram_monthly, plot_lags, lagplot

def plot_forecast_and_pred(y, y_pred, y_fore):

    x = pd.concat([y.iloc[[-1]], y_fore])

    plt.figure(figsize=(14, 6))
    plt.plot(y.index, y, label='Actual', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(y.index, y_pred, label='Fitted (Trend + Seasonal + Fourier)',
            color='orange', linestyle='--', linewidth=2)
    plt.plot(x.index, x, label='Forecast (12 months)',
            color='green', linestyle='--', linewidth=2)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Generation (thousand MWh)', fontsize=12)
    plt.title('US Electricity Generation - Time Series Decomposition', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_lag_forecast_and_pred(y, y_pred, y_fore):

    plt.figure(figsize=(14, 6))
    plt.plot(total.index, total, label='Actual', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(y_pred.index, y_pred, label='Train Fitted',
            color='orange', linestyle='--', linewidth=2)
    plt.plot(y_fore.index, y_fore, label='Test Forecast',
            color='green', linestyle='--', linewidth=2)
    plt.axvline(x=y_test.index[0], color='red', linestyle=':', linewidth=2, label='Train/Test Split')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Generation (thousand MWh)', fontsize=12)
    plt.title('US Electricity Generation - Lag Model (Train/Test Split)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)

df = pd.read_csv('Net_generation_United_States_all_sectors_monthly.csv', header=4)

df['Month'] = pd.to_datetime(df['Month'], format='%b %Y')

df_renamed = df.rename(columns={'all fuels (utility-scale) thousand megawatthours': 'Total'})

columns = df_renamed.columns.to_list()

df_renamed['Total'] = pd.to_numeric(df_renamed['Total'])

# Set Month as the index to create a DatetimeIndex
df_renamed = df_renamed.set_index('Month')

total = df_renamed['Total']

total = total.asfreq('MS')

total = total.dropna()

fourier = CalendarFourier(freq="MS", order = 2)

dp = DeterministicProcess(
    index=total.index,
    constant=True,
    order=1,
    seasonal = True,
    additional_terms = [fourier],
    drop=True,
)

X = dp.in_sample()

model = LinearRegression().fit(X, total)

total_pred = pd.Series(
    model.predict(X),
    index=X.index,
    name='Fitted',
)

X_fore = dp.out_of_sample(steps=12)

total_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

X_lag = make_lags(total, lags=12)

X_lag = X_lag.fillna(0.0)

# Create target series and data splits
total_lag = total.copy()

X_train, X_test, y_train, y_test = train_test_split(X_lag, total_lag, test_size=60, shuffle=False)

# Fit and predict
model = LinearRegression()  # `fit_intercept=True` since we didn't use DeterministicProcess
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_train), index=y_train.index)
y_fore = pd.Series(model.predict(X_test), index=y_test.index)

plot_lag_forecast_and_pred(total, y_pred, y_fore)