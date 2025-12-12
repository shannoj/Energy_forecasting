import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from utils import seasonal_plot, plot_periodogram

df = pd.read_csv('Net_generation_United_States_all_sectors_monthly.csv', header=4)

df['Month'] = pd.to_datetime(df['Month'], format='%b %Y')

df_renamed = df.rename(columns={'all fuels (utility-scale) thousand megawatthours': 'Total'})

columns = df_renamed.columns.to_list()

df_renamed['Total'] = pd.to_numeric(df_renamed['Total'])

# Set Month as the index to create a DatetimeIndex
df_renamed = df_renamed.set_index('Month')

y = df_renamed['Total']

df_renamed = df_renamed.asfreq('MS')
# Example: Fill NaNs created by asfreq with 0 or an appropriate imputation method
df_renamed['Total'] = df_renamed['Total'].fillna(method='bfill')

fourier = CalendarFourier(freq="YE", order = 10)

dp = DeterministicProcess(
    index=df_renamed.index,
    constant=True,
    order=1,
    seasonal = True,
    additional_terms = [fourier],
    drop=True,
)

X = dp.in_sample()

model = LinearRegression().fit(X, y)

y_pred = pd.Series(
    model.predict(X),
    index=X.index,
    name='Fitted',
)

print(type(df_renamed.index))

X_fore = dp.out_of_sample(steps=30)

y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)


# Plot
plt.figure(figsize=(14, 6))
plt.plot(df_renamed.index, y, label='Actual', color='blue', alpha=0.7, linewidth=1.5)
plt.plot(df_renamed.index, y_pred, label='Fitted (Trend + Seasonal + Fourier)',
         color='orange', linestyle='--', linewidth=2)
plt.plot(y_fore.index, y_fore, label='Forecast (30 months)',
         color='green', linestyle='--', linewidth=2)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Generation (thousand MWh)', fontsize=12)
plt.title('US Electricity Generation - Time Series Decomposition', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
