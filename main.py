import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.tsa.deterministic import DeterministicProcess

df = pd.read_excel('ice_electric-2025.xlsx', header=2)

df.columns = df.columns.str.replace('\n', ' ').str.strip()

columns = df.columns.to_list()

df['Trade date'] = pd.to_datetime(df['Trade date'], format='%m/%d/%y', errors='coerce')
df['Delivery start date'] = pd.to_datetime(df['Delivery start date'],format='%m/%d/%y', errors='coerce')

hubs = df['Price hub'].unique()

df_midc = df[df['Price hub'].str.contains('Mid C', case=False, na=False)].copy()

loc_df = {}
montly_avg_loc = {}

most_expensive = 0
least_expensive = 1000000000

most_expensive_hub = ''

least_expensive_hub = ''

for loc in hubs:
    loc_df[loc] = df[df['Price hub'].str.contains(loc, case=False, na=False)].copy()

for hub_name, df in loc_df.items():

    df = df.sort_values('Trade date').copy()
    # Create time-based features
    df['Year'] = df['Trade date'].dt.year
    df['Month'] = df['Trade date'].dt.month
    df['Day'] = df['Trade date'].dt.day
    df['DayOfWeek'] = df['Trade date'].dt.dayofweek
    df['Week'] = df['Trade date'].dt.isocalendar().week

    loc_df[hub_name] = df


# Second loop: Print analysis for each hub
for hub_name, df in loc_df.items():
    print("=" * 60)
    print(f"ANALYSIS FOR: {hub_name}")
    print("=" * 60)
    
    # Date gaps
    print("\nDate gaps analysis:")
    df['Date_diff'] = df['Trade date'].diff()
    gaps = df[df['Date_diff'] > pd.Timedelta(days=1)][['Trade date', 'Date_diff']]
    if len(gaps) > 0:
        print(gaps.head(10))
    else:
        print("No significant date gaps")

    # Monthly stats
    print("\nAverage price by month:")
    monthly_avg = df.groupby('Month')['Wtd avg price $/MWh'].agg(['mean', 'std', 'min', 'max'])
    montly_avg_loc[hub_name] = monthly_avg.copy()
    print(monthly_avg)

    # Spikes
    print("\nPrice spikes (> $100/MWh):")
    spikes = df[df['Wtd avg price $/MWh'] > 100]
    if len(spikes) > 0:
        print(spikes[['Trade date', 'Wtd avg price $/MWh', 'Daily volume MWh']])
    else:
        print("No price spikes > $100/MWh")
    
    print("\n")

# # Create figure with subplots before the loop
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.tsa.deterministic import DeterministicProcess

# [Your data loading code stays the same...]

# Create figure with subplots
num_hubs = len(montly_avg_loc)
fig, axes = plt.subplots(num_hubs, 1, figsize=(12, 6*num_hubs))

if num_hubs == 1:
    axes = [axes]

for idx, (hub_name, df) in enumerate(montly_avg_loc.items()):
    
    # Calculate average for most/least expensive tracking
    avg_year = df['mean'].mean()
    
    if avg_year > most_expensive:
       most_expensive = avg_year
       most_expensive_hub = hub_name
    
    if avg_year < least_expensive:
        least_expensive = avg_year
        least_expensive_hub = hub_name
    
    # Reset index and create time variable
    df = df.reset_index()
    
    # Use DeterministicProcess for polynomial trend
    dp = DeterministicProcess(
        index=df.index,
        order=1 
    )
    
    X = dp.in_sample()

    X_fore = dp.out_of_sample(steps=3)
    
    y = df['mean']
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    y_pred = pd.Series(model.predict(X), index=X.index)
    y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
    
    # Plot
    ax = axes[idx]
    
    # Actual data
    ax.plot(df.index, y, 'o-', label='Actual', color='blue', linewidth=2)
    
    # Polynomial fit
    ax.plot(df.index, y_pred, '--', label='Polynomial Trend', 
            color='green', linewidth=2, alpha=0.7)
    
    # Forecast
    forecast_index = np.arange(len(df), len(df) + len(y_fore))
    ax.plot(forecast_index, y_fore, ':', label='Forecast', 
            color='red', linewidth=2, marker='s')
    
    # Formatting
    ax.set_xlabel('Month Index', fontsize=11)
    ax.set_ylabel('Avg Price ($/MWh)', fontsize=11)
    ax.set_title(f'{hub_name} - Monthly Average Price Forecast', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add vertical line to separate historical from forecast
    ax.axvline(x=len(df)-0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('hub_forecasts.png', dpi=300, bbox_inches='tight')
plt.show()


