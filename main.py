import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

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

for hub_name, df in montly_avg_loc.items():

    avg_year = df['mean'].mean()

    if avg_year > most_expensive:
       most_expensive = avg_year
       most_expensive_hub = hub_name
    
    if avg_year < least_expensive:
        least_expensive = avg_year
        least_expensive_hub = hub_name


print(f'Most Expensive on Avg: {most_expensive_hub}, at {round(most_expensive, 2)}$ per month')

print(f"Least Expensive on Avg: {least_expensive_hub}, at {round(least_expensive, 2)}$ per month")





