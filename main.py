import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_excel('ice_electric-2025.xlsx', header=2)

df.columns = df.columns.str.replace('\n', ' ').str.strip()

columns = df.columns.to_list()

df['Trade date'] = pd.to_datetime(df['Trade date'], errors='coerce')
df['Delivery start date'] = pd.to_datetime(df['Delivery start date'], errors='coerce')

df_midc = df[df['Price hub'].str.contains('Mid C', case=False, na=False)].copy()

# Sort by date
df_midc = df_midc.sort_values('Trade date')

# Create time-based features
df_midc['Year'] = df_midc['Trade date'].dt.year
df_midc['Month'] = df_midc['Trade date'].dt.month
df_midc['Day'] = df_midc['Trade date'].dt.day
df_midc['DayOfWeek'] = df_midc['Trade date'].dt.dayofweek
df_midc['Week'] = df_midc['Trade date'].dt.isocalendar().week

# Date gaps
print("Date gaps analysis:")
df_midc['Date_diff'] = df_midc['Trade date'].diff()
print(df_midc[df_midc['Date_diff'] > pd.Timedelta(days=1)][['Trade date', 'Date_diff']].head(10))

# Monthly stats
print("\nAverage price by month:")
monthly_avg = df_midc.groupby('Month')['Wtd avg price $/MWh'].agg(['mean', 'std', 'min', 'max'])
print(monthly_avg)

# Spikes
print("\nPrice spikes (> $100/MWh):")
spikes = df_midc[df_midc['Wtd avg price $/MWh'] > 100]
print(spikes[['Trade date', 'Wtd avg price $/MWh', 'Daily volume MWh']])

# Split data: use data up to Sept 2025 for training, rest for testing
train_cutoff = pd.Timestamp('2025-09-01')
train = df_midc[df_midc['Trade date'] < train_cutoff].copy()
test = df_midc[df_midc['Trade date'] >= train_cutoff].copy()



