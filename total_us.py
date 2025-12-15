import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from statsmodels.graphics.tsaplots import plot_pacf
from utils import seasonal_plot, plot_periodogram_monthly, plot_lags, lagplot
import streamlit as st

st.sidebar.title('Navigation')

options = st.sidebar.radio('Pages', options=['Home', 'Statistics', 'Seasonal & Fourier Forecast', 'Lag Model'])

def plot_forecast_and_pred(y, y_pred, y_fore):
    """Plot Fourier model forecast"""
    fig, ax = plt.subplots(figsize=(14, 6))
    x = pd.concat([y_pred.iloc[[-1]], y_fore])
    
    ax.plot(y.index, y, label='Actual', color='blue', alpha=0.7, linewidth=1.5)
    ax.plot(y_pred.index, y_pred, label='Fitted (Trend + Seasonal + Fourier)',
            color='orange', linestyle='--', linewidth=2)
    ax.plot(x.index, x, label='Forecast (12 months)',
            color='green', linestyle='--', linewidth=2)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Total Generation (thousand MWh)', fontsize=12)
    ax.set_title('US Electricity Generation - Time Series Decomposition', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def plot_lag_forecast_and_pred(y, y_pred, y_fore, y_test_start):
    """Plot lag model with train/test split"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(y.index, y, label='Actual', color='blue', alpha=0.7, linewidth=1.5)
    ax.plot(y_pred.index, y_pred, label='Train Fitted',
            color='orange', linestyle='--', linewidth=2)
    ax.plot(y_fore.index, y_fore, label='Test Forecast',
            color='green', linestyle='--', linewidth=2)
    ax.axvline(x=y_test_start, color='red', linestyle=':', linewidth=2, label='Train/Test Split')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Total Generation (thousand MWh)', fontsize=12)
    ax.set_title('US Electricity Generation - Lag Model (Train/Test Split)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def plot_rolling_stats(mean, median, std, time, emw = False):

    fig, ax = plt.subplots(figsize=(14, 6))

    if emw:
        ax.set_title(f"{time} Month Exponential Moving Window Statistics")
        ax.plot(mean.index, mean, label = "Mean")
        ax.plot(mean.index, std, color = 'green', label = "Standard Deviation")
        ax.set_xlabel("Time (Months)")
        ax.set_ylabel('Total Generation (thousand MWh)')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        return fig

    else:
        ax.set_title(f"{time} Month Rolling Statistics")
        ax.plot(mean.index, mean, label = "Mean")
        ax.plot(mean.index, median, color= 'blue', label = "Median")
        ax.plot(mean.index, std, color = 'green', label = "Standard Deviation")
        ax.set_xlabel("Time (Months)")
        ax.set_ylabel('Total Generation (thousand MWh)')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        return fig

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

deseasonalized = total - (total_pred)

fourier_r2 = r2_score(total, total_pred)
fourier_rmse = np.sqrt(mean_squared_error(total, total_pred))
fourier_mae = mean_absolute_error(total, total_pred)

print(f"Fourier Model - R²: {fourier_r2:.4f}, RMSE: {fourier_rmse:,.2f}, MAE: {fourier_mae:,.2f}")

X_fore = dp.out_of_sample(steps=12)

total_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

X_lag = make_lags(total, lags=12)

X_lag_deseas = make_lags(deseasonalized, lags=12).dropna()
y_lag_deseas = deseasonalized.loc[X_lag_deseas.index]

# Train/test split
train_size_ses = len(X_lag_deseas) - 60
X_train_ses = X_lag_deseas.iloc[:train_size_ses]
X_test_ses = X_lag_deseas.iloc[train_size_ses:]
y_train_ses = y_lag_deseas.iloc[:train_size_ses]
y_test_ses = y_lag_deseas.iloc[train_size_ses:]

# Fit on deseasonalized data
model_lag_deseas = LinearRegression()
model_lag_deseas.fit(X_train_ses, y_train_ses)

# Predictions
y_pred_train = pd.Series(model_lag_deseas.predict(X_train_ses), index = X_train_ses.index)
y_pred_test = pd.Series(model_lag_deseas.predict(X_test_ses), index = X_test_ses.index)

# Add seasonal component back
seasonal_train = total_pred[y_train_ses.index]
seasonal_test = total_pred[y_test_ses.index]

final_pred_train = y_pred_train + seasonal_train
final_pred_test = y_pred_test + seasonal_test

train_r2 = r2_score(y_train_ses + seasonal_train, final_pred_train)
test_r2 = r2_score(y_test_ses + seasonal_test, final_pred_test)

print(test_r2)
print(train_r2)

X_lag = X_lag.fillna(0.0)

# Create target series and data splits
total_lag = total.copy()

mean_12 = total_lag.rolling(12).mean()
median_12 = total_lag.rolling(12).median()
std_12 = total_lag.rolling(12).std()

mean_6 = total_lag.rolling(6).mean()
median_6 = total_lag.rolling(6).median()
std_6 = total_lag.rolling(6).std()

mean_3 = total_lag.ewm(3).mean()
median_3 = total_lag.rolling(3).median()
std_3 = total_lag.ewm(3).std()

X_train, X_test, y_train, y_test = train_test_split(X_lag, total_lag, test_size=60, shuffle=False)

# Fit and predict
model = LinearRegression() 
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_train), index=y_train.index)
y_fore = pd.Series(model.predict(X_test), index=y_test.index)

train_r2 = r2_score(y_train, y_pred)
test_r2 = r2_score(y_test, y_fore)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_fore))

print(f"\nLag Model - Train R²: {train_r2:.4f}, Train RMSE: {train_rmse:,.2f}")
print(f"Lag Model - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:,.2f}")

if options == 'Home':
    st.title('Time Series Energy Analysis')
    st.subheader("About This Project")
    st.write("""
    This dashboard analyzes monthly US electricity generation using time series forecasting techniques:
    
    - **Statistics**: Explore rolling averages and trends
    - **Seasonal & Fourier Forecast**: Decomposition with Fourier terms (order=2)
    - **Lag Model**: Autoregressive features with 12-month lags
    
    **Data Source**: U.S. Energy Information Administration (EIA)
    """)

if options == 'Statistics':
    st.header("Rolling Statistics")

    st.write("""
    The rolling statistics that I included in this project are a good indicator of the periodic and seasonal nature of energy consumption 
    in the US. The 12 month rolling statistics show a more linear curve that displays how average energy consumptions has changed from year to year.
        """)

    window_option = st.selectbox(
        'Select Time Window',
        options=['3-Month', '6-Month', '12-Month']
    )

    if window_option == '3-Month':
        st.pyplot(plot_rolling_stats(mean_3, median_3, std_3, 3, emw=True))
    elif window_option == '6-Month':
        st.pyplot(plot_rolling_stats(mean_6, median_6, std_6, 6, emw=True))
    elif window_option == '12-Month':
        st.pyplot(plot_rolling_stats(mean_12, median_12, std_12, 12, emw=True))

    st.subheader("Interpretation")
    st.write("""
        - **Mean**: Average generation over the rolling window
        - **Standard Deviation**: Variability in generation
        """)

if options == 'Seasonal & Fourier Forecast':
    st.header("Seasonal & Fourier Decomposition Forecast")

    st.subheader("Method")

    st.write("""
        Energy consumption usually follows a seasonal pattern, with more demand occuring during the winter 
        and summer months due to the need to heat and cool, respictivley.  Fourier features are a way capture these 
        seasonality trends that occur in time series. To choose the order of fourier features to include in the decision process function,
        I looked at a plot of the periodogram, which shows the variance against each season. 
        """)
    st.pyplot(plot_periodogram_monthly(total))

    st.write("""
        There are two large peaks around the annual (1) and semi-annual (2) time frames. Thus I choose my order to be 2 for the fourier features. With my fourier features added,
        I ran the deterministic process with seasonality set set true and order equal to one. I then used a linear regression model to predict the total consumption based off the 
        data from the determinisitic process. I also included a forecast of 12 months into the future. Overall, my R^2 for the fourier seasonal model was .9092, which shows that seasonality
        has a large predictive power for montlhy energy consumption.
        """)
    
    st.pyplot(plot_forecast_and_pred(total, total_pred, total_fore))

if options == 'Lag Model':
    st.header("Lag Model")

    st.subheader("Method")

    st.write("""
        Using a lag model, I investigated whether there was any serial dependence in the energy data. Specifically I was investigating any cycles that
        may be occuring in the data. Unlike seasonality, which occurs with respect to time, cycles can be indpendent of a time variable. Cycles occur with 
        respect to what has happened in the recent past. That is why using a lag model helps interpert cyles. In a lag model data is shifted either forward or 
        backward in time so that itself becomes a feature in the model and we can use it to predict the true value. To investigate how many lags to use in my model,
        I plotted a lag plot, which shows the data plotted against the amount of lags, and a correlelagram, which shows the correlation of a lag that accounts for previous
        lags.  
    """)

    st.pyplot(plot_lags(total, lags = 12, nrows=4))

    st.write("""
        From the lag plot we can see that there is a large amount of autocorrelation between lags 1, 11 and 12. Autocorellation measures linearity though,
        and it is clear from the lag plots that a lot of the fits are non-linear. This is why its helpful to look at the plots themselves and not just at the 
        autocorrelation.
    """)

    st.pyplot(plot_pacf(total, lags = 12))
    st.write("""
        The correlelogram shows that most of the lags fall out of the interval of no correlation (the blue shaded regtion). This is why I ended up choosing to include
        12 lags. 
    """)
    st.write("""
        With the amount of lags choosen, I moved ahead to devloping my lag model. With a lag model, we can only forecast into the future with the data we have available from the lags.
        This is different from the fourier and seasonal forcasting because those models were a function of time and thus we could extend them as far as we wanted. Becuase the lag features
        are a function of previous data (the lags) we can only forecast out to how many lags we incorportated. Instead of doing that though, I just used sci-kit learn to split the data up into training and testing data
        and trained a linear regression model with the training data and forecasted with the testing data. The results are below.
    """)
    st.pyplot(plot_lag_forecast_and_pred(total_lag, y_pred, y_fore, y_test.index[0]))
    
    st.write("""
        The problem with this lag model though is that before I fitted the model to the data, I did not deseasonalize the initial data. The result is that the performance of my lag model, with and R^2 of 0.6983 for the test data, is much lower than my fourier model, which
        has an R^2 of .9092. Becuase I did not deseasonalize the data, the lags do not effectivley capture the non seasonal cycles since the predominant trend seen in the data is seasonal. To deseasonalize the data I subtracted the total predicted values of my fourier model from 
        the actual values. This left me with a series of the residuals from the seasonal prediction which I could now use in my Lag model. With the new series I went ahead and created a lag model like I did before, and the result was that
        I achieved a higher R^2 of 0.9436 for my test data. So overall, by deseasonalizing my data and applying a LAG model, I achieve a 4 percent increase in my R^2.
    """)

    st.pyplot(plot_lag_forecast_and_pred(total_lag, y_pred_train, y_pred_test, y_test.index[0]))
