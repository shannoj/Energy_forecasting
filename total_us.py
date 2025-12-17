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

def cleaning(df):
    df['Month'] = pd.to_datetime(df['Month'], format='%b %Y')

    df_renamed = df.rename(columns={'all fuels (utility-scale) thousand megawatthours': 'Total', 
                                    'coal thousand megawatthours': "Coal",
                                    'natural gas thousand megawatthours': 'Natural Gas',
                                    'nuclear thousand megawatthours':'Nuclear',
                                    'conventional hydroelectric thousand megawatthours': 'Hydroelectric',
                                    'wind thousand megawatthours':'Wind',
                                    'all solar thousand megawatthours':'Solar',
                                    })

    columns = df_renamed.columns.to_list()

    for i in columns:
        if i != 'Month':
            df_renamed[i] = pd.to_numeric(df_renamed[i])
    
    df_renamed = df_renamed.set_index('Month')

    return df_renamed

def select_variable(df, column):

    variable = df[column]

    variable = variable.dropna()

    variable = variable.asfreq("MS")

    return variable

def fourier_seasonal(variable, order):

    fourier = CalendarFourier(freq="MS", order = order)

    dp = DeterministicProcess(
        index=variable.index,
        constant=True,
        order=1,
        seasonal = True,
        additional_terms = [fourier],
        drop=True,
    )

    X = dp.in_sample()

    model = LinearRegression().fit(X, variable)

    variable_pred = pd.Series(
        model.predict(X),
        index=X.index,
        name='Fitted',
    )

    X_fore = dp.out_of_sample(steps=12)

    total_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

    fourier_r2 = r2_score(variable, variable_pred)
    fourier_rmse = np.sqrt(mean_squared_error(variable, variable_pred))
    fourier_mae = mean_absolute_error(variable, variable_pred)

    return variable_pred, X_fore, total_fore, fourier_r2, fourier_rmse, fourier_mae

def deseasonalize(variable, variable_pred):

    return (variable - variable_pred)

def make_deasonal_lag_model(variable_pred, deseasonalized, lags):

    X_lag_deseas = make_lags(deseasonalized, lags=lags).dropna()

    y_lag_deseas = deseasonalized.loc[X_lag_deseas.index]

    train_size_ses = len(X_lag_deseas) - 60
    X_train_ses = X_lag_deseas.iloc[:train_size_ses]
    X_test_ses = X_lag_deseas.iloc[train_size_ses:]
    y_train_ses = y_lag_deseas.iloc[:train_size_ses]
    y_test_ses = y_lag_deseas.iloc[train_size_ses:]

    model_lag_deseas = LinearRegression()
    model_lag_deseas.fit(X_train_ses, y_train_ses)

    y_pred_train = pd.Series(model_lag_deseas.predict(X_train_ses), index = X_train_ses.index)
    y_pred_test = pd.Series(model_lag_deseas.predict(X_test_ses), index = X_test_ses.index)

    seasonal_train = variable_pred[y_train_ses.index]
    seasonal_test = variable_pred[y_test_ses.index]

    final_pred_train = y_pred_train + seasonal_train
    final_pred_test = y_pred_test + seasonal_test

    train_r2 = r2_score(y_train_ses + seasonal_train, final_pred_train)
    test_r2 = r2_score(y_test_ses + seasonal_test, final_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train_ses + seasonal_train, final_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test_ses + seasonal_test, final_pred_test))

    return final_pred_train, final_pred_test, train_r2, test_r2, train_rmse, test_rmse

def make_lag_model(variable, lags):

    X_lag = make_lags(variable, lags=lags)
    
    X_lag = X_lag.fillna(0.0)

    variable_lag = variable.copy()

    X_train, X_test, y_train, y_test = train_test_split(X_lag, variable_lag, test_size=60, shuffle=False)

    model = LinearRegression() 
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_train), index=y_train.index)
    y_fore = pd.Series(model.predict(X_test), index=y_test.index)

    train_r2 = r2_score(y_train, y_pred)
    test_r2 = r2_score(y_test, y_fore)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_fore))

    return X_train, X_test, y_train, y_test, y_pred, y_fore, train_r2, test_r2, train_rmse, test_rmse

def make_test_statistics(variable, time):

    mean = variable.ewm(time).mean()
    median = variable.rolling(time).median()
    std = variable.ewm(time).std()

    return mean, median, std

st.sidebar.title('Navigation')

options = st.sidebar.radio('Pages', options=['Home', 'Statistics', 'Seasonal & Fourier Forecast', 'Lag Model', 'Explore Different Data'])

df = pd.read_csv('Net_generation_United_States_all_sectors_monthly.csv', header=4)

df_renamed = cleaning(df)

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

    total = select_variable(df_renamed, 'Total')

    mean_3, median_3, std_3 = make_test_statistics(total, 3)

    mean_6, median_6, std_6 = make_test_statistics(total, 6)

    mean_12, median_12, std_12 = make_test_statistics(total, 12)
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

    total = select_variable(df_renamed, 'Total')

    total_pred, X_fore, total_fore, fourier_r2, fourier_rmse, fourier_mae = fourier_seasonal(total, 2)

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

    total = select_variable(df_renamed, 'Total')

    total_pred, X_fore, total_fore, fourier_r2, fourier_rmse, fourier_mae = fourier_seasonal(total, 2)

    deasonalized = deseasonalize(total, total_pred)

    final_pred_train, final_pred_test, train_r2_des, test_r2_des, train_rmse_des, test_rmse_des = make_deasonal_lag_model(total_pred, deasonalized, 12)

    X_train, X_test, y_train, y_test, y_pred, y_fore, train_r2, test_r2, train_rmse, test_rmse = make_lag_model(total, 12)

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
    st.pyplot(plot_lag_forecast_and_pred(total, y_pred, y_fore, y_test.index[0]))
    
    st.write("""
        The problem with this lag model though is that before I fitted the model to the data, I did not deseasonalize the initial data. The result is that the performance of my lag model, with and R^2 of 0.6983 for the test data, is much lower than my fourier model, which
        has an R^2 of .9092. Becuase I did not deseasonalize the data, the lags do not effectivley capture the non seasonal cycles since the predominant trend seen in the data is seasonal. To deseasonalize the data I subtracted the total predicted values of my fourier model from 
        the actual values. This left me with a series of the residuals from the seasonal prediction which I could now use in my Lag model. With the new series I went ahead and created a lag model like I did before, and the result was that
        I achieved a higher R^2 of 0.9436 for my test data. So overall, by deseasonalizing my data and applying a LAG model, I achieve a 4 percent increase in my R^2.
    """)

    st.pyplot(fig = plot_lag_forecast_and_pred(total, final_pred_train, final_pred_test, y_test.index[0]))

if options == 'Explore Different Data':

    st.header("Explore")

    st.subheader("Instructions")

    st.write("""
    The EIA dataset contains more categories than just total energy consumption. In this section I wanted to include a more interactive version of the models I demonstrated from previous setions
    So the point of this section is to be able to choose your own categoray and explore different lags, fourier orders, and test statistics to see how it effects the time series and its
    forecast. 
    """)

    category = st.selectbox(
        'Select Category',
        options=['Total', 'Coal', 'Natural Gas', 'Nuclear', 'Hydroelectric', 'Wind', 'Solar']
    )

    variable = select_variable(df_renamed, category)
    
    st.subheader(f"Analysis for: {category}")

    with st.expander("Periodogram Analysis"):
        st.pyplot(plot_periodogram_monthly(variable))
        st.caption("Use this to determine optimal Fourier order")

    st.subheader("Fourier Seasonal Model")
    
    fourier_order = st.number_input(
        'Fourier Order',
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help='Number of sine/cosine pairs to capture seasonality'
    )

    var_pred, X_fore, var_fore, f_r2, f_rmse, f_mae = fourier_seasonal(variable, fourier_order)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{f_r2:.4f}")
    with col2:
        st.metric("RMSE", f"{f_rmse:,.0f}")
    with col3:
        st.metric("MAE", f"{f_mae:,.0f}")
    
    st.pyplot(plot_forecast_and_pred(variable, var_pred, var_fore))

    st.subheader("Lag Model Analysis")
    
    lag_number = st.number_input(
        'Number of Lags',
        min_value=1,
        max_value=24,
        value=12,
        step=1,
        help='Number of past months to use as features'
    )

    tab1, tab2 = st.tabs(["Lag Plots", "PACF"])
    
    with tab1:
        st.pyplot(plot_lags(variable, lags=lag_number, nrows=4))
    
    with tab2:
        st.pyplot(plot_pacf(variable, lags=lag_number))

    st.write("**Simple Lag Model (without deseasonalization)**")
    X_train, X_test, y_train, y_test, y_pred, y_fore, train_r2, test_r2, train_rmse, test_rmse = make_lag_model(variable, lag_number)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Train R²", f"{train_r2:.4f}")
    with col2:
        st.metric("Test R²", f"{test_r2:.4f}")
    
    st.pyplot(plot_lag_forecast_and_pred(variable, y_pred, y_fore, y_test.index[0]))

    st.write("**Deseasonalized Lag Model**")
    deseasonalized = deseasonalize(variable, var_pred)
    final_pred_train, final_pred_test, train_r2_des, test_r2_des, train_rmse_des, test_rmse_des = make_deasonal_lag_model(var_pred, deseasonalized, lag_number)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Train R²", f"{train_r2_des:.4f}")
    with col2:
        st.metric("Test R²", f"{test_r2_des:.4f}")
    
    st.pyplot(plot_lag_forecast_and_pred(variable, final_pred_train, final_pred_test, y_test.index[0]))

    st.subheader("Rolling Statistics")
    
    test_stats = st.slider(
        'Rolling Window Size (months)',
        min_value=3,
        max_value=24,
        value=12,
        step=1
    )

    mean, median, std = make_test_statistics(variable, test_stats)
    st.pyplot(plot_rolling_stats(mean, median, std, test_stats, emw=True))  







