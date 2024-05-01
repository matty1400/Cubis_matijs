import pmdarima as pm
import pandas as pd
import warnings
from math import sqrt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

df = pd.read_csv('baggagecomplaints.csv')

df = df[df['Airline']=='American Eagle']
df = df[['Date','Baggage']]

# Convert the 'Date' column to datetime and set it as the index
df['Date'] = pd.to_datetime(df['Date'])


# Explicitly specify the dtype of the index

df_train = df[df['Date']<'2010']

df_test = df[df['Date'] >= '2010']
data_actual = df_train
df_train.set_index('Date', inplace=True)
df_test.set_index('Date', inplace=True)

# Convert the 'Date' column to datetime and set it as the index
df['Date'] = pd.to_datetime(df['Date'])


# Explicitly specify the dtype of the index

df_train = df[df['Date']<'2010']

df_test = df[df['Date'] >= '2010']
data_actual = df_train
df_train.set_index('Date', inplace=True)
df_test.set_index('Date', inplace=True)

seasonal = True

model = pm.auto_arima(df_train,
                      m= 12,
                      seasonal= False,
                      stepwise= True,
                      
                       max_q= 6 )

print(model.summary())

fc, confint = model.predict(n_periods=12, return_conf_int=True)
data_fc = []
data_lower = []
data_upper = []
data_aic = []
data_fitted = []

data_fc.append(fc)
data_lower.append(confint[:, 0])
data_upper.append(confint[:, 1])
data_aic.append(model.aic())
data_fitted.append(model.fittedvalues())

data_forecast = pd.DataFrame(data_fc)


data_forecast = pd.melt(data_forecast, var_name='Date', value_name='Value')
data_actual['desc'] = 'Actual'
data_forecast['desc'] = 'Forecast'
data_forecast.rename(columns={'Value': 'Baggage'},inplace=True)

data_actual.reset_index(inplace=True)

# Combine actual and forecast data into a single DataFrame and reset the index
df_act_fc = pd.concat([data_actual, data_forecast])

df_act_fc.to_csv('output_no_seasonality.csv')
