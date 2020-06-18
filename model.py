'''
will forecast new cases

new england sucks
'''

import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot


data = pd.read_csv('input_data/data.csv')
data['floor'] = 0
periods = 15
actual_df = data[['y']]
actual_df = actual_df.rename({'y': 'obs'}, axis='columns')


m = Prophet(mcmc_samples=300,interval_width = .95)
m.fit(data)
future = m.make_future_dataframe(periods=periods)
forecast = m.predict(future)
forecast_df = forecast
forecast_df = forecast_df.join(actual_df)
#save forecast data
forecast_df.to_csv('output_data/csv/forecast.csv')

#save plots to png folder
m.plot_components(forecast).savefig('output_data/png/components.png')
m.plot(forecast).savefig('output_data/png/forecast.png')
