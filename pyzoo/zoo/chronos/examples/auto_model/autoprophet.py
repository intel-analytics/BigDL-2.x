import numpy as np
import pandas as pd

from zoo.chronos.forecast.prophet_forecaster import ProphetForecaster
from zoo.chronos.autots.model.auto_prophet import AutoProphet
from zoo.orca.common import init_orca_context, stop_orca_context

# data prepare
df = pd.read_csv("/home/junweid/nyc_taxi.csv", parse_dates=[0])
df=df.rename(columns={'timestamp':'ds','value':'y'})

# train/test split
end_date = '2015-1-28'
df_train = df[df['ds'] <= end_date]
df_test = df[df['ds'] > end_date]

# use prophet forecaster
prophet = ProphetForecaster()
prophet.fit(df_train, validation_data=df_test)

# use autoprophet for HPO
init_orca_context(cores=20, init_ray_on_spark=True)
autoprophet = AutoProphet()
autoprophet.fit(df_train, n_sampling=20)
stop_orca_context()

# evaluate
auto_searched_mse = autoprophet.evaluate(df_test, metrics=['mse'])[0]
nonauto_searched_mse = prophet.evaluate(df_test, metrics=['mse'])[0]
print("Autoprophet improve the mse by",
      str(((nonauto_searched_mse - auto_searched_mse)/nonauto_searched_mse)*100) ,'%')
print("auto_searched_mse:", auto_searched_mse)
print("nonauto_searched_mse:", nonauto_searched_mse)
