# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/chronos/chronos_autots_nyc_taxi.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 
# ![image.png](data:image/png;base64,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)
# ---

# ##### Copyright 2018 Analytics Zoo Authors.

# In[ ]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# ## **Environment Preparation**

# **Install Analytics Zoo**
# 
# You can install the latest pre-release version with chronos support using `pip install --pre --upgrade analytics-zoo[automl]`. 

# In[ ]:


# Install latest pre-release version of Analytics Zoo 
# Installing Analytics Zoo from pip will automatically install pyspark, bigdl, and their dependencies.
#.system('pip install --pre --upgrade analytics-zoo[automl]')
#exit() # restart the runtime to refresh installed pkg


# ### **Step 0: Download & prepare dataset**
# We used NYC taxi passengers dataset in [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) for demo, which contains 10320 records, each indicating the total number of taxi passengers in NYC at a corresonponding time spot.

# In[ ]:


# download the dataset
#.system('wget https://raw.githubusercontent.com/numenta/NAB/v1.0/data/realKnownCause/nyc_taxi.csv')


# In[ ]:


# load the dataset. The downloaded dataframe contains two columns, "timestamp" and "value".
import pandas as pd
df = pd.read_csv("nyc_taxi.csv", parse_dates=["timestamp"])


# ## **Time series forecasting using Chronos Forecaster**
# ### Forecaster Step1. Data transformation and feature engineering using Chronos TSDataset 
# [TSDataset](https://analytics-zoo.readthedocs.io/en/latest/doc/PythonAPI/Chronos/tsdataset.html) is our abstract of time series dataset for data transformation and feature engineering. Here we use it to preprocess the data.

# In[ ]:


from zoo.chronos.data import TSDataset
from sklearn.preprocessing import StandardScaler


# Initialize train, valid and test tsdataset from raw pandas dataframe.

# In[ ]:


tsdata_train, tsdata_valid, tsdata_test = TSDataset.from_pandas(df, dt_col="timestamp", target_col="value",
                                                                with_split=True, val_ratio=0.1, test_ratio=0.1)


# Preprocess the datasets. Here we perform:
# - deduplicate: remove those identical data records
# - impute: fill the missing values
# - gen_dt_feature: generate feature from datetime (e.g. month, day...)
# - scale: scale each feature to standard distribution.
# - roll: sample the data with sliding window.
# 
# For forecasting task, we will look back 3 hours' historical data (6 records) and predict the value of next 30 miniutes (1 records).
# 
# We perform the same transformation processes on train, valid and test set.

# In[ ]:


lookback, horizon = 6, 1

scaler = StandardScaler()
for tsdata in [tsdata_train, tsdata_valid, tsdata_test]:
    tsdata.deduplicate()          .impute()          .gen_dt_feature()          .scale(scaler, fit=(tsdata is tsdata_train))          .roll(lookback=lookback, horizon=horizon)


# ### Forecaster Step 2: Time series forecasting using Chronos Forecaster
# After preprocessing the datasets. We can use [Chronos Forecaster](https://analytics-zoo.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html) to handle the forecasting tasks.

# In[ ]:


# set random seeds to help reproduce the same result
import torch, random, numpy
from zoo.chronos.model.forecast.tcn_forecaster import TCNForecaster
torch.manual_seed(0)
numpy.random.seed(0)
random.seed(0)


# Transform TSDataset to sampled numpy ndarray and feed them to forecaster.

# In[ ]:


x, y = tsdata_train.to_numpy() 
x_val, y_val = tsdata_valid.to_numpy() 
# x.shape = (num of sample, lookback, num of input feature)
# y.shape = (num of sample, horizon, num of output feature)

forecaster = TCNForecaster(past_seq_len=lookback,  # number of steps to look back
                           future_seq_len=horizon,  # number of steps to predict
                           input_feature_num=x.shape[-1],  # number of feature to use
                           output_feature_num=y.shape[-1])  # number of feature to predict
res = forecaster.fit(x, y, validation_data=(x_val, y_val), epochs=3)


# ### Forecaster Step 3: Further deployment with fitted forecaster
# Use fitted forecaster to predict test data and plot the result

# In[ ]:


x_test, y_test = tsdata_test.to_numpy()
pred = forecaster.predict(x_test)
pred_unscale, groundtruth_unscale = tsdata_test.unscale_numpy(pred), tsdata_test.unscale_numpy(y_test)


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(24,6))
plt.plot(pred_unscale[:,:,0])
plt.plot(groundtruth_unscale[:,:,0])
plt.legend(["prediction", "ground truth"])


# Save & restore the forecaster.

# In[ ]:


forecaster.save("nyc_taxi.fxt")
forecaster.restore("nyc_taxi.fxt")

