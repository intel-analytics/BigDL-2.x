# Zouwu User Guide

### **1. Overview**
Project Zouwu provides a toolkit and reference solution that is designed and optimized for common time series applications in the Telco industry.

There're 2 ways you can use Zouwu for time series analysis.

- use AutoML enabled integrated pipeline API to build end-to-end time series analysis pipelines (i.e. [AutoTS]())
- use standalone deep learning models and/or other pipeline modules for time series analysis without AutoML(e.g. [LSTMForecaster](), [MTNetForecaster]() and [TCMFForecaster]())

Zouwu also provides reference time series use cases solutions in the Telco industry.

- **Time series forecasting** Accurate forecast of telco KPIs (e.g. traffic, utilizations, user experience, etc.) for communication networks ( 2G/3G/4G/5G/wired) can help predict network failures, allocate resource, or save energy. And time series forecasting can also be used for log and metric analysis for data center IT operations for telco. Metrics to be analyzed can be hardware or VM utilizations, database metrics or service quality indicators. We provided a reference use case where we forecast network traffic KPI's as a demo. Refer to [Network Traffic Notebook]().

- **Anomaly detection** Detecting anomaly is also very common in telco. One way of doing anomaly detection is first do forecasting, and if the actual value diverges too much from the predicted value, it would be considered anomaly. We provided such a reference use case as a demo. Refer to [Anomaly Detection Notebook]()

#### **2 Install Dependencies**

Zouwu depends on below python libraries.

```bash
python 3.6 or 3.7
pySpark
analytics-zoo
tensorflow>=1.15.0,<2.0.0
h5py==2.10.0
ray[tune]==0.8.4
psutil
aiohttp
setproctitle
pandas
scikit-learn>=0.20.0,<0.24.0
requests
```

You can always install the dependencies manually, but it is highly recommended that you use Anaconda to prepare the environments, especially if you want to run automated training on a yarn cluster (yarn-client mode only). Analytics-zoo comes with a pre-defined dependency list, you can easily use below command to install all the dependencies for zouwu.

```bash
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo[automl]==0.9.0.dev0 # or above
```

---
### **3. Use AutoML-enabled API**

You can use the **```AutoTS```** package to train a time series model with AutoML.

The general workflow has two steps:

* create a [AutoTSTrainer]() and train. It will return a [TSPipeline](). You can save it to file to use later or elsewhere.
* use [TSPipeline]() to do prediction, evaluation, and incremental fitting.

Refer to [AutoTS notebook demo]() for demonstration how to use AutoTS to build a time series forecasting pipeline.


#### **3.1 Initialize Orca Context**

AutoTS training (i.e. ```AutoTSTrainer.fit```) relies on [RayOnSpark]() to run, so you need to initalize [OrcaContext](https://testshanedoc.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) with argument ```init_ray_on_spark=True``` before the training, and stop it after training is completed. Refer to [OrcaContext User Guide]() for details about how to initialize it and stop it.

Note: [OrcaContext](https://testshanedoc.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) is not needed if you just use the trained [TSPipeline]() for inference, evaluation or incremental training.


* local mode

```python
from zoo.orca import init_orca_context, stop_orca_context
init_orca_context(cluster_mode="local", cores=4, memory='2g', num_nodes=1, init_ray_on_spark=True)
```

* yarn client mode

```python
from zoo.orca import init_orca_context, stop_orca_context
init_orca_context(cluster_mode="yarn-client",
                  num_nodes=2, cores=2,
                  driver_memory="6g", driver_cores=4,
                  conda_name='zoo',
                  extra_memory_for_ray="10g",
                  object_store_memory='5g')
```

#### **3.2 Prepare Your data**

You should prepare a training dataset, and/or a validation dataset. If you have one time series data and you want both a training dataset and a validation dataset, you can break it into two segments in time line. Usually the validation dataset should come later in time than training dataset. We have provided a simple utility function to help you split your data into train/validation/test set (refer to [code](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/automl/common/util.py#L28))

Both the training data and validation data must be provided in form of a **pandas dataframe**. And if validation data is provided, it should have the same column names as the training data. The dataframes should have at least two columns
- the target column, containing all the historical data points which you want use to predict the future data points
- the datetime column, which contains the timestamps corresponding to each data point in the target column. The datatime column values should have pandas datetime format (You can use pandas.to_datetime to convert a string into a datetime format)
You may have other input columns for each data point which you might what to use as extra feature, but you don't need to predict them. So the final input data could look something like below.

```bash
datetime    target  extra_feature_1  extra_feature_2
2019-06-06  1.2    1                2
2019-06-07  2.30   2                1
```

#### **3.3 Create an AutoTSTrainer**

Create an AutoTSTrainer. In constructor, specify ```target_col``` as the name of the target column in the training data, and specify ```dt_col``` as the name of the datetime column in the training data. If you have extra features, specify those in "extra_features_col".

```python
from zoo.zouwu.autots.forecast import AutoTSTrainer

trainer = AutoTSTrainer(dt_col="datetime",
                        target_col="target",
                        horizon=1,
                        extra_features_col=["extra_feature_1","extra_feature_2"])
```

Refer to [AutoTSTrainer API Spec]() for more details.

#### **3.4 Train a pipeline**

Use ```AutoTSTrainer.fit``` on train on input data and/or validation data with AutoML. A [TSPipeline]() will be returned. A TSPipeline include not only the model, but also the data preprocessing/post processing steps. Hyperparameters are automatically chosen for one or more of the steps in the entire pipeline during the fit process.

```python
ts_pipeline = trainer.fit(train_df, validation_df)
```
You can use built-in [visualization tool]() to inspect the training results after training stopped.


#### **3.5 Use a pipeline**

Use ```TSPipeline.fit|evaluate|predict``` to train pipeline (incremental fitting), evaluate or predict.
Incremental fitting on TSPipeline just update the model weights the standard way, which does not involve AutoML.
```python
#incremental fitting
ts_pipeline.fit(new_train_df, new_val_df, epochs=10)
#evaluate
ts_pipeline.evalute(val_df)
ts_pipeline.predict(test_df)
```
Use ```TSPipeline.save|load``` to load from file or save to file.
```python
from zoo.zouwu.autots.forecast import TSPipeline
loaded_ppl = TSPipeline.load(file)
# ... do sth. e.g. incremental fitting
loaded_ppl.save(another_file)
```

---
### **4. Use Standalone Pipeline API**

Zouwu provides below standalone built-in deep learning time series models. They are just normal models and don't have AutoML support.

* [LSTMForecaster]()
* [MTNetForecaster]()
* [TCMFForecaster]()
* [TCNForecaster]()

Besides, zouwu also provides some pipeline modules such as data pre-processsing and post-processing.

* [Data Imputation]()
* More to go ...

#### **4.1 Initialize Orca Context**

Our built-in models support distributed training, which relies on [Orca](), so you need to initalize [OrcaContext](https://testshanedoc.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) before training and stop it after training is completed. Refer to [OrcaContext User Guide]() for details about how to initialize it and stop it.

Note that [TCMFForecaster]() needs [RayOnSpark] for distributed training, you need to initilize [OrcaContext](https://testshanedoc.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) with argument ```init_ray_on_spark=True```.


* local mode

```python
from zoo.orca import init_orca_context, stop_orca_context
init_orca_context(cluster_mode="local", cores=4, memory='2g', num_nodes=1, init_ray_on_spark=True)
```

* yarn client mode

```python
from zoo.orca import init_orca_context, stop_orca_context
init_orca_context(cluster_mode="yarn-client",
                  num_nodes=2, cores=2,
                  driver_memory="6g", driver_cores=4,
                  conda_name='zoo',
                  extra_memory_for_ray="10g",
                  object_store_memory='5g')
```
#### **4.2 LSTMForecaster**

LSTMForecaster wraps a vanilla LSTM model. It is relatively simple and light-weight.

LSTMForecaster is derived from [tfpark.KerasModels]().

Refer to [network traffic notebook]() for a real-world example and [LSTMForecaster API]() for more details.

##### **4.2.1 Prepare your data**

Currently LSTMForecaster only supports univariate forecasting (i.e. to predict one series at a time). The input data can be numpy arrays or TFDataset. For more details on how to prepare the data, refer to [tfpark.KerasModels]().

You should prepare two dataset X and Y. The dimensions of X and Y should be as follows:

* X shape should be (num of samples, lookback, feature_dim)
* Y shape should be (num of samples, target_dim)
Where, feature_dim is the number of features as specified in Forecaster constructors. lookback is the number of time steps you want to look back in history. target_dim is the number of series to forecast at the same time as specified in Forecaster constructors and should be 1 here. If you want to do multi-step forecasting and use the second dimension as no. of steps to look forward, you won't get error but the performance may be uncertain and we don't recommend using that way.


##### **4.2.2 Create a LSTMForecaster**

When creating the forecaster, you should specify the arguments that matches dimensions of your input data, i.e. target_dim, feature_dim as specified in section 4.2.1.

```python
from zoo.zouwu.model.forecast.lstm_forecaster import LSTMForecaster
lstm_forecaster = LSTMForecaster(target_dim=1,
                      feature_dim=4)
```
##### **4.2.3 Use a LSTMForecaster**

You can use LSTMForecaster to do fit, evaluation, and prediction. These APIs are derived from tfpark.KerasModel, refer to [tfpark.KerasModel API]() for details.

```python
lstm_forecaster.fit(X,Y)
lstm_forecaster.predict(X)
lstm_forecaster.evaluate(X,Y)
```

#### **4.3 MTNetForecaster**

MTNetForecaster wraps a MTNet model. The model architecture mostly follows the [MTNet paper](https://arxiv.org/abs/1809.02105) with slight modifications.
MTNetForecaster is derived from [tfpark.KerasModels]().

Refer to [network traffic notebook]() for a real-world example and [MTNetForecaster API]() for more details.

##### **4.3.1 Prepare your data**

The input data can be numpy arrays or TFDataset. For more details on how to prepare the data, refer to [tfpark.KerasModels]().

* For univariate forecasting (i.e. to predict one series at a time), the input data shape for fit/evaluation/predict should match the arguments you used to create the forecaster. Specifically:

  - X shape should be (num of samples, lookback, feature_dim)
  - Y shape should be (num of samples, target_dim)
Where, feature_dim is the number of features as specified in Forecaster constructors. lookback is the number of time steps you want to look back in history. target_dim is the number of series to forecast at the same time as specified in Forecaster constructors and should be 1 here. If you want to do multi-step forecasting and use the second dimension as no. of steps to look forward, you won't get error but the performance may be uncertain and we don't recommend using that way.

* For multivariate forecasting (i.e. to predict several series at the same time), the input data shape should meet below criteria.

  - X shape should be (num of samples, lookback, feature_dim)
  - Y shape should be (num of samples, target_dim)
Where lookback should equal (lb_long_steps+1) * lb_long_stepsize, where lb_long_steps and lb_long_stepsize are as specified in MTNetForecaster constructor. target_dim should equal number of series in input.

##### **4.3.2 Create a MTNetForecaster**

When creating the forecaster, you should specify the arguments that matches dimensions of your input data, i.e. target_dim, feature_dim as specified in section 4.3.1.
```python
from zoo.zouwu.model.forecast.mtnet_forecaster import MTNetForecaster
mtnet_forecaster = MTNetForecaster(target_dim=1,
                        feature_dim=4,
                        long_series_num=1,
                        series_length=3,
                        ar_window_size=2,
                        cnn_height=2)
```
##### **4.3.3 Use a MTNetForecaster**

You can use MTNetForecaster to do fit, evaluation, and prediction. These APIs are derived from tfpark.KerasModel, refer to [tfpark.KerasModel API]() for details.

```python
mtnet_forecaster.fit(X,Y)
mtnet_forecaster.predict(X)
mtnet_forecaster.evaluate(X,Y)
```

#### **4.4 TCMFForecaster**

TCMFForecaster wraps a model architecture that follows implementation of the paper [DeepGLO paper](https://arxiv.org/abs/1905.03806) with slight modifications. It is especially suitable for extremely high dimensional multivariate time series forecasting.

##### **4.4.1 Prepare Your data**

You can either train TCMF model locally or distributedly in a cluster. TCMFForecaster accepts a python dictionary as input, and the content of the dictionary varies according to different configurations. Refer to [TCMFForecaster API Spec]() for more details.

* When training locally

First you need to prepare your data as a numpy array. Your data should have two dimensions, one is the sequential data points in time, the other is the N number of time series.  For example, suppose you have 3 time series (N=3), each time series has 5 sequential data points in time. Your data should have shape (3,5) and look something like below.
```python
>>>data
array([[1, 2, 3, 4, 5]
  [10, 20, 30, 40, 50]
  [100, 200, 300, 400, 500]])
```
The simplest way to prepare the input for TCFMForecaster is just to specify the above numpy data as y in the dict, i.e. ```{'y': data}```. You can also attach an id for each of the time series, i.e. ```{'id': id, 'y': data}```. id is also a numpy array in shape of (N, 1). In the above example, it could be something like
```
>>>id
array['ts-1','ts-10','ts-100']
```

* When training distributedly in a cluster

You need to prepare the data as an XShards for distributed training. For how to create an XShards, refer to [XShards User Guide]().

After preparation, each partition in the XShards should contain a dictionary of {'id': record id, 'y': sequential data}, where id and y are in the same shape as specified in the local mode.


##### **4.4.2 Create a TCMFForecaster**
```python
from zoo.zouwu.model.forecast.tcmf_forecaster import TCMFForecaster
model = TCMFForecaster(
        vbsize=128,
        hbsize=256,
        num_channels_X=[32, 32, 32, 32, 32, 1],
        num_channels_Y=[16, 16, 16, 16, 16, 1],
        kernel_size=7,
        dropout=0.1,
        rank=64,
        kernel_size_Y=7,
        learning_rate=0.0005,
        normalize=False,
        use_time=True,
        svd=True,)
```
##### **4.4.3 Use a TCMFForecaster**

* fit
```python
model.fit(
        x,
        val_len=24,
        start_date="2020-4-1",
        freq="1H",
        covariates=None,
        dti=None,
        period=24,
        y_iters=10,
        init_FX_epoch=100,
        max_FX_epoch=300,
        max_TCN_epoch=300,
        alt_iters=10,
        num_workers=num_workers_for_fit)
```
 * evaluate

You can either call evalute directly

```python
model.evaluate(target_value,
               metric=['mae'],
               target_covariates=None,
               target_dti=None,
               num_workers=num_workers_for_predict,
               )

```
Or predict first and then evaluate with metric name.
```python
yhat = model.predict(horizon,
                     future_covariates=None,
                     future_dti=None,
                     num_workers=num_workers_for_predict)

from zoo.automl.common.metrics import Evaluator
evaluate_mse = Evaluator.evaluate("mse", target_data, yhat)
```
 * incremental fit
```python
model.fit_incremental(x_incr, covariates_incr=None, dti_incr=None)
```

 * save and load
```python
model.save(dirname)
loaded_model = TCMFForecaster.load(dirname)
```
