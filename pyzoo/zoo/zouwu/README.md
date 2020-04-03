## Project Zouwu: Time Series Solution for Telco on Analytics Zoo

Project Zouwu provides a reference solution that is designed and optimized for common time series applications in the Telco industry, including:
* _Use case_ - reference time series use cases in the Telco industry (such as network traffic forcasting, etc.)
* _Model_ - built-in deep learning models for time series analysis (such as LSTM and [MTNet](https://arxiv.org/abs/1809.02105))
* _AutoTS_ - AutoML support for building end-to-end time series analysis pipelines (including automatic feature generation, model selection and hyperparameter tuning).

---
### Requirements
* Python 3.6 or 3.7
* PySpark 2.4.3
* Ray 0.7.0
* Tensorflow 1.15.0
* aiohttp
* setproctitle
* scikit-learn
* featuretools
* pandas
* Note that Keras is not needed to use Zouwu. But if you have Keras installed, make sure it is Keras 1.2.2. Other verisons might cause unexpected problems. 

### Install 
  * Download the Analytics Zoo source code (master). 
  * Build a local ```.whl``` for Spark2.4.3 using command below. For detailed explaination of the options for ```bulid.sh``` script, refer to [AnalytisZoo Python Developer's guide](https://analytics-zoo.github.io/master/#DeveloperGuide/python/#build-whl-package-for-pip-install)
```bash
bash analytics-zoo/pyzoo/dev/build.sh linux default -Dspark.version=2.4.3 -Dbigdl.artifactId=bigdl-SPARK_2.4 -P spark_2.4+
```
  * The succesfully built ```.whl``` can be found in directory ```analytics-zoo/pyzoo/dist/```. Install the ```.whl``` locally using below command (VERSION and PLATFORM may vary according to platform and building options). 
```
pip install analytics-zoo/pyzoo/dist/analytics_zoo-VERSION-py2.py3-none-PLATFORM_x86_64.whl
```

### Reference Use Case

Time series forecasting has many applications in telco. Accurate forecast of telco KPIs (e.g. traffic, utilizations, user experience, etc.) for communication networks ( 2G/3G/4G/5G/wired) can help predict network failures, allocate resource, or save energy. Time series forecasting can also be used for log and metric analysis for data center IT operations for telco. Metrics to be analyzed can be hardware or VM utilizations, database metrics or servce quality indicators. 

We provide a [notebook](https://github.com/shane-huang/analytics-zoo/blob/zouwu-readme-nb/pyzoo/zoo/zouwu/use-case/network_traffic/time_series_forecasting_network_traffic.ipynb) to demonstrate a time series forecasting use case using a public telco dataset, i.e. the aggregated network traffic traces at the transit link of WIDE to the upstream ISP ([dataset link](http://mawi.wide.ad.jp/~agurim/dataset/)). In this reference case, we used aggregated traffic metrics (e.g. total bytes, average MBps) in the past to forecast the traffic in the furture. We demostrate how to do univariant forecasting (predict only 1 series), and multivariant forecasting (predicts more than 1 series at the same time) using Project Zouwu.

### Usage

#### Using built-in forecast models

The built-in forecast models are all derived from [tfpark.KerasModels](https://analytics-zoo.github.io/master/#APIGuide/TFPark/model/). 

1. To start, you need to create a forecast model first. Specify ```horizon``` and ```feature_dim``` in constructor. 
    * ```horizon```: no. of steps to forecast
    * ```feature_dim```: dimension of input feature

Refer to API doc for detailed explaination of all arguments for each forecast model.
Below are some example code to create forecast models.
```python
#import forecast models
from zoo.zouwu.model.forecast import LSTMForecaster
from zoo.zouwu.model.forecast import MTNetForecaster

#build a lstm forecast model
lstm_forecaster = LSTMForecaster(horizon=1, 
                      feature_dim=4)
                      
#build a mtnet forecast model
mtnet_forecaster = MTNetForecaster(horizon=1,
                        feature_dim=4,
                        lb_long_steps=1,
                        lb_long_stepsize=3
```
 
2. Use ```forecaster.fit/evalute/predict``` in the same way as [tfpark.KerasModel](https://analytics-zoo.github.io/master/#APIGuide/TFPark/model/)

3. For univariant forecasting (i.e. to predict one series at a time), you can use either ```LSTMForecaster``` or ```MTNetForecaster```. The input data shape for ```fit/evaluation/predict``` should match the arguments you used to create the forecaster. Specifically:
   * X shape should be ```(num of samples, lookback, feature_dim)```
   * Y shape should be ```(num of samples, horizon)```
   * ```feature_dim``` is the number of features as specified in Forecaster constructors.
   * ```horizon``` is the number of steps to look forward as specified in Forecaster constructors.
   * ```lookback``` is the number of time steps you want to look back in history. 

4. For multivariant forecasting (i.e. to predict several series at the same time), you have to use ```MTNetForecaster```. The input data shape should meet below criteria. Note for multivariant forecasting, horizon must be 1. 
   * X shape should be ```(num of samples, lookback, feature_dim)```
   *  Y shape should be ```(num of samples, num_of_targets)``` 
   * ```lookback``` should equal ```(lb_long_steps+1) * lb_long_stepsize```, where ```lb_long_steps``` and ```lb_long_stepsize``` are as specified in ```MTNetForecaster``` constructor. 
   * ```num_targets``` is the number of series to forecast at the same time
       

#### Using AutoTS

The automated training in zouwu is built upon [Analytics Zoo AutoML module](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/automl) (refer to [AutoML ProgrammingGuide](https://analytics-zoo.github.io/master/#ProgrammingGuide/AutoML/overview/) and [AutoML APIGuide](https://analytics-zoo.github.io/master/#APIGuide/AutoML/time-sequence-predictor/) for details), which uses [Ray Tune](https://github.com/ray-project/ray/tree/master/python/ray/tune) for hyper parameter tuning and runs on [Analytics Zoo RayOnSpark](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/).  

The general workflow using automated training contains below two steps. 
   1. create a ```AutoTSTrainer``` to train a ```TSPipeline```, save it to file to use later or elsewhere if you wish.
   2. use ```TSPipeline``` to do prediction, evaluation, and incremental fitting as well. 

You'll need ```RayOnSpark``` for training with ```AutoTSTrainer```, so you have to init it before auto training, and stop it after training is completed. Note RayOnSpark is not needed if you just use TSPipeline for inference, evaluation or incremental training. 

   * init RayOnSpark in local mode
```python
from zoo import init_spark_on_local
from zoo.ray.util.raycontext import RayContext
sc = init_spark_on_local(cores=4)
ray_ctx = RayContext(sc=sc)
ray_ctx.init()
```
   * init RayOnSpark on yarn
   ```python
   from zoo import init_spark_on_yarn
from zoo.ray.util.raycontext import RayContext
slave_num = 2
sc = init_spark_on_yarn(
        hadoop_conf=args.hadoop_conf,
        conda_name="ray36",
        num_executor=slave_num,
        executor_cores=4,
        executor_memory="8g ",
        driver_memory="2g",
        driver_cores=4,
        extra_executor_memory_for_ray="10g")
ray_ctx = RayContext(sc=sc, object_store_memory="5g")
ray_ctx.init()
   ```
   * After training, stop RayOnSpark. 
   ```python
   ray_ctx.stop()
   ```

AutoTSTrainer and TSPipeline accepts data frames as input. An exmaple data frame looks like below. 

  |datetime|value|extra_feature_1|extra_feature_2|
  | --------|----- |---| ---|
  |2019-06-06|1.2|1|2|
  |2019-06-07|2.3|0|2|
 

1. To create an AutoTSTrainer. Specify below arguments in constructor. See below example.
    * ```dt_col```: the column specifying datetime 
    * ```target_col```: target column to predict
    * ```horizon``` : num of steps to look forward 
    * ```extra_feature_col```: a list of columns which are also included in input as features except target column
 ```python
 from zoo.zouwu.autots.forecast import AutoTSTrainer

 trainer = AutoTSTrainer(dt_col="datetime",
                         target_col="value"
                         horizon=1,
                         extra_features_col=None)

 ```
 
2. Use ```AutoTSTrainer.fit``` on train data and validation data. A TSPipeline will be returned. 
 ```python
 ts_pipeline = trainer.fit(train_df, validation_df)
 ```

3. Use ```TSPipeline.fit/evaluate/predict``` to train pipeline (incremental fitting), evaluate or predict. 
 ```python
 #incremental fitting
 ts_pipeline.fit(new_train_df, new_val_df, epochs=10)
 #evaluate
 ts_pipeline.evalute(val_df)
 ts_pipeline.predict(test_df)
 
 ```
4. Use ```TSPipeline.save/load``` to load from file or save to file. 
 ```python
 from zoo.zouwu.autots.forecast import TSPipeline
 loaded_ppl = TSPipeline.load(file)
 # ... do sth. e.g. incremental fitting
 loaded_ppl.save(another_file)
 ```
### built-in models vs. AutoTS

Now we show some comparison results between manually tuned models vs. AutoTS, using network traffic forecasting as an example.  

We first trained an LSTMForecaster with 5 epochs (*Experiment 1*). Then we use AutoTS to train the model and find out the best configuration out of 120 trials (*Experiment 2*). The final best pipeline was acutally traied 50 epochs before it stops. To make the comparison fairer, we also train 50 epochs for LSTMForecaster with the same hyper parameters (*Experiment 3*). The time and accuracy results are as shown in below table. AutoTS achieves the best accuracy, while the time is still acceptable (on single node with 2 trials runing in parallel at the same time)  

|Experiment No.|Model|Mean Squared Error (smaller the better)|Symmetric Mean Absolute Percentage Error (smaller the better)|Epochs|Training Time|
|-|--|-----|----|---|----|
|1|Manually Tuned (LSTMForecaster)|27277.36|18.22%|5|8.1s|
|2|AutoTS (LSTMForecaster)|2792.22|5.80%|50|40mins (120 trails on single node w/ 2 parallel workers)|
|3|Manually Tuned (LSTMForecaster)|6312.44|8.61%|50|1min 9s|


Below is a comparison between manually selected parameters and auto-tuned parameters form AutoTS. We can see the features selected by AutoTS make much sense in our case.  

||features|Batch size|learning rate|lstm_units*|dropout_p*|Lookback|
|--|--|--|-----|-----|-----|-----|
|LSTMForecaster|year, month, week, day_of_week, hour|1024|0.001|32, 32|0.2, 0.2|55|
|AutoTS|hour, is_weekday, is_awake|64|0.001|32, 64|0.2, 0.236|84|

_*_: There're 2 lstm layers and dropout in LSTM model, the hyper parameters in the table corresponds to the 1st and 2nd layer respectively. 
