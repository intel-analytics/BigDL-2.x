# Project Zouwu 
Time Series Solution for Telco on Analytics Zoo


## Requirements
* Python 3.6 or 3.7
* PySpark 2.4.3
* Ray 0.7.0
* Keras 1.2.2
* Tensorflow 1.15.0

## Install 
  * Download the Analytics Zoo source code (master). 
  * Build a local ```.whl``` for Spark2.4.3 using command below. For detailed explaination of the options for ```bulid.sh``` script, refer to [AnalytisZoo Python Developer's guide](https://analytics-zoo.github.io/master/#DeveloperGuide/python/#build-whl-package-for-pip-install)
```bash
bash analytics-zoo/pyzoo/dev/build.sh linux default -Dspark.version=2.4.3 -Dbigdl.artifactId=bigdl-SPARK_2.4 -P spark_2.4+
```
  * The succesfully built ```.whl``` can be found in directory ```analytics-zoo/pyzoo/dist/```. Install the ```.whl``` locally using below command (VERSION and PLATFORM may vary according to platform and building options). 
```
pip install analytics-zoo/pyzoo/dist/analytics_zoo-VERSION-py2.py3-none-PLATFORM_x86_64.whl
```

## Reference Use Case

Time series forecasting has many applications in telco. Accurate forecast of telco KPIs (e.g. traffic, utilizations, user experience, etc.) for communication networks ( 2G/3G/4G/5G/wired) can help predict network failures, allocate resource, or save energy. Time series forecasting can also be used for log and metric analysis for data center IT operations for telco. Metrics to be analyzed can be hardware or VM utilizations, database metrics or servce quality indicators. 

We provide a [notebook](https://github.com/shane-huang/analytics-zoo/blob/zouwu-readme-nb/pyzoo/zoo/zouwu/use-case/network_traffic/time_series_forecasting_network_traffic.ipynb) to demonstrate a time series forecasting use case using a public telco dataset, i.e. the aggregated network traffic traces at the transit link of WIDE to the upstream ISP ([dataset link](http://mawi.wide.ad.jp/~agurim/dataset/)). In this reference case, we used aggregated traffic metrics (e.g. total bytes, average MBps) in the past to forecast the traffic in the furture. We demostrate how to do univariant forecasting (predict only 1 series), and multivariant forecasting (predicts more than 1 series at the same time) using zouwu.

## Usage

### Train forecast models and forecast (without AutoML)

The forecast models are all derived from [tfpark.KerasModels](https://analytics-zoo.github.io/master/#APIGuide/TFPark/model/). 

1. To start, you need to create a forecast model first. Specify ```horizon``` and ```feature_dim``` in constructor. 
    * horizon: steps to look forward
    * feature_dim: dimension of input feature

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

3. For univariant forecasting, the input data shape for ```fit/evaluation/predict``` should match the arguments you used to create the forecaster. Specifically:
   * X shape should be ```(num of samples, lookback, feature_dim)```, for train, validation and test data
   * Y shape should be ```(num of samples, horizon)```, for train and validation data
   *  ```feature_dim``` and ```horizon``` as specified in constructor, ```lookback``` is the number of time steps you want to look back in history.

4. For multivariant forecasting (using ```MTNetForecaster```) where number of targets to predict (i.e. num_targets) >= 1, the input data shape should meet below criteria. 
   * X shape should be ```(num of samples, lookback, feature_dim)```, for train, validation and test data
       * Note that for MTNetForecaster, ```lookback``` = ```(lb_long_steps+1) * lb_long_stepsize```, lb_long_steps and lb_long_stepsize as specified in MTNetForecaster constructor. 
   * Y shape should be either one of below
       * ```(num of samples, num_of_targets)``` if horizon = 1
       * ```(num of samples, num_of_targets, horizon)``` if horizon > 1 && num_targets > 1
       * ```(num of samples, horizon)``` if num_targets = 1 (fallback to univariant forecasting)

### Use automated training to train a forecast pipeline and forecast

The automated training in zouwu is built upon [Analytics Zoo AutoML module](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/automl), which uses [Ray Tune](https://github.com/ray-project/ray/tree/master/python/ray/tune) for hyper parameter tuning and runs on [Analytics Zoo RayOnSpark](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/).  

The general workflow using automated training contains below two steps. 
   1. create a ```AutoTSTrainer``` to train a ```TSPipeline```, save it to file to use later or elsewhere if you wish.
   2. use ```TSPipeline``` to do prediction, evaluation, and incremental fitting as well. 

You need to initialize RayOnSpark before using auto training (i.e. ```AutoTSTrainer```), and stop it after training finished. Using TSPipeline only does not need RayOnSpark. 
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

## References
* MTNet paper [link](https://arxiv.org/abs/1809.02105)
