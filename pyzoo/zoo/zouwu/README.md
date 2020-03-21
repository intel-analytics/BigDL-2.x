# Project Zouwu - Telco Solution on Analytis Zoo


## Requirements
* Python 3.6 or 3.7
* PySpark verison 2.4.3
* Ray version 0.7.0
* Keras verison 1.2.2
* Tensorflow version 1.15.0

## Install 
  * Download the Analytics Zoo source code (master). 
  * Build a local ```.whl``` wih for Spark2.4.3 using command below. For detailed explaination of the options for ```bulid.sh``` script, refer to [AnalytisZoo Python Developer's guide](https://analytics-zoo.github.io/master/#DeveloperGuide/python/#build-whl-package-for-pip-install)
```bash
bash analytics-zoo/pyzoo/dev/build.sh linux default -Dspark.version=2.4.3 -Dbigdl.artifactId=bigdl-SPARK_2.4 -P spark_2.4+
```
  * The succesfully built ```.whl``` file will locate in directory ```analytics-zoo/pyzoo/dist/```. Install the local .whl using below command. 
```
pip install analytics-zoo/pyzoo/dist/analytics_zoo-VERSION-py2.py3-none-PLATFORM_x86_64.whl
```


## Usage

### Forecast using Forecast Models (without AutoML)

The forecast models are all derived from [tfpark.KerasModels](https://analytics-zoo.github.io/master/#APIGuide/TFPark/model/). 

1. To start, you need to create a forecast model. Specify ```horizon``` and ```feature_dim``` in constructor. 
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

### Use AutoML for training time series pipeline

* AutoTSTrainer accepts data frames as input. An exmaple data frame looks like below. 

  |datetime|value|extra_feature_1|extra_feature_2|
  | --------|----- |---| ---|
  |2019-06-06|1.2|1|2|
  |2019-06-07|2.3|0|2|
  

* Create an AutoTSTrainer. Specify below arguments in constructor. 
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
 
* Use ```AutoTSTrainer.fit``` on train data and validation data. A TSPipeline will be returned. 
 ```python
 ts_pipeline = trainer.fit(train_df, validation_df)
 ```
 * Use ```TSPipeline.fit/evaluate/predict``` to train pipeline (incremental fitting), evaluate or predict. 
 ```python
 #incremental fitting
 ts_pipeline.fit(new_train_df, new_val_df, epochs=10)
 #evaluate
 ts_pipeline.evalute(val_df)
 ts_pipeline.predict(test_df)
 
 ```
 * Use ```TSPipeline.save/load``` to load from file or save to file. 
 ```python
 from zoo.zouwu.autots.forecast import TSPipeline
 loaded_ppl = TSPipeline.load(file)
 # ... do sth. e.g. incremental fitting
 loaded_ppl.save(another_file)
 ```

## Example and References
* Example notebook can be found in ```analytics-zoo/apps/zouwu/network_traffic```
* MTNet paper [link](https://arxiv.org/abs/1809.02105)
