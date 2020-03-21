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

4. For multivariant forecasting (i.e. using ```MTNetForecaster```) where number of targets to predict >= 1. The input data shape should match 
   * X shape should be ```(num of samples, lookback, feature_dim)```, for train, validation and test data
       * Note that for MTNetForecaster, ```lookback``` = ```(lb_long_steps+1) * lb_long_stepsize```
   * Y shape should be 
       * ```(num of samples, num_of_targets)``` if horizon = 1
       * ```(num of samples, num_of_targets, horizon)``` if horizon > 1 && num_targets > 1
       * ```(num of samples, horizon)``` if num_targets = 1 (fallback to univariant forecasting)

### Use AutoML for training time series pipeline
 ```
 ```
 

## Example and References
* Example notebook can be found in ```analytics-zoo/apps/zouwu/network_traffic```
* MTNet paper [link](https://arxiv.org/abs/1809.02105)
