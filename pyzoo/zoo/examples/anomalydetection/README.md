## Summary
This is the example for anomaly detection, which demonstrates how to use Analytics Zoo to build `AnomalyDetector` based on LSTM to detect anomalies for time series data.

__Remark__: Due to some permission issue, this example cannot be run on Windows platform.


## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.


## Data Preparation
We use one of the datasets in Numenta Anomaly Benchmark (NAB[https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv]) for demo, i.e. NYC taxi passengers dataset, which contains 10320 records, each indicating the total number of taxi passengers in NYC at specific time. 
Before you run the example, download the data, unzip it and put into a directory.

The following scripts we provide will serve to download and extract the data for you:
```bash
bash ${ANALYTICS_ZOO_HOME}/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh dir
```
Remarks:
- `ANALYTICS_ZOO_HOME` is the folder where you extract the downloaded package and `dir` is the directory you wish to locate the corresponding downloaded data.
- If `dir` is not specified, the data will be downloaded to the current working directory.


## Run after pip install
You can easily use the following commands to run this example:
```bash
export SPARK_DRIVER_MEMORY=4g
nyc_path=the directory containing NBA nyc taxi data

python anomaly_detection.py --input_dir ${nyc_path} 
```
See [here](#options) for more configurable options for this example.

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.


## Run with prebuilt package
Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode:

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
nyc_path=the directory containing containing NBA nyc_taxi.csv data

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 4g \
    --executor-memory 4g \
    zoo/examples/anomalydetection/anomaly_detection.py --input_dir ${nyc_path}
```
See [here](#options) for more configurable options for this example.

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.


## Options
* `--input_dir` This option is __required__. The path where NBA nyc_taxi.csv locates.
* `-b` `--batch_size` The number of samples per gradient update. Default is 1024.
* `--nb_epoch` The number of iterations to train the model. Default is 20.
* `--unroll_length` The length of precious values to predict future value. Default is 24.
