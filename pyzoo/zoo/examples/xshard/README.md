## Summary
This is the example to demonstrates how to use Analytics Zoo Xshard data preprocesing APIs.
For the detail guide of Xshard data preprocessing, please refer [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/xshard)

## Install Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__.

## Data Preparation
We use one of the datasets in Numenta Anomaly Benchmark (NAB[https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv]) for demo, i.e. NYC taxi passengers dataset, which contains 10320 records, each indicating the total number of taxi passengers in NYC at specific time. 
Before you run the example, download the data(nyc_taxi.csv), and put into a directory.

## Run after pip install
You can easily use the following commands to run this example:
```bash
export SPARK_DRIVER_MEMORY=4g
nyc_path=the directory containing NBA nyc taxi data

python ray-pandas.py -f ${nyc_path} 
