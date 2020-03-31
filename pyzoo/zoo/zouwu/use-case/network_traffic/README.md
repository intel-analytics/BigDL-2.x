## Network traffic use case in Zouwu

---
### Introduction
We demonstrate how to use Zouwu to forecast future network traffic indicators based on historical
time series data. 

In the reference use case, we use a public telco dataset, which is aggregated network traffic traces at the transit link of WIDE
to the upstream ISP ([dataset link](http://mawi.wide.ad.jp/~agurim/dataset/)). 
 

This use case example contains two notebooks:

- **network_traffic_autots_forecasting.ipynb** demonstrates how to use `AutoTS` to automatically
generate the best end-to-end time series analysis pipeline.

- **network_traffic_model_forecasting.ipynb** demonstrates how to leverage Zouwu's built-in models 
ie. LSTM and MTNet, to do time series forecasting. Both univariate and multivariate analysis are
demonstrated in the example.


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

### Usage

#### Prepare environment

We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments.
```
conda create -n zoo python=3.6 #zoo is conda enviroment name, you can set another name you like.
source activate zoo
pip install analytics-zoo[automl]
```
Note that the extra dependencies (including `ray`, `psutil`, `aiohttp`, `setproctitle`, `scikit-learn`,`featuretools`, `tensorflow`, `pandas`, `requests`, `bayesian-optimization`) will be installed by specifying `[automl]`.  
You can also find the details of zoo installation [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) 

#### Prepare dataset
* run `get_data.sh` to download the full dataset. It will download the monthly aggregated traffic data in year 2018 and 2019 (i.e "201801.agr", "201912.agr") into data folder. The raw data contains aggregated network traffic (average MBPs and total bytes) as well as other metrics.
* run `extract_data.sh` to extract relevant traffic KPI's from raw data, i.e. AvgRate for average use rate, and total for total bytes. The script will extract the KPI's with timestamps into `data/data.csv`.

#### Run Jupyter
* Install jupyter by `conda install jupyter`
* Run `jupyter notebook`.

