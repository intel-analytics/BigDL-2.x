TSDataset
===========

chronos.data.tsdataset
----------------------------------------

Time series data is a special data formulation with specific operations. TSDataset is an abstract of time series dataset, which provides various data processing operations (e.g. impute, deduplicate, resample, scale/unscale, roll) and feature engineering methods (e.g. datetime feature, aggregation feature). Cascade call is supported for most of the methods.
TSDataset can be initialized from a pandas dataframe and be converted to a pandas dataframe or numpy ndarray.

.. automodule:: zoo.chronos.data.tsdataset
    :members:
    :undoc-members:
    :show-inheritance:

chronos.data.experimental.xshards_tsdataset
----------------------------------------

Time series data is a special data formulation with specific operations. XShardsTSDataset is an abstract of time series dataset, which provides various data processing operations (e.g. impute, deduplicate, resample, scale/unscale, roll) and feature engineering methods (e.g. datetime feature, aggregation feature). Cascade call is supported for most of the methods.
XShardsTSDataset can be initialized from xshards of pandas dataframe and be converted to xshards of numpy in an distributed and parallized fashion.

.. automodule:: zoo.chronos.data.experimental.xshards_tsdataset
    :members:
    :undoc-members:
    :show-inheritance:

built-in dataset
--------------------------------------------

The build-in dataset has data download and preprocessing kilometers. Just specify the name and path, and the processed tsdata will be returned. Currently we support 
`nyc_taxi <https://raw.githubusercontent.com/numenta/NAB/v1.0/data/realKnownCause/nyc_taxi.csv>`__, `fsi <https://github.com/CNuge/kaggle-code/raw/master/stock_data/individual_stocks_5yr.zip>`__, `network_traffic <http://mawi.wide.ad.jp/~agurim/dataset/>`__, `AIOps <http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/machine_usage.tar.gz>`__.

.. automodule:: zoo.chronos.data.experimental.xshards_tsdataset
    :members:
    :undoc-members:
    :show-inheritance:
