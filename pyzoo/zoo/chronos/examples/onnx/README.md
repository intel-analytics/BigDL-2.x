# Use ONNX to improve the speed of predict inference
This example will demonstrate the effect of ONNX for predict on forecast and autotest, which is expected to be ~5X.

## Prepare the environment
We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:
```
conda create -n env python=3.7 # "zoo" is conda environment name, you can use any name you like.
conda activate env
pip install analytics-zoo[automl]
```

## Getting Started with Orca
First, initialize [Orca Context](https://analytics-zoo.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html):
```python
num_nodes = 1 if args.cluster_mode == "local" else args.num_workers
init_orca_context(cluster_mode="yarn", cores=4,
                  memory="10g", num_nodes=num_nodes, init_ray_on_spark=True)
```

## Options
* `--cluster_mode` The mode for the Spark cluster. local or yarn. Default to be `local`. You can refer to OrcaContext documents [here](https://analytics-zoo.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) for details.
* `--epoch` Max number of epochs to train in each trial. Default to be 1.
* `--cpus_per_trail` Number of cpus for each trial. Default to be 2.
* `--n_sampling` Number of times to sample from the search_space. Default to be 1.
* `--memory` The memory you want to use on each node. Default to be 10g.
* `--num_workers` The number of workers to be used in the cluster. You can change it depending on your own cluster setting. Default to be 2.
* `--cores` "The number of cpu cores you want to use on each node. Default to be 4.

## Run on yarn cluster for yarn-client mode after pip install 
```
python onnx_nyc_taxi.py/onnx_network_traffic.py --cluster_model yarn
```

## Prepare data
**autotest**: We are using the `nyc taxi` provided by NAB, from 2014-07-01 to 2015-01-31 taxi fare information For more details, please refer to [here](https://raw.githubusercontent.com/numenta/NAB/v1.0/data/realKnownCause/nyc_taxi.csv)

**forecaster**: For demonstration, we use the publicly available `network traffic` data repository maintained by the [WIDE project](http://mawi.wide.ad.jp/mawi/) and in particular, the network traffic traces aggregated every 2 hours (i.e. AverageRate in Mbps/Gbps and Total Bytes) in year 2018 and 2019 at the transit link of WIDE to the upstream ISP ([dataset link](http://mawi.wide.ad.jp/~agurim/dataset/))

First, `get_public_dataset` automatically download the specified data set and return the tsdata that can be used directly after preprocessing.
```python
# Just specify the name and path, (e.g. network_traffic)
name = 'network_traffic'
path = '~/.chronos/dataset/'
tsdata, _, tsdata_test = get_public_dataset(name, path, with_split=True, val_ratio=0.1)
minmax = MinMaxScaler()
for tsdata in [tsdata_train, tsdata_test]:
    tsdata.gen_dt_feature(one_hot_features=["HOUR", "WEEK"])\
            .impute("last")\
            .scale(minmax, fit=tsdata is tsdata_train)
            .roll(lookback=40, horizon=1)
```

Next, Create an Seq2SeqForecaster
```python
forecaster = Seq2SeqForecaster(past_seq_len=40,
                               future_seq_len=1,
                               input_feature_num=32,
                               output_feature_num=2,
                               metrics=['mse', 'smape'],
                               seed=0)
```

Finally, call `fit` on Forecaster.
```python
# input tuple.
x_train, y_train = tsdata_train.to_numpy()
forecaster.fit((x_train, y_train), epochs=args.epochs)
```

## Reference
ONNX will not affect the result of evaluate, and will speed up predict.
```python
# evaluate/evaluate_with_onnx
# evaluate mse is: 0.0014
# evaluate smape is: 9.6629

# evaluate_onnx mse is: 0.0014
# evaluate_onnx smape is: 9.6629

# predict/predict_with_onnx:
# inference time is: 0.136s
# inference(onnx) time is: 0.030s 
```