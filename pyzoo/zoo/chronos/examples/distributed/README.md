# Run forecaster example on orca distributed
LSTM, TCN and Seq2seq users can easily train their forecasters in a distributed fashion to handle extra large dataset and utilize a cluster. The functionality is powered by Project Orca.

## Prepare the environment
We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:
```
conda create -n zoo python=3.7 # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install --pre --upgrade analytics-zoo[automl]
```

## Prepare data
we use the publicly available `network traffic` data repository maintained by the [WIDE project](http://mawi.wide.ad.jp/mawi/) and in particular, the network traffic traces aggregated every 2 hours (i.e. AverageRate in Mbps/Gbps and Total Bytes) in year 2018 and 2019 at the transit link of WIDE to the upstream ISP ([dataset link](http://mawi.wide.ad.jp/~agurim/dataset/))

`get_public_dataset` automatically download the specified data set and return the tsdata that can be used directly after preprocessing.
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

## Initiation forecaster and fit
```python
x_train, y_train = tsdata_train.to_numpy()
forecaster = Seq2SeqForecaster(past_seq_len=100,
                               future_seq_len=10,
                               input_feature_num=x_train.shape[-1],
                               output_feature_num=2,
                               metrics=['mse'],
                               distributed=True,
                               workers_per_node=args.workers_per_node,
                               seed=0)

# batch_size // worker_per_node: distribute data to all nodes. 
forecaster.fit((x_train, y_train), epochs=args.epochs,
               batch_size=512//(1 if not forecaster.distributed else args.workers_per_node))
```

## Evaluate
```python
mse = forecaster.evaluate((unscale_x_test, unscale_y_test))
print(f'evaluate is: {mse.get("MSE").numpy():.4f}')
```

## Options
* `--cluster_mode` The mode for the Spark cluster. local or yarn. Default to be `local`. You can refer to OrcaContext documents [here](https://analytics-zoo.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) for details.
* `--memory` The memory you want to use on each node. You can change it depending on your own cluster setting.
* `--cores` The number of cpu cores you want to use on each node. You can change it depending on your own cluster setting.
* `--epochs` Max number of epochs to train in each trial. Default to be 2.
* `--workers_per_node` the number of worker you want to use.The value defaults to 1. The param is only effective when distributed is set to True.
