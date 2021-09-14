# Speed up predict
This example will demonstrate the effect of ONNX for predict on forecast and autotest.

## Prepare the environment
We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:
```
conda create -n zoo python=3.7 # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install --pre --upgrade analytics-zoo[automl]
```

## Options
* `--epoch` Max number of epochs to train in each trial. Default to be 1.

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

## Forecaster
Create an Seq2SeqForecaster
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

## Result
ONNX will not affect the result of evaluate, and will speed up predict.
```python
x_test, y_test = tsdata_train.to_numpy()
mse, smape = forecaster.evaluate((x_test,y_test))
# evaluate mse is: 0.0014
# evaluate smape is: 9.6629
mse, smape = forecaster.evaluate_with_onnx((x_test,y_test))
# evaluate_onnx mse is: 0.0014
# evaluate_onnx smape is: 9.6629

forecaster.predict(x_test)
# inference time is: ~0.136s
forecaster.predict_with_onnx(x_test)
# inference(onnx) time is: ~0.030s 
```