# MTNetForecaster

## Introduction

MTNet is a memory-network based solution for multivariate time-series forecasting. In a specific task of multivariate time-series forecasting, we have several variables observed in time series and we want to forecast some or all of the variables' value in a future time stamp.

MTNet is proposed by paper [A Memory-Network Based Solution for Multivariate Time-Series Forecasting](https://arxiv.org/abs/1809.02105). MTNetForecaster is derived from tfpark.KerasMode, and can use all methods of KerasModel. Refer to [tfpark.KerasModel API Doc](../../APIGuide/TFPark/model.md) for details.

For the detailed algorithm description, please refer to [here](../Algorithm/MTNetAlgorithm.md).

## Method

### Arguments

- **`target_dim`**: Specify the number of variables we want to forecast. i.e. the the dimension of model output feature. This value defaults to 1.
- **`feature_dim`**: Specify the number of variables we have in the input data. i.e. the the dimension of model input feature. This value defaults to 1.
- **`lb_long_steps`**: Specify the number of long-term historical data series. This value defaults to 1. Typically, as stated in the [paper](https://arxiv.org/abs/1809.02105), the value is set to 7.
- **`lb_long_stepsize`**: Specify the length of long-term historical data series, which is equal to the length of short-term data series. This value defaults to 1. The value should be larger or equal to 1.
- **`ar_window_size`**: Specify the auto regression window size in MTNet. This value defaults to 1. Typically, the value is set between 1 to 3 as an integer. The value should larger or equal to `lb_long_stepsize`.
- **`cnn_kernel_size`**: Specify convolutional layer filter height in MTNet. This value defaults to 1. Typically, the value is set between 1 to 3 as an integer.
- **`metric`**: Specify the metric for validation and evaluation. This value defaults to MSE.
- **`uncertainty`**: Specify whether the forecaster can perform the calculation of uncertainty.

### \__init__

```python
MTNetForecaster(target_dim=1,
               feature_dim=1,
               lb_long_steps=1,
               lb_long_stepsize=1,
               ar_window_size=1,
               cnn_kernel_size=1,
               metric="mean_squared_error",
               uncertainty: bool = False,
            )

```

### fit, evaluate, predict

Refer to **fit**, **evaluate**, **predict** defined in [tfpark.KerasModel API Doc](../../APIGuide/TFPark/model.md)

## Reference

Yen-YuChang, Fan-YunSun, Yueh-HuaWu, Shou-DeLin,  [A Memory-Network Based Solution for Multivariate Time-Series Forecasting](https://arxiv.org/abs/1809.02105). 

