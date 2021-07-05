Forecasters
=====================

chronos.model.forecast.lstm_forecaster
----------------------------------------

Long short-term memory(LSTM) is a special type of recurrent neural network(RNN). We implement the basic version of LSTM - VanillaLSTM for this forecaster for time-series forecasting task. It has two LSTM layers, two dropout layer and a dense layer.

For the detailed algorithm description, please refer to `here <https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/Chronos/Algorithm/LSTMAlgorithm.md>`__.

.. automodule:: zoo.chronos.model.forecast.lstm_forecaster
    :members:
    :undoc-members:
    :show-inheritance:


chronos.model.forecast.seq2seq_forecaster
-------------------------------------------

Seq2SeqForecaster wraps a sequence to sequence model based on LSTM, and is suitable for multivariant & multistep time series forecasting.

.. automodule:: zoo.chronos.model.forecast.seq2seq_forecaster
    :members:
    :undoc-members:
    :show-inheritance:


chronos.model.forecast.tcn_forecaster
----------------------------------------

Temporal Convolutional Networks (TCN) is a neural network that use convolutional architecture rather than recurrent networks. It supports multi-step and multi-variant cases. Causal Convolutions enables large scale parallel computing which makes TCN has less inference time than RNN based model such as LSTM.

.. automodule:: zoo.chronos.model.forecast.tcn_forecaster
    :members:
    :undoc-members:
    :show-inheritance:


chronos.model.forecast.tcmf_forecaster
----------------------------------------

Analytics Zoo Chronos TCMFForecaster provides an efficient way to forecast high dimensional time series.

TCMFForecaster is based on DeepGLO algorithm, which is a deep forecasting model which thinks globally and acts locally.
You can refer to `the deepglo paper <https://arxiv.org/abs/1905.03806>`__ for more details.

TCMFForecaster supports distributed training and inference. It is based on Orca PyTorch Estimator, which is an estimator to do PyTorch training/evaluation/prediction on Spark in a distributed fashion. Also you can choose to enable distributed training and inference or not.

**Remarks**:

* You can refer to `TCMFForecaster installation <https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/Chronos/tutorials/TCMFForecaster.md/#step-0-prepare-environment>`__ to install required packages.
* Your operating system (OS) is required to be one of the following 64-bit systems: **Ubuntu 16.04 or later** and **macOS 10.12.6 or later**.

.. automodule:: zoo.chronos.model.forecast.tcmf_forecaster
    :members:
    :undoc-members:
    :show-inheritance:


chronos.model.forecast.mtnet_forecaster
----------------------------------------

MTNet is a memory-network based solution for multivariate time-series forecasting. In a specific task of multivariate time-series forecasting, we have several variables observed in time series and we want to forecast some or all of the variables' value in a future time stamp.

MTNet is proposed by paper `A Memory-Network Based Solution for Multivariate Time-Series Forecasting <https://arxiv.org/abs/1809.02105>`__. MTNetForecaster is derived from tfpark.KerasMode, and can use all methods of KerasModel. Refer to `tfpark.KerasModel API Doc <https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/APIGuide/TFPark/model.md>`__ for details.

For the detailed algorithm description, please refer to `here <https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/Chronos/Algorithm/MTNetAlgorithm.md>`__.

.. automodule:: zoo.chronos.model.forecast.mtnet_forecaster
    :members:
    :undoc-members:
    :show-inheritance:


chronos.model.forecast.arima_forecaster
----------------------------------------

AutoRegressive Integrated Moving Average (ARIMA) is a class of statistical models for analyzing and forecasting time series data. It consists of 3 components: AR (AutoRegressive), I (Integrated) and MA (Moving Average). In ARIMAForecaster we use the SARIMA model (Seasonal ARIMA), which is an extension of ARIMA that additionally supports the direct modeling of the seasonal component of the time series.

.. automodule:: zoo.chronos.model.forecast.arima_forecaster
    :members:
    :undoc-members:
    :show-inheritance:


chronos.model.forecast.prophet_forecaster
----------------------------------------

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

For the detailed algorithm description, please refer to `here <https://github.com/facebook/prophet>`__.

.. automodule:: zoo.chronos.model.forecast.prophet_forecaster
    :members:
    :undoc-members:
    :show-inheritance:
    
    
chronos.model.forecast.tfpark_forecaster
----------------------------------------

.. automodule:: zoo.chronos.model.forecast.tfpark_forecaster
    :members:
    :undoc-members:
    :show-inheritance:
