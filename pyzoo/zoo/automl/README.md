# (Experimental) AutoML
_A distributed **Automated Machine Learning** libary based on **ray** and **tensorflow, keras**_

---

This library provides a framework and implementations for automatic feature engineering, model selection and hyper parameter optimization. We also provide a built-in automatically optimized model: _**TimeSeriesPredictor**_ , which can be used for time series data analysis or anomaly detection. 



## How to use _TimeSeuqencePredictor_ for time series analysis

Current implementation only supports univariant prediction, which means features can be multivariant, while target value should only be one on each data point of the sequence.  

```automl.regression.TimeSequencePredictor.fit```

```automl.regression.TimeSequencePredictor.evaluate ```

```automl.regression.TimeSequencePredictor.predict```


## Using automl library to implement your own model
Major elements
- ```FeatureTransformer```

- ```Model```

- ```Pipeline```

- ```SearchEngine```
