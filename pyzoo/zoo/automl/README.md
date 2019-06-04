# (Experimental) AutoML
_A distributed **Automated Machine Learning** libary based on **ray** and **tensorflow, keras**_

---

This library provides a framework and implementations for automatic feature engineering, model selection and hyper parameter optimization. We also provide a built-in automatically optimized model: _**TimeSequencePredictor**_ , which can be used for time series data analysis or anomaly detection. 


## Automated training of Time Series Prediction Model 

### Training a model using _TimeSeuqencePredictor_

You can use _TimeSequencePredictor_ to train a time series regression model. It trains a model on history time sequence data, and predict future sequences. Current implementation only supports univariant prediction, which means features can be multivariant, while target value should only be one on each data point of the sequence.  

 1. Create a _TimeSequencePredictor_
```python
from zoo.automl.regression.time_sequence_predictor import TimeSequencePredictor

tsp = TimeSequencePredictor(dt_col="datetime", target_col="value", extra_features_col=None, future_seq_len=1)
```
 2. Train the model on historical time sequence, with automatic searching for better feature combinitions, model and optimization hyper parameters configurations. We provide arguments for you to control roughly the number of trials during searching - it may take a long time to finish if you use **"long"** recipes. ```fit``` returns a _Pipeline_ object (see next section for details).
```python
pipeline = tsp.fit(train_df, metric="mean_squared_error")
```

### Prediction and Evaluation using _Pipeline_ object
A _Pipeline_ is a  which includes feature transformation and model inference. Pipeline can be saved and loaded for future deployment.     
 1. Prediction using _Pipeline_ object 
 ```python
 result_df = pipeline.predict(test_df)
 ```
 2. Evaluation using _Pipeline_ object
 ```python
 mse, rs = pipeline.evaluate(test_df, metric=["mean_squared_error", "r_square"])
 ```
 3. Save the _Pipeline_ object to a file
 ```python
 pipeline.save("/tmp/saved_pipeline/my.ppl")
 ```
 4. Load the _Pipeline_ object from a file
 ```python
 pipeline.load("/tmp/saved_pipeline/my.pipeline")
 ```

## Implement your own Automated ML model

- ```FeatureTransformer```

- ```Model```

- ```Pipeline```

- ```SearchEngine```
