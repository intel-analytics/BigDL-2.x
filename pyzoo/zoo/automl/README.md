# (Experimental) AutoML
_A distributed **Automated Machine Learning** libary based on **ray** and **tensorflow, keras**_

---

This library provides a framework and implementations for automatic feature engineering, model selection and hyper parameter optimization. It also provide a built-in automatically optimized model: _**TimeSequencePredictor**_ , which can be used for time series data analysis or anomaly detection. 


## Automated training of Time Series Prediction Model 

### Training using _TimeSeuqencePredictor_

_TimeSequencePredictor_ can be used to train a model on historical time sequence data and predict future sequences. Current implementation only supports univariant prediction, which means target value should only be a scalar on each data point of the sequence. Input features can be multivariant.  

 1. Create a _TimeSequencePredictor_
```python
from zoo.automl.regression.time_sequence_predictor import TimeSequencePredictor

tsp = TimeSequencePredictor(dt_col="datetime", target_col="value", extra_features_col=None, future_seq_len=1)
```
 2. Train the model on historical time sequence, with automatic searching for better feature combinitions, model and optimization hyper parameters configurations. ```fit``` provide arguments for you to control roughly the number of trials during searching - it may take a long time to finish if you use **"long"** recipes. ```fit``` returns a _Pipeline_ object (see next section for details). Now we don't support resume training - i.e. calling ```fit``` multiple times retrains on the input data from scratch. 
```python
pipeline = tsp.fit(train_df, metric="mean_squared_error")
```

### Prediction and Evaluation using _TimeSequencePipeline_ 
A _TimeSequencePipeline_ contains a chain of feature transformers and models, which does end2end time sequence prediction on input data. _TimeSequencePipeline_ can be saved and loaded for future deployment.     
 * Prediction using _Pipeline_ object 
 ```python
 result_df = pipeline.predict(test_df)
 ```
 * Evaluation using _Pipeline_ object
 ```python
 mse, rs = pipeline.evaluate(test_df, metric=["mean_squared_error", "r_square"])
 ```
 * Save the _Pipeline_ object to a file
 ```python
 pipeline.save("/tmp/saved_pipeline/my.ppl")
 ```
 * Load the _Pipeline_ object from a file
 ```python
 from zoo.automl.pipeline.time_sequence import load_ts_pipeline
 pipeline = load_ts_pipeline("/tmp/saved_pipeline/my.ppl")
 ```

## Extensions and new AutoML applications
- _BaseFeatureTransformer_

- _BaseModel_

- _Pipeline_

- _SearchEngine_
