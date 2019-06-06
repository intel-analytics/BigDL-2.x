# (Experimental) AutoML
_A distributed **Automated Machine Learning** libary based on **ray** and **tensorflow, keras**_


---

This library provides a framework and implementations for automatic feature engineering, model selection and hyper parameter optimization. It also provides a built-in automatically optimized model: _**TimeSequencePredictor**_ , which can be used for time series data analysis or anomaly detection. 


## 1 Automated Time Series Prediction 

### Training a model using _TimeSeuqencePredictor_

_TimeSequencePredictor_ can be used to train a model on historical time sequence data and predict future sequences. Note that: 
  * Current implementation only supports univariant prediction, which means target value should only be a scalar on each data point of the sequence. Input features can be multivariant.  
  * We require input time series data to be uniformly sampled in timeline. Missing data points will lead to errors or unreliable prediction result. 

 1. Create a _TimeSequencePredictor_
   * ```dt_col``` and ```target_col``` are datetime cols and target column in the input dataframe 
   * ```future_seq_len``` is how many data points ahead to predict. 
```python
from zoo.automl.regression.time_sequence_predictor import TimeSequencePredictor

tsp = TimeSequencePredictor(dt_col="datetime", target_col="value", extra_features_col=None, future_seq_len=1)
```

 2. Train on historical time sequence. 
   * ```recipe``` contains parameters to control the search space, stop criteria and number of samples (e.g. for random search strategy, how many samples are taken from the search space). Some recipe with large number of samples may lead to a large trial pool and take very long time to finish. Current avaiable recipes are: _BasicRecipe_ and _RandomRecipe_, both have argument ```num_samples``` to control the number of samples. Note that for BasicRecipe, the actual number of trials generated will be 2*```num_samples```, as it needs to do a grid search from 2 possble values for every random sample.   
   * ```fit``` returns a _Pipeline_ object (see next section for details). 
   * Now we don't support resume training - i.e. calling ```fit``` multiple times retrains on the input data from scratch. 
   * input train dataframe look like below: 
   
  |datetime|value|...|
  | --------|----- | ---|
  |2019-06-06|1.2|...|
  |2019-06-07|2.3|...|
  
```python
pipeline = tsp.fit(train_df, metric="mean_squared_error", recipe=RandomRecipe(num_samples=1))
```

### Prediction and Evaluation using _TimeSequencePipeline_ 
A _TimeSequencePipeline_ contains a chain of feature transformers and models, which does end-to-end time sequence prediction on input data. _TimeSequencePipeline_ can be saved and loaded for future deployment.      
 
 * Prediction using _Pipeline_ object

Output dataframe look likes below (assume predict n values forward). col `datetime` is the starting timestamp.  

  |datetime|value_0|value_1|...|value_{n-1}|
  | --------|----- | ------|---|---- |
  |2019-06-06|1.2|2.8|...|4.4|
 ```python
 result_df = pipeline.predict(test_df)
 ```
 
 * Evaluation using _Pipeline_ object
 ```python
 mse, rs = pipeline.evaluate(test_df, metric=["mean_squared_error", "r_square"])
 ```

### Saving and Loading a _TimeSequencePipeline_
 * Save the _Pipeline_ object to a file
 ```python
 pipeline.save("/tmp/saved_pipeline/my.ppl")
 ```
 * Load the _Pipeline_ object from a file
 ```python
 from zoo.automl.pipeline.time_sequence import load_ts_pipeline
 
 pipeline = load_ts_pipeline("/tmp/saved_pipeline/my.ppl")
 ```

## 2 AutoML Framework Overview

FeatureTransformers (implementations of _BaseFeatureTransformer_) and Models (implementations of _BaseModel_) are basic components of AutoML framework, each of which bears a set of hyper paramters that can be automatically configured using hyper parameter search engine (_SearchEngine_). Once the best configuration is found after several trials, such configuration is used to configure feature transformers and models, which are in turn used to compose a pipeline (implementation of _Pipeline_). Pipeline can be saved to file and loaded later for inference.    

* _zoo.automl.feature.abstract.BaseFeatureTransformer_ - abstract interface of feature transformer
* _zoo.automl.model.abstract.BaseModel_ - abstract interface of model
* _zoo.automl.search.abstract.SearchEngine_ - abstract interface of hyper paramter optimization engine.
* _zoo.automl.pipeline.abstract.Pipeline_ - abstract interface of pipeline, which is composed by a series of feature transfomers and models.

 

