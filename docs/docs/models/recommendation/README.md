# Analytics Zoo Recommender API

Analytics Zoo provides two Recommenders, including Wide and Deep (WND) model and Neural network-based Collaborative Filtering (NCF) model. Each model could be fed into NNFrames and BigDL Optimizer directly for training.

Recommenders can handle models with either explict or implicit feedback, given corresponding features.

We also provide three user-friendly APIs to predict user item pairs, and recommend items (users) for users (items). 

## Wide and Deep
Wide and Deep Learning Model, proposed by [Google, 2016](https://arxiv.org/pdf/1606.07792.pdf), is a DNN-Linear mixed model, which combines the strength of memorization and generalization. It's useful for generic large-scale regression and classification problems with sparse input features (e.g., categorical features with a large number of possible feature values). It has been used for Google App Store for their app recommendation.

**Scala:**
```scala
WideAndDeep(modelType = "wide_n_deep", numClasses, columnInfo, hiddenLayers = Array(40, 20, 10))
```

Parameters:

* `modelType`: String. "wide", "deep", "wide_n_deep" are supported. Default is "wide_n_deep".
* `numClasses`: The number of classes. Positive integer.
* `columnInfo` An instance of [ColumnFeatureInfo]().
* `hiddenLayers`: Units of hidden layers for the deep model. Array of positive integers. Default is Array(40, 20, 10).

See [here](https://github.com/intel-analytics/zoo/blob/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/WideAndDeepExample.scala) for the Scala example that trains the `WideAndDeep` model on MovieLens 1M dataset and uses the model to do prediction and recommendation.


**Python**
```python
WideAndDeep(class_num, column_info, model_type="wide_n_deep", hidden_layers=(40, 20, 10))
```

Parameters:

* `class_num`: The number of classes. Positive int.
* `column_info`: An instance of [ColumnFeatureInfo]().
* `model_type`: String, 'wide', 'deep' and 'wide_n_deep' are supported. Default is 'wide_n_deep'.
* `hidden_layers`: Units of hidden layers for the deep model. Tuple of positive int. Default is (40, 20, 10).

See [here](https://github.com/intel-analytics/analytics-zoo/blob/master/apps/recommendation/wide_n_deep.ipynb) for the Python notebook that trains the `WideAndDeep` model on MovieLens 1M dataset and uses the model to do prediction and recommendation.


After training the model, users can predict user item pairs, and recommend items(users) for users(items) given a RDD of UserItemFeature, which includes user item-pair candidates and corresponding features.

```scala
val userItemPairPrediction = wideAndDeep.predictUserItemPair(validationpairFeatureRdds)
val userRecs = wideAndDeep.recommendForUser(validationpairFeatureRdds, 3)
val itemRecs = wideAndDeep.recommendForItem(validationpairFeatureRdds, 3)
```

## Neural network-based Collaborative Filtering
NCF ([He, 2015](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)) leverages a multi-layer perceptrons to learn the user–item interaction function. At the mean time, NCF can express and generalize matrix factorization under its framework. `includeMF`(Boolean) is provided for users to build a `NeuralCF` model with or without matrix factorization. 

**Scala**
```scala
NeuralCF(userCount, itemCount, numClasses, userEmbed = 20, itemEmbed = 20, hiddenLayers = Array(40, 20, 10), includeMF = true, mfEmbed = 20)
```

Parameters:

* `userCount`: The number of users. Positive integer.
* `itemCount`: The number of items. Positive integer.
* `numClasses`: The number of classes. Positive integer.
* `userEmbed`: Units of user embedding. Positive integer. Default is 20.
* `itemEmbed`: Units of item embedding. Positive integer. Default is 20.
* `hiddenLayers`: Units hiddenLayers for MLP. Array of positive integers. Default is Array(40, 20, 10).
* `includeMF`: Whether to include Matrix Factorization. Boolean. Default is true.
* `mfEmbed`: Units of matrix factorization embedding. Positive integer. Default is 20.

See [here](https://github.com/intel-analytics/analytics-zoo/blob/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/NeuralCFexample.scala) for the Scala example that trains the `NeuralCF` model on MovieLens 1M dataset and uses the model to do prediction and recommendation.


**Python**
```python
NeuralCF(user_count, item_count, class_num, user_embed=20, item_embed=20, hidden_layers=(40, 20, 10), include_mf=True, mf_embed=20)
```

Parameters:

* `user_count`: The number of users. Positive int.
* `item_count`: The number of classes. Positive int.
* `class_num:` The number of classes. Positive int.
* `user_embed`: Units of user embedding. Positive int. Default is 20.
* `item_embed`: itemEmbed Units of item embedding. Positive int. Default is 20.
* `hidden_layers`: Units of hidden layers for MLP. Tuple of positive int. Default is (40, 20, 10).
* `include_mf`: Whether to include Matrix Factorization. Boolean. Default is True.
* `mf_embed`: Units of matrix factorization embedding. Positive int. Default is 20.

See [here](https://github.com/intel-analytics/analytics-zoo/blob/master/apps/recommendation/ncf-explicit-feedback.ipynb) for the Python notebook that trains the `NeuralCF` model on MovieLens 1M dataset and uses the model to do prediction and recommendation.

After training the model, users can predict user item pairs, and recommend items(users) for users(items) given a RDD of UserItemFeature, which includes user item-pair candidates and corresponding features.

```scala
val userItemPairPrediction = ncf.predictUserItemPair(validationpairFeatureRdds)
val userRecs = ncf.recommendForUser(validationpairFeatureRdds, 3)
val itemRecs = ncf.recommendForItem(validationpairFeatureRdds, 3)
```

## Prediction and Recommendation

1. Predict for user-item pairs. RDD of [UserItemPrediction]() will be returned.

```scala
predictUserItemPair(featureRdd)
```

Parameters:
* `featureRdd`: RDD of [UserItemFeature]().


2. Recommend a number of items for each user.