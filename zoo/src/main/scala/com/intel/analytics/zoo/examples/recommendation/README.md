# Overview

This is a Scala example for nueral collaberative filtering. We demostrate how to build neural network recommendation system with explict/implicit feedback. 
The system ([Recommendation systems: Principles, methods and evaluation](http://www.sciencedirect.com/science/article/pii/S1110866515000341)) normally prompts the user through the system interface to provide ratings for items in order to construct and improve his model. The accuracy of recommendation depends on the quantity of ratings provided by the user.  
Except traditional prediction, we provide 3 unique dataframe APIs to predict user item pairs, and recommend items(users) for for users(items). 

Data: 
* The dataset we used is movielens-1M ([link](https://grouplens.org/datasets/movielens/1m/)), which contains 1 million ratings from 6000 users on 4000 movies.  There're 5 levels of rating. We will try classify each (user,movie) pair into 5 classes and evaluate the effect of algortithms using Mean Absolute Error.  
  
References: 
* A Keras implementation of Movie Recommendation([notebook](https://github.com/ririw/ririw.github.io/blob/master/assets/Recommending%20movies.ipynb)) from the [blog](http://blog.richardweiss.org/2016/09/25/movie-embeddings.html).
* Nerual Collaborative filtering ([He, 2015](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf))


You can run the full NCFExmaple by following steps.

1. Prepare movie data.

Download movielens-1M ([link](https://grouplens.org/datasets/movielens/1m/)), unzip it and put into ./data/ml-1m/

2. Run this example

Command to run the example in Spark local mode:
```
    spark-submit \
    --master local[physcial_core_number] \
    --driver-memory 10g --executor-memory 20g \
    --class com.intel.analytics.zoo.example.recommendation.NCFExample \
    ./dist/lib/zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
    --batchSize 32 \
    --inputDir ./data/ml-1m \

```

Command to run the example in Spark yarn mode:
```
    spark-submit \
    --master yarn \
    --deploy-mode client \
    --executor-cores 8 \
    --num-executors 4 \
    --driver-memory 10g \
    --executor-memory 150g \
    --class com.intel.analytics.zoo.example.recommendation.NCFExample \
    ./dist/lib/zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
    --batchSize 32 \  
    --inputDir hdfs://xxx

```
