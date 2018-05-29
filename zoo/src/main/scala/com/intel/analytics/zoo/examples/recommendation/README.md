# Overview

There are two Scala examples for recommender models, including wide and deep(WND) model and Neural network-based Collaborative Filtering(NCF) model.
The system ([Recommendation systems: Principles, methods and evaluation](http://www.sciencedirect.com/science/article/pii/S1110866515000341)) normally prompts the user through the system interface to provide ratings for items in order to construct and improve his model. The accuracy of recommendation depends on the quantity of ratings provided by the user.  

In these two examples, we demostrate how to use BigDL to build neural network recommendation system with explict/implicit feedback, we also provide 3 unique APIs to predict user item pairs, and recommend items(users) for users(items). 

## Data: 
   The dataset we used for both WND and NCF is ([movielens-1M](https://grouplens.org/datasets/movielens/1m/)). Ratings.dat contains 1 million ratings from 6000 users on 4000 movies, 5 levels of rating are considered as 5 classes. Users.dat includes UserID, gender, age, occupation and zip-code. movies.dat includes movieID, title and genres. Please refer to ([readme](http://files.grouplens.org/datasets/movielens/ml-1m-README.txt)) for more details.
   Before you run the example, download the data ([movielens-1M](https://grouplens.org/datasets/movielens/1m/)), unzip it and put into ./data/ml-1m/.

## Wide and deep
   WND Learning Model, proposed by ([Google, 2016](https://arxiv.org/pdf/1606.07792.pdf)), is a DNN-Linear mixed model. WND combines the strength of memorization and generalization. It's useful for generic large-scale regression and classification problems with sparse input features(e.g., categorical features with a large number of possible feature values). It has been used for Google App Store for their app recommendation.
### Run the wide and deep example
    Command to run the example in Spark local mode:
```
    spark-submit \
    --master local[physcial_core_number] \
    --driver-memory 10g --executor-memory 20g \
    --class com.intel.analytics.zoo.example.recommendation.WideAndDeepExample \
    ./dist/lib/zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
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
    --class com.intel.analytics.zoo.example.recommendation.WideAndDeepExample \
    ./dist/lib/zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
    --inputDir hdfs://xxx

```

## Neural network-based Collaborative Filtering
   NCF([He, 2015](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)) leverages a multi-layer perceptrons to learn the user–item interaction function, at the mean time, NCF can express and generalize matrix factorization under its framework. includeMF(Boolean) is provided for users to build a NCF with or without matrix factorization. 
### Run the NCF example
Command to run the example in Spark local mode:
```
    spark-submit \
    --master local[physcial_core_number] \
    --driver-memory 10g --executor-memory 20g \
    --class com.intel.analytics.zoo.example.recommendation.NeuralCFexample \
    ./dist/lib/zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
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
    --class com.intel.analytics.zoo.example.recommendation.NeuralCFexample \
    ./dist/lib/zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
    --inputDir hdfs://xxx

```

## References: 
* A Keras implementation of Movie Recommendation([notebook](https://github.com/ririw/ririw.github.io/blob/master/assets/Recommending%20movies.ipynb)) from the [blog](http://blog.richardweiss.org/2016/09/25/movie-embeddings.html).
* Nerual Collaborative filtering ([He, 2015](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf))
* Wide and deep Learning Model ([Google, 2016](https://arxiv.org/pdf/1606.07792.pdf))