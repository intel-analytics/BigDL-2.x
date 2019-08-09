# Overview

There are three Scala examples for recommender models, including wide and deep(WND) model, Neural network-based Collaborative Filtering(NCF) model and Session Recommender model.
The system ([Recommendation systems: Principles, methods and evaluation](http://www.sciencedirect.com/science/article/pii/S1110866515000341)) normally prompts the user through the system interface to provide ratings for items in order to construct and improve his model. The accuracy of recommendation depends on the quantity of ratings provided by the user.  

In NeuralCFexample and WideAndDeepExample, we demostrate how to use Analyticd Zoo to build neural network recommendation system with explict/implicit feedback, we also provide 3 unique APIs to predict user item pairs, and recommend items(users) for users(items). In the example of SessionRecExp, we demostrate how to build rnn based recommendations on short session data and history purchase.

## Data preparation: 
   The dataset we used for NCF is ([movielens-1M](https://grouplens.org/datasets/movielens/1m/)). Ratings.dat contains 1 million ratings from 6000 users on 4000 movies, 5 levels of rating are considered as 5 classes. Users.dat includes UserID, gender, age, occupation and zip-code. movies.dat includes movieID, title and genres. Please refer to ([readme](http://files.grouplens.org/datasets/movielens/ml-1m-README.txt)) for more details.  
   The datasets we used for WND are [movielens-1M](https://grouplens.org/datasets/movielens/1m/) and [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income).  
   If you run the example with movielens-1M dataset, download the data ([movielens-1M](https://grouplens.org/datasets/movielens/1m/)), unzip it and put into `./data/ml-1m/`. If you run the example with Census Income Dataset, download `adult.data` and `adult.test` to `./data/census`
   
   The dataset we used for SessionRecommender is ecommerce data provided by OfficeDepot.The dataset (atcHistory) describes agent’s purchase history and session add-to-cart items in sequence from OfficeDepot website (www.officedepot.com). It contains 406896 agents and 27482 items. These data were sampled and encoded for information security purposes. you can download the data and and put it into `./data/ecommerce/`

## Download Analytics Zoo
   You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.

## Wide and deep
   WND Learning Model, proposed by ([Google, 2016](https://arxiv.org/pdf/1606.07792.pdf)), is a DNN-Linear mixed model. WND combines the strength of memorization and generalization. It's useful for generic large-scale regression and classification problems with sparse input features(e.g., categorical features with a large number of possible feature values). It has been used for Google App Store for their app recommendation.
### Run the wide and deep example with ml-1m dataset
``` bash
   export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
   master=... // spark master
   ${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
   --master $master \
   --driver-memory 4g \
   --executor-memory 4g \
   --class com.intel.analytics.zoo.examples.recommendation.WideAndDeepExample \
   --inputDir ./data/ml-1m \
   --dataset ml-1m
```

### Run the wide and deep example with Census Income dataset
``` bash
   export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
   master=... // spark master
   ${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
   --master $master \
   --driver-memory 4g \
   --executor-memory 4g \
   --class com.intel.analytics.zoo.examples.recommendation.WideAndDeepExample \
   --inputDir ./data/census \
   --batchSize 320 \
   --maxEpoch 20 \
   --dataset census
```


## Neural network-based Collaborative Filtering
   NCF([He, 2015](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)) leverages a multi-layer perceptrons to learn the user–item interaction function, at the mean time, NCF can express and generalize matrix factorization under its framework. includeMF(Boolean) is provided for users to build a NCF with or without matrix factorization. 
### Run the NCF example
``` bash
   export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
   master=... // spark master
   ${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
   --master $master \
   --driver-memory 4g \
   --executor-memory 4g \
   --class com.intel.analytics.zoo.examples.recommendation.NeuralCFexample \
   --inputDir ./data/ml-1m 
```


## Session Recommender
   Session Recommender ([Hidasi, 2015](https://arxiv.org/pdf/1511.06939.pdf)) uses an RNN-based approach for session-based recommendations. The model is enhanced in NetEase ([Wu, 2016](https://ieeexplore.ieee.org/document/7498326)) by adding multiple layers to model users' purchase history. In Analytics Zoo, `includeHistory`(Boolean) is provided for users to build a `SessionRecommender` model with or without history. 
### Run the Session Recommender example
``` bash
   export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
   master=... // spark master
   ${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
   --master $master \
   --driver-memory 4g \
   --executor-memory 4g \
   --class com.intel.analytics.zoo.examples.recommendation.SessionRecExp \
   --input ./data/ecommerce
   --outputDir ./output/
```
## References: 
* A Keras implementation of Movie Recommendation, ([notebook](https://github.com/ririw/ririw.github.io/blob/master/assets/Recommending%20movies.ipynb)) from the [blog](http://blog.richardweiss.org/2016/09/25/movie-embeddings.html).
* Nerual Collaborative filtering, ([He, 2015](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf))
* Wide and deep Learning Model, ([Google, 2016](https://arxiv.org/pdf/1606.07792.pdf))
* Session-based recommendations with recurrent neura networks, ([Hidasi, 2015](https://arxiv.org/pdf/1511.06939.pdf))
* Personal recommendation using deep recurrent neural networks in netEase, ([Wu, 2016](https://ieeexplore.ieee.org/document/7498326))