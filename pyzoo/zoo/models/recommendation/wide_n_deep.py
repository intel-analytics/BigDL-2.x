
# coding: utf-8

# ## Wide & Deep Recommender Demo

# Wide and Deep Learning Model, proposed by Google in 2016, is a DNN-Linear mixed model. Wide and deep learning has been used for Google App Store for their app recommendation.
# 
# In this tutorial, we use Recommender API of Analytics Zoo to build a wide linear model and a deep neural network, which is called Wide&Deep model, and use optimizer of BigDL to train the neural network. Wide&Deep model combines the strength of memorization and generalization. It's useful for generic large-scale regression and classification problems with sparse input features (e.g., categorical features with a large number of possible feature values).

# ## Intialization

# * import necessary libraries

# In[1]:


from zoo.models.recommendation import *
from zoo.models.recommendation.utils import *
from zoo.common.nncontext import init_nncontext
import os
import sys
import datetime as dt

import matplotlib

import matplotlib.pyplot as plt


# * Initilaize NN context, it will get a SparkContext with optimized configuration for BigDL performance.

# In[2]:


sc = init_nncontext("WideAndDeep Example")


# ## Data Preparation

# * Download and read movielens 1M rating data, understand the dimension.

# In[3]:


from bigdl.dataset import movielens
movielens_data = movielens.get_id_ratings("/tmp/movielens/")
min_user_id = np.min(movielens_data[:,0])
max_user_id = np.max(movielens_data[:,0])
min_movie_id = np.min(movielens_data[:,1])
max_movie_id = np.max(movielens_data[:,1])
rating_labels= np.unique(movielens_data[:,2])

print(movielens_data.shape)
print(min_user_id, max_user_id, min_movie_id, max_movie_id, rating_labels)


# * Transform ratings into dataframe, read user and item data into dataframes.

# In[5]:


sqlContext = SQLContext(sc)
from pyspark.sql.types import *
from pyspark.sql import Row

Rating = Row("userId", "itemId", "label")
User = Row("userId", "gender", "age" ,"occupation")
Item = Row("itemId", "title" ,"genres")

ratings = sc.parallelize(movielens_data)     .map(lambda line: map(int, line))     .map(lambda r: Rating(*r))
ratingDF = sqlContext.createDataFrame(ratings)

users= sc.textFile("/tmp/movielens/ml-1m/users.dat")    .map(lambda line: line.split("::")[0:4])    .map(lambda line: (int(line[0]), line[1], int(line[2]), int(line[3])))    .map(lambda r: User(*r))
userDF = sqlContext.createDataFrame(users)

items = sc.textFile("/tmp/movielens/ml-1m/movies.dat")     .map(lambda line: line.split("::")[0:3])     .map(lambda line: (int(line[0]), line[1], line[2].split('|')[0]))     .map(lambda r: Item(*r))
itemDF = sqlContext.createDataFrame(items)


# * Join data together, and transform data. For example, gender is going be used as categorical feature, occupation and gender will be used as crossed features.

# In[6]:


from pyspark.sql.functions import col, udf

gender_udf = udf(lambda gender: categorical_from_vocab_list(gender, ["F", "M"], start=1))
bucket_cross_udf = udf(lambda feature1, feature2: hash_bucket(str(feature1) + "_" + str(feature2), bucket_size=100))
genres_list = ["Crime", "Romance", "Thriller", "Adventure", "Drama", "Children's",
      "War", "Documentary", "Fantasy", "Mystery", "Musical", "Animation", "Film-Noir", "Horror",
      "Western", "Comedy", "Action", "Sci-Fi"]
genres_udf = udf(lambda genres: categorical_from_vocab_list(genres, genres_list, start=1))
     
allDF = ratingDF.join(userDF, ["userId"]).join(itemDF, ["itemId"])         .withColumn("gender", gender_udf(col("gender")).cast("int"))         .withColumn("age-gender", bucket_cross_udf(col("age"), col("gender")).cast("int"))         .withColumn("genres", genres_udf(col("genres")).cast("int"))
allDF.show(5)


# * Speficy data feature information shared by the WideAndDeep model and its feature generation. Here, we use occupation gender for wide base part, age and gender crossed as wide cross part, genres and gender as indicators, userid and itemid for embedding.  

# In[7]:


max_user_id = 1000
max_movie_id = 1000
bucket_size = 100
column_info = ColumnFeatureInfo(
            wide_base_cols=["occupation", "gender"],
            wide_base_dims=[21, 3],
            wide_cross_cols=["age-gender"],
            wide_cross_dims=[bucket_size],
            indicator_cols=["genres", "gender"],
            indicator_dims=[19, 3],
            embed_cols=["userId", "itemId"],
            embed_in_dims=[max_user_id, max_movie_id],
            embed_out_dims=[64, 64],
            continuous_cols=["age"])


# * Transform data to  RDD of Sample. We use optimizer of BigDL directly to train the model, it requires data to be provided in format of RDD(Sample). A Sample is a BigDL data structure which can be constructed using 2 numpy arrays, feature and label respectively. The API interface is Sample.from_ndarray(feature, label).  Wide&Deep model need two input tensors, one is SparseTensor for the Wide model, another is a DenseTensor for the Deep model.

# In[41]:


rdds = allDF\
    .filter(col("userId") <= 1000)\
    .filter(col("itemId") <= 1000).rdd.map(lambda row: to_user_item_feature(row, column_info)).repartition(4)
trainPairFeatureRdds, valPairFeatureRdds = rdds.randomSplit([0.8, 0.2], seed= 1)
valPairFeatureRdds.persist()
train_data= trainPairFeatureRdds.map(lambda pair_feature: pair_feature.sample)
test_data= valPairFeatureRdds.map(lambda pair_feature: pair_feature.sample)


# ##  Create the Wide&Deep model.

# * In Analytics Zoo, it is simple to build Wide&Deep model by calling WideAndDeep API. You need specify model type, and class number, as well as column information of features according to your data. You can also change other default parameters in the network, like hidden layers. The model could be fed into an Optimizer of BigDL or NNClassifier of analytics-zoo. Please refer to the document for more details. In this example, we demostrate how to use optimizer of BigDL.

# In[31]:


wide_n_deep = WideAndDeep(5, column_info, "wide_n_deep")


# ## Create optimizer and train the model

# In[32]:


wide_n_deep.compile(optimizer = "adam",
                    loss= SparseCategoricalCrossEntropy(zero_based_label = False),
                    metrics=['accuracy'])


# In[33]:


tmp_log_dir = create_tmp_path()
wide_n_deep.set_tensorboard(tmp_log_dir, "training_wideanddeep")


# Train the network. Wait some time till it finished.. Voila! You've got a trained model

# In[34]:


#%%time
# Boot training process
wide_n_deep.fit(train_data,
                batch_size = 400,
                nb_epoch = 10,
                validation_data = test_data)
print("Optimization Done.")


# ## Prediction and recommendation

# * Zoo models make inferences based on the given data using model.predict(val_rdd) API. A result of RDD is returned. predict_class returns the predicted label.

# In[42]:


results = wide_n_deep.predict(test_data)
results.take(5)

results_class = wide_n_deep.predict_class(test_data)
results_class.take(5)


# * In the Analytics Zoo, Recommender has provied 3 unique APIs to predict user-item pairs and make recommendations for users or items given candidates.
# * Predict for user item pairs

# In[36]:


userItemPairPrediction = wide_n_deep.predict_user_item_pair(valPairFeatureRdds)
for result in userItemPairPrediction.take(5): print(result)


# * Recommend 3 items for each user given candidates in the feature RDDs

# In[37]:


userRecs = wide_n_deep.recommend_for_user(valPairFeatureRdds, 3)
for result in userRecs.take(5): print(result)


# * Recommend 3 users for each item given candidates in the feature RDDs

# In[38]:


itemRecs = wide_n_deep.recommend_for_item(valPairFeatureRdds, 3)
for result in itemRecs.take(5): print(result)


# In[48]:


#%%time
# test_data.take(3)
# test_rdd = sc.parallelize(test_data.collect())
# evaluate_result=wide_n_deep.evaluate(test_rdd, 200)
# print("Top1 accuracy: %s" % evaluate_result[0].result)


# ##  Draw the convergence curve

# In[44]:




# plt.figure(figsize = (12,12))
# plt.subplot(2,1,1)
# plt.plot(loss[:,0],loss[:,1],label='loss')
# plt.xlim(0,loss.shape[0]+10)
# plt.grid(True)
# plt.title("loss")
# plt.subplot(2,1,2)
# plt.plot(top1[:,0],top1[:,1],label='top1')
# plt.xlim(0,loss.shape[0]+10)
# plt.title("top1 accuracy")
# plt.grid(True)


# In[43]:


train_loss = np.array(wide_n_deep.get_train_summary("Loss"))
val_loss = np.array(wide_n_deep.get_validation_summary("Loss"))
#plot the train and validation curves
# each event data is a tuple in form of (iteration_count, value, timestamp)
plt.plot(train_loss[:,0],train_loss[:,1],label='train loss')
plt.plot(val_loss[:,0],val_loss[:,1],label='val loss',color='green')
plt.scatter(val_loss[:,0],val_loss[:,1],color='green')
plt.legend();
plt.show()


# In[17]:


valPairFeatureRdds.unpersist()


# In[ ]:


sc.stop()

