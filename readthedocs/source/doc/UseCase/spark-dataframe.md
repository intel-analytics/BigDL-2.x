# Apache Spark Dataframes for Distribtued Deep Learning

---

<a target="_blank" href="https://colab.research.google.com/github/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/ncf_dataframe.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>&nbsp; <a target="_blank" href="https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/ncf_dataframe.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>

---
**In this guide we will describe how to use Spark Dataframes to scale-out data processing for distribtued deep learning.**

The dataset in this guide we used is [movielens-1M](https://grouplens.org/datasets/movielens/1m/), which contains 1 million ratings of 5 levels from 6000 users on 4000 movies. We will read the data into Spark Dataframe and directly use the Spark Dataframe as the input to the distributed training. 

### **1. Read file into Spark DataFrame** 

Spark supports to read files in CSV, JSON, TEXT, Parquet, and many more file formats into Spark DataFrame. Spark SQL provides `spark.read.csv("path")` to read a CSV file into Spark DataFrame. 

```python
from zoo.common.nncontext import *
sc = init_nncontext()
sqlcontext = SQLContext(sc)
# read csv with specifying column names
df = sqlcontext.read.csv(new_rating_files, sep=':', inferSchema=True).toDF(
  "user", "item", "label", "timestamp")
```

### **2. Preprocess Spark DataFrame** 

```python
# update label starting from 0. That's because ratings go from 1 to 5, while the matrix column index goes from 0 to 4
df = df.withColumn('label', df.label-1)

# split to train/test dataset
train_data, test_data = df.randomSplit([0.8, 0.2], 100)
```

### **3. Put Spark DataFrame in distributed training** 

```python
est = Estimator.from_keras(model_creator=model_creator, workers_per_node=1)

stats = est.fit(train_data,
                epochs=epochs,
                batch_size=batch_size,
                feature_cols=['user', 'item'], # specifies which column(s) to be used as inputs
                label_cols=['label'], # specifies which column(s) to be used as labels
                validation_data=test_data)
```
