---
## **Introduction**

Analytics Zoo Orca data provides data-parallel pre-processing support for Python AI.

It supports data pre-processing from different data sources, like TensorFlow DataSet, PyTorch DataLoader, MXNet DataLoader, etc. and it supports various data format, like Pandas DataFrame, Numpy, Images, Parquet.

The distributed backend engine can be [Spark](https://spark.apache.org/) or [Ray](https://github.com/ray-project/ray). We now support Spark-based transformations to do the pre-processing, and provide functionality to seamlessly move data to Ray cluster for later Ray based training/inference. 

---
## **XShards**

XShards is a collection of data in Orca data API. For different backend, we have two XShards: SparkXShards and RayXShards.

SparkXShards is a collection of data which can be pre-processed in parallel on Spark. SparkXShards can accept various data sources, like csv/json/parquet, Numpy, TensorFlow DataSet, Image, etc. And do the specified processing in parallel on Spark using common Python libraries such as Pandas, Numpy, PIL, TensorFlow Dataset, PyTorch DataLoader, etc.

RayXShards is a collection of data which can be pre-processed in parallel on Ray. We support ingesting data in parallel to Ray from SparkXShards, and users can performance later training/inference on Ray.


### **XShards General Operations**

#### **Pre-processing on XShards**

You can do pre-processing with your customized function on XShards using below API:
```
transform_shard(func, *args)
```
* `func` is your pre-processing function. In this function, you can do the pre-processing with the data using common Python libraries such as Pandas, Numpy, PIL, TensorFlow Dataset, PyTorch DataLoader, etc., then return the processed object. 
* `args` are the augurments for the pre-processing function.

This method would parallelly pre-process each element in the XShards with the customized function, and return a new XShards after transformation.

##### **SharedValue**
SharedValue can be used to give every node a copy of a large input dataset in an efficient manner.
This is an example of using SharedValue:
```
def func(df, item_set)
   item_set = item_set.value
   ....

item_set= ...
item_set= orca.data.SharedValue(item_set)
full_data.transform_shard(func, item_set)
```

#### **Get all the elements in XShards**

You can get all of elements in XShards with such API:
```
collect()
```
This method returns a list that contains all of the elements in this XShards.

To get the more examples on orca data API, you can refert to [Example](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/orca/data)

#### **Repartition XShards**

You can repartition XShards to different number of partitions.
```
repartition(num_partitions)
```
* `num_partitions` is the target number of partitions for the new XShards.

The method returns a new SparkXShards that has exactly num_partitions partitions.


#### **Split XShards**

You can split one XShards into multiple SparkXShards. Each element in the SparkXShards needs be a list or tuple with same length.
```
split()
```
This method returns splits of SparkXShards. If each element in the input SparkDataShard is not a list or tuple, return list of input SparkDataShards.

#### **Save/Load XShards**

You can save SparkXShards as SequenceFiles of serialized objects.
The serializer used is pyspark.serializers.PickleSerializer.
```
save_pickle(path, batchSize=10)
```
* `path` is target save path.
* `batchSize` batch size for each chunk in sequence file.

And you can load pickle file to XShards if you use save_pickle() to save data.
```
zoo.orca.data.XShards.load_pickle(path, minPartitions=None)
```
* `path`: The pickle file path/directory.
* `minPartitions`: The minimum partitions for the XShards.

This method return a SparkXShards object from pickle files.

#### **Move SparkXShards to Ray backend**

You can put data of the SparkXShards to Ray cluster object store for later processing on Ray.
```
to_ray()
```
This method save data of SparkXShards to Ray object store, and return a new RayXShards which contains plasma ObjectID, the plasma object_store_address and the node IP on each partition.



### **XShards with Pandas DataFrame**

#### **Read data into XShards**

You can read csv/json files/directory into XShards with such APIs:
```
zoo.orca.data.pandas.read_csv(file_path, **kwargs)

zoo.orca.data.pandas.read_json(file_path, **kwargs)
```
* The `file_path` could be a csv/json file, list of multiple csv/json file paths, a directory containing csv/json files. Supported file systems are local file system,` hdfs`, and `s3`.
* `**kwargs` is read_csv/read_json options supported by pandas.

After calling these APIs, you would get a XShards of Pandas DataFrame.

#### **Partition by Pandas DataFrame columns**
You can re-partition SparkXShards of Pandas DataFrame with specified columns.
```
partition_by(cols, num_partitions=None)
```
* `cols`: DataFrame columns to partition by.
* `num_partitions`: target number of partitions. If not specified, the new SparkXShards would keep the current partition number.

This method return a new SparkXShards partitioned using the specified columns.

#### **Get unique element list of XShards of Pandas Series**

You can get a unique list of elements of this SparkXShards. This is useful when you want to count/get unique set of some column in the XShards of Pandas DataFrame. 
```
unique()
```
This method return a unique list of elements of the SparkXShards of Pandas Series.

### **XShards with Numpy**

#### **Load Numpy data in XShards**

You can partition local in memory data and form a SparkXShards.
```
zoo.orca.data.XShards.partition(data)
```
* `data`: The local data can be numpy.ndarray, a tuple, list, dict of numpy.ndarray, or a nested structure made of tuple, list, dict with ndarray as the leaf value.

This method returns a SparkXShards which dispatch local data in parallel on Spark.



