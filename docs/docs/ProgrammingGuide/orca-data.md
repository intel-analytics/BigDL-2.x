---
## **Introduction**

Analytics Zoo Orca data provides data-parallel pre-processing support for Python AI.

It supports data pre-processing from different data sources, like TensorFlow DataSet, PyTorch DataLoader, MXNet DataLoader, etc. and it supports various data format, like Pandas DataFrame, Numpy, Images, Parquet.

The distributed backend engine can be [Spark](https://spark.apache.org/) or [Ray](https://github.com/ray-project/ray). We now support spark-based transformations to do the pre-processing, and provide functionality to seamlessly move data to Ray cluster for later Ray based training/inference. 

---
## **XShards**

XShards is a collection of data in Orca data API. 

### **XShards general operations**

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

#### **Save XShards**

You can save SparkXShards as SequenceFiles of serialized objects.
The serializer used is pyspark.serializers.PickleSerializer.
```
save_pickle(path, batchSize=10)
```
* `path` is target save path.
* `batchSize` batch size for each chunk in sequence file.

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


