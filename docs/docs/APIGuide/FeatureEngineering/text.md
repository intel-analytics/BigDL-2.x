Analytics Zoo provides a series of Text APIs for end-to-end text processing pipeline,
including text loading, pre-processing, inference, etc.

---
## **Load Texts as TextSet**
`TextSet` is a collection of `TextFeature` where each `TextFeature` keeps information of a single text record.

`TextSet` can either be a `DistributedTextSet` with distributed text RDD or `LocalTextSet` with local text array.

You can read texts from local or distributed text path as a `TextSet` using the following API:

**Scala**
```scala
TextSet.read(path, sc = null, minPartitions = 1)
```

* `path`: String. Folder path to texts. Local file system and HDFS are supported. If you want to read from HDFS, `sc` needs to be defined.
Currently under this specified path, there are supposed to be several subdirectories, each of which contains several text files belonging to this category. 
Each category will be a given a label (starting from 0) according to its position in the ascending order sorted among all subdirectories. 
Each text will be a given a label according to the directory where it is located.
More text formats will be supported in the future.
* `sc`: An instance of SparkContext. If specified, texts will be read as a `DistributedTextSet`. 
Default is null and in this cases texts will be read as a `LocalTextSet`. 
* `minPartitions`: Integer. A suggestion value of the minimal partition number for input texts.
Only need to specify this when sc is not null. Default is 1.


**Python**
```python
TextSet.read(path, sc=None, min_partitions=1)
```

* `path`: String. Folder path to texts. Local file system and HDFS are supported. If you want to read from HDFS, `sc` needs to be defined.
Currently under this specified path, there are supposed to be several subdirectories, each of which contains several text files belonging to this category. 
Each category will be a given a label (starting from 0) according to its position in the ascending order sorted among all subdirectories. 
Each text will be a given a label according to the directory where it is located.
More text formats will be supported in the future.
* `sc`: An instance of SparkContext. If specified, texts will be read as a `DistributedTextSet`. 
Default is None and in this cases texts will be read as a `LocalTextSet`. 
* `min_partitions`: Int. A suggestion value of the minimal partition number for input texts.
Only need to specify this when sc is not None. Default is 1.


---
**TextSet Transformations**
