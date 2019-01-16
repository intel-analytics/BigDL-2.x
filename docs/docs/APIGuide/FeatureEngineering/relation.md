A relation represents the relationship between two items.

**Scala**
```scala
relation = Relation(id1, id2, label)
```

**Python**
```python
relation = Relation(id1, id2, label)
```

* `id1`: String. The id of one item.
* `id2`: String. The id of the other item.
* `label`: Int. The label between the two items. Usually you can put 0 as label if they are unrelated and 1 if they are related.

---
## **Read Relations**
__From csv or txt file__

Each record is supposed to contain id1, id2 and label described above in order.

For csv file, it should be without header.
For txt file, each line should contain one record with fields separated by comma.

**Scala**
```scala
relationsRDD = Relations.read(path, sc, minPartitions = 1)
relationsArray = Relations.read(path)
```

**Python**
```python
relations_rdd = Relations.read(path, sc, min_partitions = 1)
relations_list = Relations.read(path)
```

* `path`: The path to the relations file, which can either be a local file path or HDFS path (in this case `sc` needs to be specified).
* `sc`: An instance of SparkContext. If specified, return RDD of Relation. Otherwise, return array or list of Relation.
* `min_partitions`: Integer. A suggestion value of the minimal partition number for input
texts. Only takes effect when sc is specified. Default is 1.

---
## **Generate Relation Pairs**