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
Read relations from csv or txt file.
For csv file, it should be without header.
For txt file, each line should contain one record with fields separated by comma.