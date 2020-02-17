## TFDataset


### Introduction

**TFDatasets** is the main entrance point in TFPark for importing and manipulating data.
It represents a distributed collection of elements (backed by a RDD) to be fed into a
TensorFlow graph for training, evaluation or inference. It provides a rich set of tools
to import data from various data sources and work as a unified interface to interact with
other components of TFPark.

This guide will walk you through some common cases of importing data and you can find detailed description
of TFDataset's API in [Analytics-Zoo API Guide](../../APIGuide/TFPark/tf-dataset.md).


### Basics

`TFDataset`'s job is to take in dataset, distribute the data across the Spark Cluster and transform each data
record into the format that is compatible with TFPark.

Here are a few common features that every TFDataset share:

1. `TFDataset` will automatically stack consecutive records into batches, so manually batching is not necessary
(and not supported). `batch_size` argument (for training) or `batch_per_thread` argument (for inference or evaluation)
should be set when creating TFDataset. The `batch_size` here is used for training and it means the total batch size
in distributed training. In other words, it equals to the total number of records processed in one iteration in the
whole cluster. `batch_size` should be a multiple of the total number of cores that is allocated for this Spark application
so that we can distributed the workload evenly across the cluster. You may need to adjust your other training
hyper-parameters when `batch_size` is changed. `batch_per_thread` is used for inference or evaluation
and it means the number of records process in one iteration in one partition. `batch_per_thread` is argument for tuning
performance and it does not affect the correctness or accuracy of the prediction or evaluation. Too small `batch_per_thread`
might slow down the prediction/evaluation.

2. For training, `TFDataset` can optionally takes a validation data source for validation at the the end of each epoch.
The validation data source should has the same structure of the main data source used for training.
 

### Working with in-memory ndarray

If your input data is quite small, the simplest way to create `TFDataset` to convert them to ndarrays and use
`TFDataset.from_ndarrays()`
E.g.

```python
import numpy as np
from zoo.tfpark import TFDataset
feature_data = np.random.randn(100, 28, 28, 1)
label_data = np.random.randint(0, 10, size=(100,))
dataset = TFDataset.from_ndarrays((feature_data, label_data), batch_size=32)
```

### Working with RDD data


### Working with text data

### Working with image data

### Working with Analytics Zoo Feature Engineering tools

### Working with TensorFlow Dataset

