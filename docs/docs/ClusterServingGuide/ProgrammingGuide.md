# Programming Guide
Analytics Zoo Cluster Serving is a multi-model supporting and highly scalable service for model inference based on Analytics Zoo. It implements Pub-sub schema via integrating Analytics Zoo with data pipeline platform (e.g. Redis), task scheduling platform (e.g. Spark), where you can push data to the queue and get result from the queue.

Currently the data pipeline platforms we support include: Redis

This page contains the guide for you to run Analytics Zoo Cluster Serving, including following:

* [Quick Start]()
* [Configuration]()
* [Start and Stop Serving]()
* [Data Pipeline I/O]()
* [Logs and Visualization]()

## Quick Start

This section provides a quick start example for you to run Analytics Zoo Cluster Serving. To simplify the examples, we use docker to run Cluster Serving in these examples. If you do not have docker installed, [install docker]() first.

### Simplest End-to-end Example 
We use default `config.yaml` configuration, for more details, see [Configuration]()
```
model:
  # model path must be set
  path: /opt/work/model/
data:
  # default, localhost:6379
  src:
  # default, 3, 224, 224
  shape:
params:
  # default, 4
  batch_size:
  # default, mklblas
  engine_type:
  # default, 1
  top_n:
log: 
  # default, y
  error:
  # default, n
  summary:
spark:
  # default, local[*]
  master:
  # default, 4g
  driver_memory:
  # default, 1g
  executor_memory:
  # default, 1
  num_executors:
  # default, 4
  executor_cores:
  # default, 4
  total_executor_cores:
```



## Configuration

Your Cluster Serving configuration can all be set in `config.yaml`.

See an example of `config.yaml` below
```
## Analytics-zoo Cluster Serving

model:
  # model path must be set
  path: /opt/work/model
data:
  # default, localhost:6379
  src:
  # default, 3,224,224
  shape:
params:
  # default, 4
  batch_size: 64
  # default, 1
  top_n: 5  
  # default, mklblas
  engine_type:
log: 
  # default, y
  error:
  # default, n
  summary: y
spark:
  # default, local[*]
  master: local[*]
  # default, 4g
  driver_memory: 32g
  # default, 1g
  executor_memory: 128g
  # default, 1
  num_executors: 8
  # default, 4
  executor_cores: 1
  # default, 4
  total_executor_cores: 8
```
### Find Config File
You may have different approaches to access configuration file of Cluster Serving according to the way you run it. Before you run Cluster Serving, you have to find your config file first according to the instructions below and modify it to what your need.
#### Docker User
For Docker user, the `config.yaml` is in `analytics-zoo/docker/cluster-serving/config.yaml`


### Model configuration

#### Model Supported
Currently Analytics Zoo Cluster Serving supports models: Tensorflow, Caffe, Pytorch, BigDL, OpenVINO.

You need to put your model file into a directory and the directory could have layout like following according to model type

**Tensorflow**

```
|-- model
   |-- frozen_graph.pb
```

**Caffe**

```
|-- model
   |-- xx.prototxt
   |-- xx.caffemodel
```

**Pytorch**

```
|-- model
   |-- xx.pt
```

**BigDL**

```
|--model
   |-- xx.model
```

**OpenVINO**

```
|-- model
   |-- xx.xml
   |-- xx.bin
```

#### Docker User
If you run Cluster Serving with docker, put your model file into `model` directory. You do not need to set `model:path` in `config.yaml` because a default model location is set in docker image.

### Input Data Configuration
* src: the queue you subscribe for your input data, e.g. a default config of Redis on local machine is `localhost:6379`.
* shape: the shape of your input data, e.g. a default config for pretrained imagenet is `3,224,224`.

### Inference Parameter Configuration
* batch_size: the batch size you use for model inference, we recommend this value to be not small than 4 and not larger than 512, as batch size increases, you may get some gain in throughput and multiple times slow down in latency (inference time per batch).
* top_n: the top-N result you want for output, **note:** if the top-N number is larger than model output size of the the final layer, it would just return all the outputs.

### Log Configuration
* error: whether to write error to log file, "y" for yes, otherwise no
* summary: whether to write Cluster Serving summary to Tensorborad, "y" for yes, otherwise no

### Spark Configuration
* master: parameter `master` in spark
* driver_memory: parameter `driver-memory` in spark
* executor_memory: parameter `executor-memory` in spark
* num_executors: parameter `num-executors` in spark
* executor_cores: paramter `executor-cores` in spark
* total_executor_cores: parameter ` total-executor-cores` in spark

For more details of these config, please refer to [Spark Official Document](https://spark.apache.org/docs/latest/configuration.html)

## Start and Stop Serving

**Start**

`start-cluster-serving.sh`

**Stop**

`stop-cluster-serving.sh`

**Restart**

`restart-cluster-serving.sh`

## Data Pipeline I/O
use api guide
## Logs and Visualization

### Logs

We use log to save serving information and error, to enable this feature, use following config in [Configuration](). By default, this feature is enabled.
```
log:
  error: y
```
If you are the only user to run Cluster Serving, the error logs would also print to your interactive shell. Otherwise, you can not see the logs in the terminal. In this ocasion, you have to refer to your log.

To see your log, run 

**Serving Logs**

`cluster-serving-log.sh`

**Redis Logs**

`redis-log.sh`

### Visualization

we use tensorboard

We integrate Tensorboard into Cluster Serving. This feature is enabled by default. By disabling this feature, you could have a slight gain of serving performance since there is some cost to stat the information.

```
log:
  summary: y
```
Tensorboard service is started with Cluster Serving, once your serving is run, you can go to `localhost:6006` to see visualization of your serving.

Analytics Zoo Cluster Serving provides 3 attributes in Tensorboard so far, `Micro Batch Throughput`, `Partition Number`, `Total Records Number`.

* `Micro Batch Throughput`: The overall throughput, including preprocessing and postprocessing of your serving, the line should be relatively stable after first few records. If this number has a drop and remains lower than previous, you might have lost the connection of some nodes in your cluster.

* `Partition Number`: The partition number of your serving, this number should be stable all the time, and note that if you have N nodes in your cluster, you should have this partition number at least N.

* `Total Records Number`: The total number of records that serving gets so far.

**Note**: If you run serving on local mode, you could get another attribute `Throughput`, this is the throughput of prediction only, regardless of preprocessing and post processing. If you run serving on cluster mode, you could only see this attribute on remote nodes.

### Example
See [Quick Start](##quick-start) here to practise how to utilize summary and log of Analytics Zoo Cluster Serving.
