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

Clone Analytics Zoo Repo to your local repository and go to `analytics-zoo/docker/cluster-serving/`. Download the model [here]() and copy the files in your `model` directory, then use one command to start Cluster Serving.
```
docker run
```
Go to `analytics-zoo/pyzoo/zoo/serving/`, and run python program to push data into queue. Note that you need `pip install opencv-python` if they do not exist in your Python environment.

Then you can see the inference output. 
```
image: fish1.jpeg, classification-result: class: 1's prob: 0.9974158
image: cat1.jpeg, classification-result: class: 287's prob: 0.52377725
image: dog1.jpeg, classification-result: class: 207's prob: 0.9226527
```
Wow! You made it!

For more details, you could also see the log and performance by go to `localhost:6006` in your browser and refer to [Log and Visualization](), or view the source code of `quick_start.py` [here](), or refer to [API Guide]().


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

In some cases, you may need to keep your data in queue and control the serving at the same time. Thus, we provide following scripts to start, stop, restart Cluster Serving. 
### Start

Once you run docker image of Cluster Serving, the serving is automatically started. However, if you have stopped serving, you could start it by `bash start-cluster-serving.sh`.

### Stop

To stop Cluster Serving for soeme purpose, e.g. save compute resources and keep your data, your could run `bash stop-cluster-serving.sh`

### Restart

In the case that Cluster Serving encounters some unknown error, you could restart serving by
`restart-cluster-serving.sh`

## Data Pipeline I/O
We support Python API for Data Pipeline in Cluster Serving. We provide basic usage here, for more details, please see [API Guide]().
### Input API
To input data to queue, you need a `InputQueue` instance, and using `enqueue` method by giving an image path or image ndarray. See following example.
```
from zoo.serving.client import InputQueue
input_api = InputQueue()
input_api.enqueue_image('path/to/image')
```
### Output API
To get data from queue, you need a `OutputQueue` instance, and using `query` or `dequeue` method. `query` method takes image uri as parameter and return the corresponding result, `dequeue` takes no parameter and just return all results and also delete them in data queue. See following example.
```
from zoo.serving.client import OutputQueue
output_api = OutputQueue()
img1_result = output_api.query('img1')
all_result = output_api.dequeue() # the output queue is empty after this code
```

## Logs and Visualization

### Logs

We use log to save serving information and error.

To see your log, run 

**Serving Logs**

`bash cluster-serving-log.sh`

**Redis Logs**

`bash redis-log.sh`

### Visualization

We integrate Tensorboard into Cluster Serving. 

Tensorboard service is started with Cluster Serving, once your serving is run, you can go to `localhost:6006` to see visualization of your serving.

Analytics Zoo Cluster Serving provides 3 attributes in Tensorboard so far, `Serving Throughput`, `Total Records Number`.

* `Serving Throughput`: The overall throughput, including preprocessing and postprocessing of your serving, the line should be relatively stable after first few records. If this number has a drop and remains lower than previous, you might have lost the connection of some nodes in your cluster.

* `Total Records Number`: The total number of records that serving gets so far.

