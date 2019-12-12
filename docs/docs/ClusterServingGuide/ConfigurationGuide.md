# Configuraion Guide

Your Cluster Serving configuration can all be set in `config.yaml`, see an example of `config.yaml` below
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
## Model configuration
### Model Supported 
Currently Analytics Zoo Cluster Serving supports models: Tensorflow, Caffe, Pytorch, BigDL, OpenVINO.

You need to put your model file into a directory and the directory could have layout like following according to model type
#### Tensorflow
```
|-- model
   |-- frozen_graph.pb
```
#### Caffe
```
|-- model
   |-- xx.prototxt
   |-- xx.caffemodel
```
#### Pytorch
```
|-- model
   |-- xx.pt
```
#### BigDL
```
|--model
   |-- xx.model
```
#### OpenVINO
```
|-- model
   |-- xx.xml
   |-- xx.bin
```

### Docker User
If you run Cluster Serving with docker, put your model file into `model` directory. You do not need to set `model:path` in `config.yaml` because a default model location is set in docker image.

## Input Data Configuration
* src: the queue you subscribe for your input data, e.g. a default config of Redis on local machine is `localhost:6379`.
* shape: the shape of your input data, e.g. a default config for pretrained imagenet is `3,224,224`.

## Inference Parameter Configuration
* batch_size: the batch size you use for model inference
* top_n: the top-N result you want for output, **note:** if the top-N number is larger than model output size of the the final layer, it would just return all the outputs.

## Log Configuration
* error: whether to write error to log file, "y" for yes, otherwise no
* summary: whether to write Cluster Serving summary to Tensorborad, "y" for yes, otherwise no

## Spark Configuration
* master: parameter `master` in spark
* driver_memory: parameter `driver-memory` in spark
* executor_memory: parameter `executor-memory` in spark
* num_executors: parameter `num-executors` in spark
* executor_cores: paramter `executor-cores` in spark
* total_executor_cores: parameter ` total-executor-cores` in spark

For more details of these config, please refer to [Spark Official Document](https://spark.apache.org/docs/latest/configuration.html)

