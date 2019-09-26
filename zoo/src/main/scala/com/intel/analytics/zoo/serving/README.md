# Analytics Zoo Cluster Serving

Analytics Zoo Cluster Serving is an easy to run service which you can leverage Analytics Zoo distributed framework to accelerate the inference task. It implements Pub-sub data flow model via queue (e.g. Redis) where you can put data into the queue and get result from the queue.

Currently the queue we support includes: Redis

## Start the Serving
### Run with docker

We suggest you to run Analytics Zoo Cluster Serving with Docker, it is the simplest way to run.
#### Prerequisites
To run with docker, what you need is

* docker
* your model (stored in `model:path:` in `config.yaml`)

#### Steps to run
1) open `config.yaml`, set `model:path:` to `/path/to/your/model/directory` (make sure there is only one model in your directory to avoid ambiguity), and set other parameters according to the instructions if you need.

2) run `bash docker-run.sh`.

## Data I/O

### Push and Get data to/from queue
You can call methods in `pyzoo/zoo/serving/api` to put the data into queue

Once the data is inqueued, Analytics Zoo Cluster Serving would dequeue the data from queue automatically, and do inference based on your model, and write result according to your config.

You can also get the result by calling methods in `pyzoo/zoo/serving/api` to get result.

Example code to push and get data is provided in `pyzoo/zoo/serving/api`.

## Model Supported
Currently Analytics Zoo Cluster Serving supports following model types, following "model directory" refers to the path you specify in `model:path` of `config.yaml`.

* Caffe - put definition file `.prototxt` and weight file `.caffemodel` into model directory
* TensorFlow - put frozen graph protobuf `.pb` into model directory, note that currently only `frozen_graph.pb` is supported
* BigDL - put weight file `.model` into model directory
* OpenVINO - put definition file `.xml` and weight file `.bin` into model directory

## FAQ
* Java heap space - If Cluster Serving ends by raising error `Java heap space`, try increase spark driver memory in `spark:driver_memory` of `config.yaml`.
* OutOfMemory (TensorFlow model) - If Cluster Serving (when running TensorFlow model) ends by raising error `OutOfMemory`, try reduce core number in `spark:master:local[core_number]` (in local mode) and `spark:executor_cores` (in distributed mode). The reason why TensorFlow needs different configuration is that the model preparing step is different from others.
* core dumped (OpenVINO model) - If Cluster Serving (when running OpenVINO model) ends by `core dumped`, try another machine may help (The error cause has not been confirmed yet)
