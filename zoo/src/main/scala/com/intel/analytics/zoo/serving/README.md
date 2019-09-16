# Analytics Zoo Cluster Serving

Analytics Zoo Cluster Serving is an easy to run service which you can leverage Analytics Zoo distributed framework to accelerate the inference task. It implements Pub-sub data flow model via queue (e.g. Redis) where you can put data into the queue and get result from the queue.

Currently the queue we support includes: Redis

## Start the Serving
### Run with docker

We suggest you to run Analytics Zoo Cluster Serving with Docker, it is the simplest way to run.
#### Prerequisites
To run with docker, what you need is

* docker
* your model

#### Steps to run
1) open `config.yaml`, set `model:path:` to `/path/to/your/model/directory` (make sure there is only one model in your directory to avoid ambiguity), and set other parameters according to the instructions if you need.

2) run `bash docker-run.sh`

## Data I/O

### Push data into queue
You can call methods in `pyzoo/zoo/serving/api` to put the data into queue

Once the data is inqueued, Analytics Zoo Cluster Serving would dequeue the data from queue automatically, and do inference based on your model, and write result according to your config.

### Get data from queue
You can also get the result by calling methods in `pyzoo/zoo/serving/api` to get result.

## Model Supported
Currently Analytics Zoo Cluster Serving supports following model types

* Caffe
* TensorFlow
* BigDL
* OpenVINO
