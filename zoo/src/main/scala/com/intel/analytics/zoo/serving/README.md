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

### Run with spark-submit 

#### Prerequisites
To run with spark-submit, what you need is

* Spark 2.4.0 +
* Analytics Zoo 0.6.0 +
* Redis
* your model

#### Steps to run
1) modify the environment variables in `start.sh` according to your environments and requirements.

* SPARK_HOME: String, directory of your spark

* ModelType: String, The type of your model, currently supported value: caffe
* WeightPath: String, The path of file storing your model weight
* DefPath: String, The path of file storing your model definition (for caffe), if not `ModelType=caffe`, you could ignore this
* topN: Int, The number N of topN results you want to push into queue, default: 1
* redisPath: String, the url of your queue including host and port, default: `localhost:6379`

2) `sh start.sh` to start the service

## Data I/O

### Push data into queue
You can call methods in `pyzoo/zoo/serving/api` to put the data into queue

Once the data is inqueued, Analytics Zoo Cluster Serving would dequeue the data from queue automatically, and do inference based on your model, and write result according to your config.

### Get data from queue
You can also get the result by calling methods in `pyzoo/zoo/serving/api` to get result.
