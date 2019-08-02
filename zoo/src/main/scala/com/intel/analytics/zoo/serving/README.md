# Analytics Zoo Pub-sub-serving

Analytics Zoo Pub-sub-serving is a easy to run program which you can leverage Analytics Zoo distributed framework to accelerate the inference task. The pub-sub refers to a queue where you can put data into it and get result.

Currently the queue we support includes: Redis

### Prerequisites
The required environment you should prepare is simple, just a cluster with

* Spark 2.4.3 +

Besides, just as normal inference steps, you should prepare your data and model.


### Steps to run
1) modify the environment variables in `start.sh` according to your environments and requirements.

* SPARK_HOME: String, directory of your spark

* ModelType: String, The type of your model, currently supported value: caffe
* WeightPath: String, The path of file storing your model weight
* DefPath: String, The path of file storing your model definition (for caffe), if not `ModelType=caffe`, you could ignore this
* topN: Int, The number N of topN results you want to push into queue, default: 1
* redisPath: String, the url of your queue including host and port, default: `localhost:6379`

2) `sh start.sh` to start the service

3) you can push data into redis by reading the instructions in `/api/api.py`
