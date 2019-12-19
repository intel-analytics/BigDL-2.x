# Analytics Zoo Cluster Serving

Analytics Zoo Cluster Serving is a multi-model supporting and highly scalable service for model inference based on [Analytics Zoo](). It implements Pub-sub schema via integrating Analytics Zoo with data pipeline platform (e.g. Redis), task scheduling platform (e.g. Spark),  where you can push data to the queue and get result from the queue.

Currently the data pipeline platforms we support include: Redis

To start Analytics Zoo Cluster Serving, you need the following steps (add links later)
1. [Configuration]()
2. [Start the Serving]()
3. [Data I/O]()

We suggest you refer to following Quick Start to begin your first practice.
## Quick Start
We provide the following instructions as a quick start of Analytics Zoo Cluster Serving. In this example, we will use Caffe model pretrained by ImageNet, and use docker image to start the serving. We will use the serving to do inference for several example images [here]().

1. Clone Analytics Zoo repo to your local repository or download the zip and unzip it.
```
git clone https://github.com/intel-analytics/analytics-zoo.git
```

2. Install [docker](), get Analytics Zoo Cluster Serving docker image [here](), download the image and load the image to your local.
```
docker load -i ./
```
3. `cd analytics-zoo/docker/cluster-serving` and you can see `config.yaml`. We provide the pretrained Caffe model which you can see [here]() in `model:path:` so that there is no need for you to set any config in this example.

4. Run the `docker-run` command in shell to start the serving.
```
docker run --name cluster-serving --net=host -v $(pwd)/model:/opt/work/model -v $(pwd)/config.yaml:/opt/work/config.yaml analytics-zoo/cluster-serving:0.7.0-spark_2.4.0
```
As you see 
```
INFO  ClusterServingHelper:58 - connected to redis localhost:6379
```
It means serving is ready for your data input.

5. In another terminal, push sample image [here]() to queue (in this example is Redis). Note that you need `pip install opencv-python` if your Python does not have it.

```
python ./analytics-zoo/pyzoo/zoo/serving/quick_start.py
```
Then you can see the inference output. 
```
image: fish1.jpeg, classification-result: class: 1's prob: 0.9974158
image: cat1.jpeg, classification-result: class: 287's prob: 0.52377725
image: dog1.jpeg, classification-result: class: 207's prob: 0.9226527
```
Wow! You made it! 


You could also see the log and performance according to [Log and Visualization]()


To stop the serving and remove the container.
```
docker stop cluster-serving
docker rm cluster-serving
```
So far, you have finish a brief end-to-end usage of Analytics Zoo Cluster Serving.

By the way, you could refer to below documents to see more details of Analytics Zoo Cluster Serving.
## Configuration
### Model
#### Set model path
You can set model path in `config.yaml`, `model: path`

Your path should be a directory containing model files, for example, for a caffe model defined by 2 files, `graph.prototxt` and `weight.caffemodel`, there should exists the following layout
```
- caffe_model (directory)
  - graph.prototxt
  - weight.caffemodel
```
and your `config.yaml` should contain following
```
model:
  path: path/to/caffe_model
```
#### Model Supported
Currently Analytics Zoo Cluster Serving supports following model types, following "model directory" refers to the path you specify in `model:path` of `config.yaml`.

* Caffe - put definition file `.prototxt` and weight file `.caffemodel` into model directory
* TensorFlow - put frozen graph protobuf `.pb` into model directory, note that currently only `frozen_graph.pb` is supported
* BigDL - put weight file `.model` into model directory
* OpenVINO - put definition file `.xml` and weight file `.bin` into model directory
* Pytorch - put weight file `.pt` into model directory
### Other parameters
#### data field
* src: the queue you subscribe for your input data, e.g. a default config of Redis on local machine is `localhost:6379`.
* shape: the shape of your input data, e.g. a default config for pretrained imagenet is `3,224,224`.
#### params field
* batch_size: the batch size you use for model inference
* top_n: the top-N result you want for output, **note:** if the top-N number is larger than model output size of the the final layer, it would just return all the outputs.
#### spark field
You should config this field if you are running Cluster Serving on Spark
* master: parameter `master` in spark
* driver_memory: parameter `driver-memory` in spark
* executor_memory: parameter `executor-memory` in spark
* num_executors: parameter `num-executors` in spark
* executor_cores: paramter `executor-cores` in spark
* total_executor_cores: parameter ` total-executor-cores` in spark

For more details of these config, please refer to [Spark Official Document](https://spark.apache.org/docs/latest/configuration.html)

## Start the Serving
Currently ClusterServing supports running with docker.
### Run with Docker

#### Prerequisites
To run with docker, what you need are:
* docker
* your model (stored in `model:path:` in `config.yaml`)
#### Steps to run
1) Open `config.yaml`, set `model:path:` to `/path/to/your/model/directory` (make sure there is only one model in your directory to avoid ambiguity), and set other parameters according to the instructions if you need.

2) Run `bash docker-build.sh` to build docker image to local (Or get pre-built docker image from our support).

3) Run `bash docker-run.sh`.
#### Stop the serving and change a model
To stop the serving in docker
```
docker stop cluster-serving
```
To remove the container no longer used in order to save disk space
```
docker rm cluster-serving
```
If you want to change a new model for serving, change the model path config in `config.yaml` and `bash docker-run.sh`.
#### Run multiple serving
If you want to run multiple serving on a same machine (not recommended because there would be no performance gain and output data may be in mess if you do not set configuration in the right way), you can modify `docker-run.sh` to set a new name of your serving container to avoid the naming conflict.

## Data I/O
### Push and Get data to/from queue
We provide Python API to interact with queues. Once the data is inqueued, Analytics Zoo Cluster Serving would dequeue the data from queue automatically, and do inference based on your model, and write result according to your config.

For detail API documentation, see [API Guide]().
#### Python API
```
input_api = Input()
input_api.enqueue_image(uri, data)
output_api = Output()
d = output_api.dequeue()
```
where `uri` is String type used as data identifier and `data` is `np.ndarray` type, `d` is a Python dict storing the result, format is in `key: class index, value: class probability`

## FAQ
* Java heap space - If Cluster Serving ends by raising error `Java heap space`, try increase spark driver memory in `spark:driver_memory` of `config.yaml`.
* OutOfMemory (TensorFlow model) - If Cluster Serving (when running TensorFlow model) ends by raising error `OutOfMemory`, try reduce core number in `spark:master:local[core_number]` (in local mode) and `spark:executor_cores` (in distributed mode). The reason why TensorFlow needs different configuration is that the model preparing step is different from others.
## Other Notes
For advanced users, you may refer to following notes to improve your Cluster Serving performance.
### System environment variables
* `OMP_NUM_THREADS`: OpenMP* Threads, affecting the performance of OpenVINO, Pytorch.
### Redis config
* Data persistence: We disable the Redis data persistence in Docker image in order to save disk, because so far we have not discovered the necessary scenario for persisting data and disabling persistence will get rid of potential mistakes. If you have the requirement to persist the Redis data on disk, please inform us, or run serving without Docker.
* Maximum stream length: We cut half of the Redis stream if the stream is about to reach the maximum size, if you want to remove this limitation, please inform us.
