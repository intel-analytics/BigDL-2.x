# Programming Guide
Analytics Zoo Cluster Serving is a lightweight distributed, real-time serving solution that supports a wide range of deep learning models (such as TensorFlow, PyTorch, Caffe, BigDL and OpenVINO models). It provides a simple pub/sub API, so that the users can easily send their inference requests to the input queue (using a simple Python API); Cluster Serving will then automatically manage the scale-out and real-time model inference across a large cluster (using distributed streaming frameworks such as Apache Spark Streaming, Apache Flink, etc.) 

This page contains the guide for you to run Analytics Zoo Cluster Serving, including following:

* [Quick Start]()

* [Build Your Own Cluster Serving]()

   1. [Prepare (modify configuration)]() 
   
   2. [Launch Cluster Serving]()
   
   3. [Conduct Inference]()

* [Optional Operations]()

     [Logs and Visualization]()

     [Update Model]()
     
     [Manually Start and Stop Serving]()

## Quick Start

This section provides a quick start example for you to run Analytics Zoo Cluster Serving. To simplify the examples, we use docker to run Cluster Serving in these examples. If you do not have docker installed, [install docker]() first.

Clone Analytics Zoo Repo to your local repository and go to `analytics-zoo/docker/cluster-serving/`. Download the model [here]() and copy the files to your `model` directory. Your directory content should be:
```
cluster-serving | 
               -- |model
                 -- xx.caffemodel
                 -- xx.prototxt
```
Then use one command to start Cluster Serving.
```
docker run -itd --name cluster-serving --net=host -v $(pwd)/model:/opt/work/model -v $(pwd)/config.yaml:/opt/work/config.yaml analytics-zoo/cluster-serving:0.7.0-spark_2.4.3
```
Go to `analytics-zoo/pyzoo/zoo/serving/`, and run python program to push data into queue. Note that you need `pip install opencv-python` if they do not exist in your Python environment.

Then you can see the inference output. 
```
image: fish1.jpeg, classification-result: class: 1's prob: 0.9974158
image: cat1.jpeg, classification-result: class: 287's prob: 0.52377725
image: dog1.jpeg, classification-result: class: 207's prob: 0.9226527
```
Wow! You made it!

Note that you are using default configuration to run Cluster Serving, which is in local mode. To run in cluster mode, you could modify the configuration, see [Configuration Guide]() below.

For more details, you could also see the log and performance by go to `localhost:6006` in your browser and refer to [Log and Visualization](), or view the source code of `quick_start.py` [here](), or refer to [API Guide]().

## Build your Own Cluster Serving

### Prepare (modify configuration)
The Configuration file is `analytics-zoo/docker/cluster-serving/config.yaml`. Your Cluster Serving configuration can all be set by modifying this file.

See an example of `config.yaml` below
```
## Analytics Zoo Cluster Serving Config Example

model:
  # model path must be set
  path: /opt/work/model
data:
  # default, localhost:6379
  src:
  # default, 3,224,224
  image_shape:
params:
  # default, 4
  batch_size: 64
  # default, 1
  top_n: 5  
spark:
  # default, local[*], change this to spark://, yarn, etc if you want to run on cluster
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

#### Model configuration
##### Model Supported
Currently Analytics Zoo Cluster Serving supports models: Tensorflow, Caffe, Pytorch, BigDL, OpenVINO.

You need to put your model file into a directory and the directory could have layout like following according to model type, note that only one model is allowed in your directory.

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

##### Docker User
If you run Cluster Serving with docker, put your model file into `model` directory. You do not need to set `model:path` in `config.yaml` because a default model location is set in docker image.
##### DIY
For DIY user, you could put the model in your local directory, and set `model:/path/to/dir`.

#### Input Data Configuration
* src: the queue you subscribe for your input data, e.g. a default config of Redis on local machine is `localhost:6379`.
* shape: the shape of your input data, e.g. a default config for pretrained imagenet is `3,224,224`.

#### Inference Parameter Configuration
* batch_size: the batch size you use for model inference, we recommend this value to be not small than 4 and not larger than 512, as batch size increases, you may get some gain in throughput and multiple times slow down in latency (inference time per batch).
* top_n: the top-N result you want for output, **note:** if the top-N number is larger than model output size of the the final layer, it would just return all the outputs.

#### Spark Configuration
* master: parameter `master` in spark
* driver_memory: parameter `driver-memory` in spark
* executor_memory: parameter `executor-memory` in spark
* num_executors: parameter `num-executors` in spark
* executor_cores: paramter `executor-cores` in spark
* total_executor_cores: parameter ` total-executor-cores` in spark

For more details of these config, please refer to [Spark Official Document](https://spark.apache.org/docs/latest/configuration.html)
## Launch Cluster Serving
### Docker
For docker user, you can use following one line command to start Cluster Serving.
```
docker run -itd --name cluster-serving --net=host -v $(pwd)/model/:/opt/work/model/ -v $(pwd)/config.yaml:/opt/work/config.yaml analytics-zoo/cluster-serving:0.7.0-spark_2.4.3
```
### DIY
For DIY user, you need to install and start your a Redis server.

Download the spark-redis dependency jar [here]().

Go to `analytics-zoo/docker/cluster-serving/`, move two dependency jars to this directory. One is `analytics-zoo/dist/lib/*.jar` and another is the spark-redis jar which you download above.

Then, `bash start-cluster-serving.sh`.

If you want to visualize the inference summary, you also need to install Tensorboard.

## Conduct Inference
We support Python API for conducting inference with Data Pipeline in Cluster Serving. We provide basic usage here, for more details, please see [API Guide]().
### Input and Output API
To input data to queue, you need a `InputQueue` instance, and using `enqueue` method by giving an image path or image ndarray. See following example.
```
from zoo.serving.client import InputQueue
input_api = InputQueue()
input_api.enqueue_image('path/to/image')
```
To get data from queue, you need a `OutputQueue` instance, and using `query` or `dequeue` method. `query` method takes image uri as parameter and return the corresponding result, `dequeue` takes no parameter and just return all results and also delete them in data queue. See following example.
```
from zoo.serving.client import OutputQueue
output_api = OutputQueue()
img1_result = output_api.query('img1')
all_result = output_api.dequeue() # the output queue is empty after this code
```
### Output Format
Consider the code above, in [Input and Output API] Section.
```
img1_result = output_api.query('img1')
```
The `img1_result` is a json format string, like following:
```
'{"class_1":"prob_1","class_2":"prob_2",...,"class_n","prob_n"}'
```
Where `n` is the number of `top_n` in your configuration file. Taking the example of quick start above, an output string of `top_n: 1` could be:
```
'{"1":"0.9974158"}'
```
which means: (The code to parse the json string to below could be viewed in [Quick Start]())
```
image: fish1.jpeg, classification-result: class: 1's prob: 0.9974158
```

This string could be parsed by `json.loads`.
```
import json
result_class_prob_map = json.loads(img1_result)
```

## Optional Operations
### Manually Start and Stop Serving
Once you run docker image of Cluster Serving, the serving is automatically started. However, in some cases, you may need to keep your data in queue and manually start or stop the serving to release hardware resources or update model. Thus, we provide following scripts to start, stop, restart Cluster Serving. 
#### Start
If you have stopped serving, you could start it by `bash start-cluster-serving.sh`.

#### Stop
To stop Cluster Serving for some purpose, e.g. save compute resources and keep your data, your could run `bash stop-cluster-serving.sh`

#### Restart
In the case that Cluster Serving encounters some unknown error, you could restart serving by
`bash restart-cluster-serving.sh`

### Update Model
To update your model, you could replace your model file in your model directory, and restart Cluster Serving by `bash restart-cluster-serving.sh`. Note that you could also change your config in `config.yaml` and restart serving.

### Logs and Visualization

#### Logs

We use log to save Cluster Serving information and error.

To see log, for docker user, run `docker logs cluster-serving`, otherwise, you could view it through stdout.

#### Visualization

We integrate Tensorboard into Cluster Serving. 

Tensorboard service is started with Cluster Serving, once your serving is run, you can go to `localhost:6006` to see visualization of your serving.

Analytics Zoo Cluster Serving provides 3 attributes in Tensorboard so far, `Serving Throughput`, `Total Records Number`.

* `Serving Throughput`: The overall throughput, including preprocessing and postprocessing of your serving, the line should be relatively stable after first few records. If this number has a drop and remains lower than previous, you might have lost the connection of some nodes in your cluster.

* `Total Records Number`: The total number of records that serving gets so far.

