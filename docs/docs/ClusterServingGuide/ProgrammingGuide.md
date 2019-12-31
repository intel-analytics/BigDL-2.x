# Programming Guide
Analytics Zoo Cluster Serving is a lightweight distributed, real-time serving solution that supports a wide range of deep learning models (such as TensorFlow, PyTorch, Caffe, BigDL and OpenVINO models). It provides a simple pub/sub API, so that the users can easily send their inference requests to the input queue (using a simple Python API); Cluster Serving will then automatically manage the scale-out and real-time model inference across a large cluster (using distributed streaming frameworks such as Apache Spark Streaming, Apache Flink, etc.) 

(Note currently only image classification models are supported).

This page contains the guide for you to run Analytics Zoo Cluster Serving, including following:

* [Quick Start]()

* [Deploy Your Own Cluster Serving]()

   1. [Installation]()

   2. [Configuration]() 
   
   3. [Launching Service]()
   
   4. [Model inference]()

* [Additional Operations]()

     - [Manually Start and Stop Serving]()

     - [Logs and Visualization]()

     - [Update Model]()     

## Quick Start

This section provides a quick start example for you to run Analytics Zoo Cluster Serving. To simplify the examples, we use docker to run Cluster Serving in these examples. If you do not have docker installed, [install docker]() first.

Download Analytics Zoo latest release [here]() to your local repository and go to `analytics-zoo/docker/cluster-serving/`. We have prepared a small Tensorflow model in your `analytics-zoo/docker/cluster-serving/model` directory. Your directory content should be:
```
cluster-serving | 
               -- | model
                 -- frozen_graph.pb
                 -- graph_meta.json
```
Then use one command to start Cluster Serving.
```
docker run -itd --name cluster-serving --net=host analytics-zoo/cluster-serving:0.7.0-spark_2.4.3
```
Install `opencv-python` if they do not exist in your Python environment.

Go to `analytics-zoo/pyzoo/zoo/serving/`, and run python program `python quick_start.py` to push data into queue and get inference result. 

Then you can see the inference output in console. 
```
image: fish1.jpeg, classification-result: class: 1's prob: 0.9974158
image: cat1.jpeg, classification-result: class: 287's prob: 0.52377725
image: dog1.jpeg, classification-result: class: 207's prob: 0.9226527
```
Wow! You made it!

Note that the Cluster Serving quick start example will run on your local node only. Check the [Build Your Own Cluster Serving]() section for how to configure and run Cluster Serving in a distributed fashion.

For more details, you could also see the log and performance by go to `localhost:6006` in your browser and refer to [Log and Visualization](), or view the source code of `quick_start.py` [here](), or refer to [API Guide]().

## Deploy your Own Cluster Serving
### 1. Installation
Currently Analytics Zoo Cluster Serving supports installation by docker, with all dependencies already packaged, which is recommended. If you do not install with docker, you can install by download release, pip. Note that in this way you need to install Redis and TensorBoard (for visualizing the serving status) on the local node.
#### Docker
```
docker pull zoo-cluster-serving
```
then,
```
docker run zoo-cluster-serving
```
Go inside the container and finish following operations.
#### Not Docker
For Not Docker user, first, install [Redis]() and [TensorBoard]() (for visualizing the serving status) and start them.

Download the spark-redis dependency jar [here](), go to `analytics-zoo/docker/cluster-serving/`, move two dependency jars to this directory. One is `analytics-zoo/dist/lib/*.jar` and another is the spark-redis jar which you download above.

Install Analytics Zoo by download release or pip.

##### Download Release
Download Analytics Zoo from [release page]() on the local node.

##### Pip
`pip install analytics-zoo`.

### 2. Configuration
#### 2.1 How to Config
The way to set configuration may be different depending on how you [install]() Cluster Serving.

Your Cluster Serving configuration can all be set by modifying config file `config.yaml`. See an example of `config.yaml` below
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
  batch_size:
  # default, 1
  top_n:
spark:
  # default, local[*], change this to spark://, yarn, k8s:// etc if you want to run on cluster
  master: local[*]
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
Config file `config.yaml` will be generated in your current working directory, you can set your config by modifying it.

#### 2.2 Preparing Model
Currently Analytics Zoo Cluster Serving supports Tensorflow, Caffe, Pytorch, BigDL, OpenVINO models. (Note currently only image classification models are supported).

You need to put your model file into a directory and the directory could have layout like following according to model type, note that only one model is allowed in your directory.

**Tensorflow**

```
|-- model
   |-- frozen_graph.pb
   |-- graph_meta.json
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

Put the model in any of your local directory, and set `model:/path/to/dir`.

#### 2.3 Other Configuration
The field `input` contains your input data configuration.

* src: the queue you subscribe for your input data, e.g. a default config of Redis on local machine is `localhost:6379`, note that please use the host address in your network instead of localhost or 127.0.0.1 when you run serving in cluster, make sure other nodes in cluster could also recognize this address.
* image_shape: the shape of your input data, e.g. a default config for pretrained imagenet is `3,224,224`, you should use the same shape of data which trained your model, in Tensorflow the format is usually HWC and in other models the format is usually CHW.

The field `params` contains your inference parameter configuration.

* batch_size: the batch size you use for model inference, we recommend this value to be not small than 4 and not larger than 512, as batch size increases, you may get some gain in throughput and multiple times slow down in latency (inference time per batch).
* top_n: the top-N result you want for output, **note:** if the top-N number is larger than model output size of the the final layer, it would just return all the outputs.

The field `spark` contains your spark configuration.

* master: Your cluster address, same as parameter `master` in spark
* driver_memory: same as parameter `driver-memory` in spark
* executor_memory: same as parameter `executor-memory` in spark
* num_executors: same as parameter `num-executors` in spark
* executor_cores: same as paramter `executor-cores` in spark
* total_executor_cores: same as parameter ` total-executor-cores` in spark

For more details of these config, please refer to [Spark Official Document](https://spark.apache.org/docs/latest/configuration.html)
### 3. Launching Service
```
cluster-serving-start
```

### 4. Model Inference
We support Python API for conducting inference with Data Pipeline in Cluster Serving. We provide basic usage here, for more details, please see [API Guide]().
### Input and Output API
To input data to queue, you need a `InputQueue` instance, and using `enqueue` method by giving an image path or image ndarray. See following example.
```
from zoo.serving.client import InputQueue
input_api = InputQueue()
input_api.enqueue_image('my-image1', 'path/to/image1')

import cv2
image2 = cv2.imread('path/to/image2')
input_api.enqueue_image('my-image2', image2)
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
A more readable format of this is shown in [Quick Start Example]() above: (The code to parse the json string to below could be viewed in [Quick Start]())
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
We provide following scripts to start, stop, restart Cluster Serving. 
#### Start
```
cluster-serving-start
```

#### Stop
```
cluster-serving-stop
```
#### Restart
```
cluster-serving-restart
```
Restart is usually used when config or model is updated and you have to restart serving to make it work.

### Update Model
To update your model, you could replace your model file in your model directory, and restart Cluster Serving by `cluster-serving-restart`. Note that you could also change your config in `config.yaml` and restart serving.

### Logs and Visualization

#### Logs

We use log to save Cluster Serving information and error.

To see log, for docker user, run `docker logs cluster-serving`, otherwise, you could view it through stdout.

#### Visualization

We integrate Tensorboard into Cluster Serving. 

Tensorboard service is started with Cluster Serving, once your serving is run, you can go to `localhost:6006` to see visualization of your serving.

Analytics Zoo Cluster Serving provides 2 attributes in Tensorboard so far, `Serving Throughput`, `Total Records Number`.

* `Serving Throughput`: The overall throughput, including preprocessing and postprocessing of your serving, the line should be relatively stable after first few records. If this number has a drop and remains lower than previous, you might have lost the connection of some nodes in your cluster.

* `Total Records Number`: The total number of records that serving gets so far.

