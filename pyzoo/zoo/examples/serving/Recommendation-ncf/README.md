# Cluster Serving
This example demonstrates how to use Cluster Serving with NCF model (Neural Collaborative Filtering).


## Environment
* Python 3.6/3.7
* JDK 8
* Apache Spark 2.x (This version needs to be same with the version you use to build Analytics Zoo)
* Redis 5.0 and above

## Install or download Analytics Zoo  
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.  

## Prepare the link
You can use the link [here](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/recommendation-ncf)
 to train ncf-bigdl model.

## Run after pip install 
* Step one: 

You can simply run command : `cluster-serving init`. It will set the envs for you.
And after the command, you can see `config.yaml` file in your current file path.

* Step two: 

Modify `config.yaml` with follow changes:
```
...
model:
  # model path must be set
  path: /path/to/your/model
...
  data_type: tensor
  # default, 3, 224, 224
  image_shape: 
  # must be provided given data_type is tensor. eg: [1,2] (tensor) [[1],[2,1,2],[3]] (table)
  tensor_shape: [[1],[1]]
  # default, topN(1)
  filter:
...
``` 

* Step three: 

Run `cluster-serving start`. Then you will see the log in your terminal.
* Step four: 

Open another terminal. Run `python input.py` then you can see the output `Result is :[[2,0.72023946]]`

## Run with pre-build package

* Step one: 

Run following commands:
```
export ANALYTICS_ZOO_HOME=/path/to/your/prebuild/package
source $ANALYTICS_ZOO_HOME/bin/analytics-zoo-env.sh
cd $ANALYTICS_ZOO_HOME/bin/cluster-serving
```
More information about cluster serving installation, please check the link [here.](https://analytics-zoo.github.io/master/#ClusterServingGuide/ProgrammingGuide/)

* Step two: 

After step one, you can follow step two & step three in pip part to load your model with cluster serving.

* Step three: 

Open another terminal and reset the env args:
```
export ANALYTICS_ZOO_HOME=/path/to/your/prebuild/package
source $ANALYTICS_ZOO_HOME/bin/analytics-zoo-env.sh
```
Then you can simply run `python input.py` then you can see the output `Result is :[[2,0.72023946]]`

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.