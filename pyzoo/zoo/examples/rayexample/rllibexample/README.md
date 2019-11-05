# Multi agent training example
There are two examples to demonstrate how to use Analytics-Zoo API to run Ray examples: 
[multiagent_two_trainers](https://github.com/ray-project/ray/blob/master/rllib/examples/multiagent_two_trainers.py)

## Requirements 
- Python 3.5 or 3.6
- JDK 1.8
- Apache Spark 2.4.3 ( ≥ 2.4.0 )
- Analytics-Zoo 0.6.0
- Ray 0.7.2 and above(If you use ray0.6.6, you may change the code about some APIs since ray changed.)
- TensorFlow

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments, especially if you want to run on a yarn cluster(yarn-client mode only). 
```shell script
conda create -n zoo python=3.6 #zoo is conda enviroment name, you can set another name you like.
source activate zoo
pip install analytics-zoo
pip install ray
pip install psutil aiohttp pandas gym opencv-python
```
More install instructions see [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/).

More running guidance after pip install see [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install).

## Run on local after pip install
```
python multiagent_two_trainers.py --num-iters 10
```
More optional configs see [here](#Options)

## Run on yarn cluster after pip install 
```
export YARN_CONF=... # path to your hadoop/yarn directory

python multiagent_two_trainers.py --hadoop_conf $YARN_CONF --conda_name ... #your conda name
```
 
More optional configs see [here](#Options)

## Options
- `--object_store_memory` This option can be used on local mode. The store memory you need to use on local. Default is 4g.
- `--driver_cores` This option can be set in both mode. The number of driver's or local's cpu cores you want to use. Default is 8.
##### Options only for yarn
- `--hadoop_conf` This option is **required** when you want to run on yarn. The path to your configuration folder of hadoop.
- `--conda_name` This option is **required** when you want to run on yarn. Your conda environment's name.
- `--num-iters` The number of iterations to train the model. Default is 20.
- `--slave_num` The number of slave node you want to to use. Default is 2.
- `--executor_cores` The number of slave(executor)'s cpu cores you want to use. Default is 8.
- `--executor_memory` The size of slave(executor)'s memory you want to use. Default is 10g.
- `--driver_memory` The size of driver's memory you want to use. Default is 2g
- `--extra_executor_memory_for_ray` The size of slave(executor)'s extra memory to store data. Default is 20g.
