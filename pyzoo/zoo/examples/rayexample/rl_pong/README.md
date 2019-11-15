# Pong example
This is an example to demonstrate how to use Analytics Zoo API to run [Ray](https://github.com/ray-project/ray) examples: 
[pong](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)

See [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/) for more details for RayOnSpark support in Analytics Zoo.

## Requirements 
- Python 3.5 or 3.6
- JDK 1.8
- Apache Spark 2.4.3 ( ≥ 2.4.0 )
- Analytics-Zoo 0.6.0
- Ray 0.6.6 and above
- Gym and gym[atari]

## Prepare environments
We recommend you to use [**Anaconda**](https://www.anaconda.com/distribution/#linux) to prepare the environments.
And follow step 1 to 4 [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/#steps-to-run-rayonspark) to prepare environment.

You also need to install **Gym** in your conda environment and make sure your pyspark version is above 2.4.0,
or if you use spark instead of pyspark, please set the environment arg **SPARK_HOME**
```shell script
export SPARK_HOME = PATH TO YOUR SPARK FLOD ( if you use pyspark, unset this)

pip install gym gym[atari]
```

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install)
for more running guidance after pip install. 

## Run on local after pip install
```
python rl_pong --iterations -1
```
See [here](#Options) for more configurable options for this example.

## Run on yarn cluster after pip install 
```
export YARN_CONF=... # path to your hadoop/yarn directory
python rl_pong.py --hadoop_conf $YARN_CONF --conda_name ... #your conda name
```
See [here](#Options) for more configurable options for this example.

## Options
- `--object_store_memory`  The store memory you need to use on local. Default is 4g.
- `--driver_cores` The number of driver's or local's cpu cores you want to use. Default is 8.
- `--interations` The number of iterations to train the model. Default is -1. And by default, the training would not stop.
- `--batch_size` The number of roll-outs to do per batch. Default is 10.

**Options only for yarn**
- `--hadoop_conf` This option is **required** when you want to run on yarn. The path to your configuration folder of hadoop.
- `--conda_name` This option is **required** when you want to run on yarn. Your conda environment's name.
- `--slave_num` The number of slave nodes you want to to use. Default is 2.
- `--executor_cores` The number of slave(executor)'s cpu cores you want to use. Default is 8.
- `--executor_memory` The size of slave(executor)'s memory you want to use. Default is 10g.
- `--driver_memory` The size of driver's memory you want to use. Default is 2g
- `--extra_executor_memory_for_ray` The size of slave(executor)'s extra memory to store data. Default is 20g.
