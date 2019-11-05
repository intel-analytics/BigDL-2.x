# Pong example
This is an example to demonstrate how to use Analytics-Zoo API to run Ray examples: 
[pong]("https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5")

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__.

## Requirements 
- Python 3.5 or 3.6
- JDK 1.8
- Apache Spark 2.4.3 ( ≥ 2.4.0 )
- Analytics-Zoo 0.6.0
- Ray 0.6.6 and above
- Gym and gym[atari]

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments, especially if you want to run on a yarn cluster(yarn-client mode only). 
And follow the instruction [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/#steps-to-run-rayonspark) to prepare ray-on-spark environment.

And some packages are also needed:
```shell script
pip install gym gym[atari]
```
More install instructions see [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/).

More running guidance after pip install see [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install).

## Run on local after pip install
```
python rl_pong --iterations -1
```
Optional configs see [here](#Options)

## Run on yarn cluster after pip install 
```
export YARN_CONF=... # path to your hadoop/yarn directory
python rl_pong.py --hadoop_conf $YARN_CONF --conda_name ... #your conda name
```
Optional configs see [here](#Options)

## Options
- `--object_store_memory` This option can be used on local mode. The store memory you need to use on local. Default is 4g.
- `--driver_cores` This option can be set in both mode. The number of driver's or local's cpu cores you want to use. Default is 8.
- `--interations` This option can be set in both mode. The number of iterations to train the model. Default is -1. And by default, the training would not stop.
- `--batch_size` This option can be set in both mode. The number of roll-outs to do per batch. Default is 10.
#### Options only for yarn
- `--hadoop_conf` This option is **required** when you want to run on yarn. The path to your configuration folder of hadoop.
- `--conda_name` This option is **required** when you want to run on yarn. Your conda environment's name.
- `--slave_num` The number of slave node you want to to use. Default is 2.
- `--executor_cores` The number of slave(executor)'s cpu cores you want to use. Default is 8.
- `--executor_memory` The size of slave(executor)'s memory you want to use. Default is 10g.
- `--driver_memory` The size of driver's memory you want to use. Default is 2g
- `--extra_executor_memory_for_ray` The size of slave(executor)'s extra memory to store data. Default is 20g.
