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
```shell script
conda create -n zoo python=3.6 #zoo is conda enviroment name, you can set another name you like.
source activate zoo
pip install ray
pip install psutil aiohttp gym gym[atari]
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.   

Use pip install is highly recommend.

## Run after pip install 
You can install Analytics-Zoo directly by `pip install analytics-zoo==0.6.0dev`.
### Run on local after pip install
```
python rl_pong --iterations -1
```
`--iterations` default set is 10, if you set -1 (<0 or too large),
it will not stop until you shut down.you can change them depend on what you need


### Run on yarn cluster after pip install 
```
export YARN_CONF=... # path to your hadoop/yarn directory
python rl_pong.py --hadoop_conf $YARN_CONF --conda_name ... #your conda name
```
- options   
`--iterations` can set in same way.   
More optional configs see [here](#options)

## options for yarn.
- `--conda_name`, default="ray36", your conda environment's name
- `--slave_num`, default=2, the number of slave nodes
- `--executor_cores`, default=8, slave(executor)'s cpu cores
- `--executor_memory`, default="10g", slave(executor)'s memory you want to use
- `--driver_memory`, default="2g", driver's memory you want to use
- `--driver_cores`, default=8, driver's cpu cores
- `--extra_executor_memory_for_ray`, default="20g", slave(executor)'s extra memory

## Optional config for all
- `--object_store_memory`, default="4g", store memory