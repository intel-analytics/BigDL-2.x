# Parameter server example
There are two examples to demonstrate how to use Analytics Zoo API to run [Ray](https://github.com/ray-project/ray) examples: 
[async_parameter_server](https://github.com/ray-project/ray/blob/master/doc/examples/parameter_server/async_parameter_server.py)
and [sync_parameter_server](https://github.com/ray-project/ray/blob/master/doc/examples/parameter_server/sync_parameter_server.py).

See [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/) for more details for RayOnSpark support in Analytics Zoo.

## Prepare environments
We recommend you to use [**Anaconda**](https://www.anaconda.com/distribution/#linux) to prepare the environments.
And follow step 1 to 4 [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/#steps-to-run-rayonspark) to prepare environment.

You also need to install **TensorFlow** in your conda environment and make sure your pyspark version is above 2.4.0,
or if you use spark instead of pyspark, please set the environment arg **SPARK_HOME**
```shell script
export SPARK_HOME = PATH TO YOUR SPARK FLOD ( if you use pyspark, unset this)

pip install tensorflow
```

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install)
for more running guidance after pip install. 

## Run on local after pip install
```
python async_parameter_server.py --iterations 20 --num_workers 2
python sync_parameter_server --iterations 20 --num_workers 2
```
See [here](#Options) for more configurable options for this example.

## Run on yarn cluster for yarn-client mode after pip install 
```
export YARN_CONF=... # path to your hadoop/yarn directory

python async_parameter_server.py --hadoop_conf $YARN_CONF --conda_name ...# your conda name
python sync_parameter_server --hadoop_conf $YARN_CONF --conda_name...# your conda name
```
 
See [here](#Options) for more configurable options for this example.

## Options
- `--object_store_memory` The store memory you need to use on local. Default is 4g.
- `--driver_cores` The number of driver's or local's cpu cores you want to use. Default is 8.
- `--iterations` The number of iterations to train the model. Default in both examples is 50.

**Options only for yarn**
- `--hadoop_conf` This option is **required** when you want to run on yarn. The path to your configuration folder of hadoop.
- `--conda_name` This option is **required** when you want to run on yarn. Your conda environment's name.
- `--num_workers` The number of slave nodes you want to to use. Default is 4.
- `--executor_cores` The number of slave(executor)'s cpu cores you want to use. Default is 8.
- `--executor_memory` The size of slave(executor)'s memory you want to use. Default is 10g.
- `--driver_memory` The size of driver's memory you want to use. Default is 2g.
- `--extra_executor_memory_for_ray` The size of slave(executor)'s extra memory to store data. Default is 20g.

