# Multi agent training example
Here we demonstrate how to use Analytics Zoo API to run [mutil-agent](https://github.com/ray-project/ray/blob/master/rllib/examples/multiagent_two_trainers.py)
example provided by [Ray](https://github.com/ray-project/ray).


See [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/) for more details for RayOnSpark support in Analytics Zoo.

## Prepare environments
Follow steps 1 to 4 [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/#steps-to-run-rayonspark) to prepare environment.

You also need to install **TensorFlow** in your conda environment and make sure your pyspark version is above 2.4.0.

Here we require ray above 0.7.2, and if you use ray 0.6.6, you may need to make changes to some APIs accordingly.
```shell script
pip install ray>=0.7.2
pip install tensorflow
```

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install)
for more running guidance after pip install. 

## Run on local after pip install
```
python multiagent_two_trainers.py --iterations 20
```
See [here](#Options) for more configurable options for this example.

## Run on yarn cluster after pip install 
```
python multiagent_two_trainers.py --hadoop_conf ... # path to your hadoop/yarn directory --conda_name ... #your conda name
```
 
See [here](#Options) for more configurable options for this example.

## Options
- `--object_store_memory`The store memory you need to use on local. Default is 4g.
- `--driver_cores` The number of driver's or local's cpu cores you want to use. Default is 8.
- `--iterations` The number of iterations to train the model. Default is 10.

**Options for yarn only**
- `--hadoop_conf` This option is **required** when you want to run on yarn. The path to your configuration folder of hadoop.
- `--conda_name` This option is **required** when you want to run on yarn. Your conda environment's name.
- `--slave_num` The number of slave nodes you want to to use. Default is 2.
- `--executor_cores` The number of slave(executor)'s cpu cores you want to use. Default is 8.
- `--executor_memory` The size of slave(executor)'s memory you want to use. Default is 10g.
- `--driver_memory` The size of driver's memory you want to use. Default is 2g
- `--extra_executor_memory_for_ray` The size of slave(executor)'s extra memory to store data. Default is 20g.
