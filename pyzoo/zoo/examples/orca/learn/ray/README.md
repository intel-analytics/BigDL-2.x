# Orca RayOnSpark example

[Ray](https://github.com/ray-project/ray) is an open source distributed framework for emerging AI applications. With the _**RayOnSpark**_ support in Analytics Zoo, Users can seamlessly integrate Ray applications into the big data processing pipeline on the underlying Big Data cluster (such as [Hadoop/YARN](../../../../../../readthedocs/source/doc/UserGuide/hadoop.md) or [K8s](../../../../../../readthedocs/source/doc/UserGuide/k8s.md)).

## Prepare environments
We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the Python environment. 
When installing analytics-zoo with pip, you can specify the extras key `[ray]` to additionally install the additional dependencies essential for running Ray (i.e. `ray==0.8.4`, `psutil`, `aiohttp`, `setproctitle`, `pyarrow==0.17.0`):

```bash
conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
conda activate zoo

pip install analytics-zoo[ray]
```

## Run example
You can run this example on local mode and yarn client mode. Note that this example requires at least 10G of free memory, please check your hardware.

- Run with Spark Local mode:
```bash
python raytest.py --cluster_mode local
```

- Run with Yarn Client mode:
```bash
python raytest.py --cluster_mode yarn
```

In above commands
* `--cluster_mode` The mode of spark cluster, supporting local and yarn. Default is "local".


## Results
You can find the logs as the result of ray remote method:
```
[1, 1, 1, 1, 1]
```
