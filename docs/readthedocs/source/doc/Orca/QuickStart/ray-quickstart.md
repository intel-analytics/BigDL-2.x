# Ray Quickstart

---

![](../../../../image/colab_logo_32px.png)[Run in Google Colab](https://colab.research.google.com/github/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/ray_sharded_parameter_server.ipynb) &nbsp;![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/ray_sharded_parameter_server.ipynb)

---

**In this guide we will describe how to run Ray applications using the RayOnSpark support in Analytics Zoo in 2 simple steps.**

### **Step 0: Prepare Environment**

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment. Please refer to the [install guide](../../UserGuide/python.md) for more details.

```bash
conda create -n zoo python=3.7 # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install analytics-zoo[ray]
```

### **Step 1: Initialize**

We recommend using `init_orca_context` to initiate and run Analytics Zoo on the underlying cluster. The Ray cluster would be launched as well by specifying `init_ray_on_spark=True`.

```python
if args.cluster_mode == "local":
    sc = init_orca_context(cluster_mode="local", cores=4, init_ray_on_spark=True)# run in local mode
elif args.cluster_mode == "k8s":
    sc = init_orca_context(cluster_mode="k8s", num_nodes=2, cores=2, init_ray_on_spark=True) # run on K8s cluster
elif args.cluster_mode == "yarn":
    sc = init_orca_context(cluster_mode="yarn-client", cores=2, memory="10g", num_nodes=2, init_ray_on_spark=True) # run on Hadoop YARN cluster
```

**Note:** You should `export HADOOP_CONF_DIR=/path/to/hadoop/conf/dir` when running on Hadoop YARN cluster. View [Hadoop User Guide](./../../UserGuide/hadoop.md) for more details.

You can retrieve the information of the Ray cluster via `OrcaContext`:

```python
ray_ctx = OrcaContext.get_ray_context()
address_info = ray_ctx.address_info  # The dictionary information of the ray cluster, including node_ip_address, object_store_address, webui_url, etc.
redis_address = ray_ctx.redis_address  # The redis address of the ray cluster.
```

### **Step 2: Run Ray Applications**

After the initialization, you can directly run Ray applications on the underlying cluster. Ray [tasks](https://docs.ray.io/en/master/walkthrough.html#remote-functions-tasks) and [actors](https://docs.ray.io/en/master/actors.html) would be launched across the cluster.

The following example uses actor handles to implement a sharded parameter server example for distributed asynchronous stochastic gradient descent. 

#### One parameter server

A parameter server is simply an object that stores the parameters (or "weights") of a machine learning model (this could be a neural network, a linear model, or something else). It exposes two methods: one for getting the parameters and one for updating the parameters.

By adding the `@ray.remote` decorator, the `ParameterServer` class becomes a Ray actor.

```python
import numpy as np
dim = 10
@ray.remote
class ParameterServer(object):
    def __init__(self, dim):
        self.parameters = np.zeros(dim)
    
    def get_parameters(self):
        return self.parameters
    
    def update_parameters(self, update):
        self.parameters += update

ps = ParameterServer.remote(dim)
```

In a typical machine learning training application, worker processes will run in an infinite loop that does the following:

1. Get the latest parameters from the parameter server.
2. Compute an update to the parameters (using the current parameters and some data).
3. Send the update to the parameter server.

By adding the `@ray.remote` decorator, the `worker` function becomes a Ray remote function.

```python
@ray.remote
def worker(ps, dim, num_iters):
    for _ in range(num_iters):
        # Get the latest parameters.
        parameters = ray.get(ps.get_parameters.remote())
        # Compute an update.
        update = 1e-3 * parameters + np.ones(dim)
        # Update the parameters.
        ps.update_parameters.remote(update)
        # Sleep a little to simulate a real workload.
        time.sleep(0.5)

# Test that worker is implemented correctly. You do not need to change this line.
ray.get(worker.remote(ps, dim, 1))

# Start two workers.
worker_results = [worker.remote(ps, dim, 100) for _ in range(2)]
```

As the worker tasks are executing, you can query the parameter server from the driver and see the parameters changing in the background.

```
print(ray.get(ps.get_parameters.remote()))
```

#### Sharded parameter server

As the number of workers increases, the volume of updates being sent to the parameter server will increase. At some point, the network bandwidth into the parameter server machine or the computation down by the parameter server may be a bottleneck. It is better to shard the parameters across several parameter servers in this case.

```python
@ray.remote
class ParameterServerShard(object):
    def __init__(self, sharded_dim):
        self.parameters = np.zeros(sharded_dim)
    
    def get_parameters(self):
        return self.parameters
    
    def update_parameters(self, update):
        self.parameters += update


total_dim = (10 ** 8) // 8  # This works out to 100MB (we have 25 million
                            # float64 values, which are each 8 bytes).
num_shards = 2  # The number of parameter server shards.

assert total_dim % num_shards == 0, ('In this exercise, the number of shards must '
                                     'perfectly divide the total dimension.')

# Start some parameter servers.
ps_shards = [ParameterServerShard.remote(total_dim // num_shards) for _ in range(num_shards)]

assert hasattr(ParameterServerShard, 'remote'), ('You need to turn ParameterServerShard into an '
                                                 'actor (by using the ray.remote keyword).')
```

The code below implements a worker that does the following.

1. Gets the latest parameters from all of the parameter server shards.
2. Concatenates the parameters together to form the full parameter vector.
3. Computes an update to the parameters.
4. Partitions the update into one piece for each parameter server.
5. Applies the right update to each parameter server shard.

```python
@ray.remote
def worker_task(total_dim, num_iters, *ps_shards):
    # Note that ps_shards are passed in using Python's variable number
    # of arguments feature. We do this because currently actor handles
    # cannot be passed to tasks inside of lists or other objects.
    for _ in range(num_iters):
        # Get the current parameters from each parameter server.
        parameter_shards = [ray.get(ps.get_parameters.remote()) for ps in ps_shards]
        assert all([isinstance(shard, np.ndarray) for shard in parameter_shards]), (
               'The parameter shards must be numpy arrays. Did you forget to call ray.get?')
        # Concatenate them to form the full parameter vector.
        parameters = np.concatenate(parameter_shards)
        assert parameters.shape == (total_dim,)

        # Compute an update.
        update = np.ones(total_dim)
        # Shard the update.
        update_shards = np.split(update, len(ps_shards))
        
        # Apply the updates to the relevant parameter server shards.
        for ps, update_shard in zip(ps_shards, update_shards):
            ps.update_parameters.remote(update_shard)


# Test that worker_task is implemented correctly. You do not need to change this line.
ray.get(worker_task.remote(total_dim, 1, *ps_shards))
```

When all worker processes and server processes are running on the same machine, network bandwidth will not be a limitation and sharding the parameter server will not help. You can experiment it by changing the number of parameter server shards, the number of workers, and the size of the data.

```python
import time
num_workers = 4

# Start some workers. Try changing various quantities and see how the
# duration changes.
start = time.time()
ray.get([worker_task.remote(total_dim, 5, *ps_shards) for _ in range(num_workers)])
print('This took {} seconds.'.format(time.time() - start))
```

To see the difference, you would need to run the application on multiple machines. There are still regimes where sharding a parameter server can help speed up computation on the same machine (by parallelizing the computation that the parameter server processes have to do). If you want to see this effect, you should implement a synchronous training application. In the asynchronous setting, the computation is staggered and so speeding up the parameter server usually does not matter.

**Note:** You should call `stop_orca_context()` when your program finishes:

```python
stop_orca_context()
```
