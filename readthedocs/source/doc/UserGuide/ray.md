# Ray User Guide

---

[Ray](https://github.com/ray-project/ray) is an open source distributed framework for emerging AI applications. Users can seamlessly integrate Ray applications into the big data processing pipeline on the underlying cluster (such as [K8s](./k8s.md) or [Hadoop/YARN](./hadoop.md) cluster) with Analytics Zoo.

_**Note:** Analytics Zoo has been tested on Ray 0.8.4 and you are highly recommended to use this tested version._


### **1. Install**

Follow the guide [here](./python.html#install) for environment preparation. When installing analytics-zoo with pip, you can specify the extras key `[ray]` to additionally install the additional dependencies essential for running Ray (i.e. `ray==0.8.4`, `psutil`, `aiohttp`, `setproctitle`, `pyarrow==0.17.0`):

```bash
pip install analytics-zoo[ray]
```

---
### **2. Initialize**

We recommend using `init_orca_context` to initiate and run Analytics Zoo on the underlying cluster. The Ray cluster would be launched as well by specifying `init_ray_on_spark=True`. For example, to launch Spark and Ray on standard Hadoop/YARN clusters in [YARN client mode](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn):

```python
from zoo.orca import init_orca_context

sc = init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2, init_ray_on_spark=True)
```


---
### **3. Run**

- After the initialization, you can directly run Ray applications on the underlying cluster. The following code shows a simple example:

  ```python
  import ray

  @ray.remote
  class Counter(object):
        def __init__(self):
            self.n = 0
  
        def increment(self):
            self.n += 1
            return self.n


  counters = [Counter.remote() for i in range(5)]
  print(ray.get([c.increment.remote() for c in counters]))
  ```

  In the code above, the [Ray actors](https://docs.ray.io/en/master/actors.html) would be created and located across the underlying cluster.

- You can retrieve the Ray cluster information via [`OrcaContext`](../Orca/Overview/orca-context.md):

  ```python
  from zoo.orca import OrcaContext
  
  ray_ctx = OrcaContext.get_ray_context()
  address_info = ray_ctx.address_info  # The dictionary information of the ray cluster, including node_ip_address, object_store_address, webui_url, etc.
  redis_address = ray_ctx.redis_address  # The redis address of the ray cluster.
  ```


---
### **4. FAQ**
If you encounter the following error when launching Ray on the underlying cluster, especially when you are using a Spark standalone cluster:

```
This system supports the C.UTF-8 locale which is recommended. You might be able to resolve your issue by exporting the following environment variables:

    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
```

Add the environment variables when calling `init_orca_context` would resolve the issue:

```python
sc = init_orca_context(cluster_mode, init_ray_on_spark=True, env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
```
