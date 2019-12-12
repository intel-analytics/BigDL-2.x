# Quick Start
This page provides a quick start example for you to run Analytics Zoo Cluster Serving.

## Simplest End-to-end Example 
We use default `config.yaml` configuration, for more details, see [Configuration]()
```
model:
  # model path must be set
  path: /opt/work/model/
data:
  # default, localhost:6379
  src:
  # default, 3, 224, 224
  shape:
params:
  # default, 4
  batch_size:
  # default, mklblas
  engine_type:
  # default, 1
  top_n:
log: 
  # default, y
  error:
  # default, n
  summary:
spark:
  # default, local[*]
  master:
  # default, 4g
  driver_memory:
  # default, 1g
  executor_memory:
  # default, 1
  num_executors:
  # default, 4
  executor_cores:
  # default, 4
  total_executor_cores:
```

