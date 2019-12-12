# Visualization and Log
This page shows how to see log and visualization in Analytics Zoo Cluster Serving.
## Visualization
We integrate Tensorboard into Cluster Serving, to enable this feature, use following config in [Configuration](). By default, this feature is disabled. By disabling this feature, you could have a slight gain of serving performance since there is some cost to stat the information.
```
log:
  summary: y
```
Tensorboard service is started with Cluster Serving, once your serving is run, you can go to `localhost:6006` to see visualization of your serving.

Analytics Zoo Cluster Serving provides 3 attributes in Tensorboard so far, `Micro Batch Throughput`, `Partition Number`, `Total Records Number`.

* `Micro Batch Throughput`: The overall throughput, including preprocessing and postprocessing of your serving, the line should be relatively stable after first few records. If this number has a drop and remains lower than previous, you might have lost the connection of some nodes in your cluster.

* `Partition Number`: The partition number of your serving, this number should be stable all the time, and note that if you have N nodes in your cluster, you should have this partition number at least N.

* `Total Records Number`: The total number of records that serving gets so far.

**Note**: If you run serving on local mode, you could get another attribute `Throughput`, this is the throughput of prediction only, regardless of preprocessing and post processing. If you run serving on cluster mode, you could only see this attribute on remote nodes.


## Log
We use log to save serving information and error, to enable this feature, use following config in [Configuration](). By default, this feature is enabled.
```
log:
  error: y
```
If you are the only user to run Cluster Serving, the error logs would also print to your interactive shell. Otherwise, you can not see the logs in the terminal. In this ocasion, you have to refer to your log.

To see your log, run 
```
./cluster-serving-log.sh
```
## Example
See [example] here to practise how to utilize summary and log of Analytics Zoo Cluster Serving.
