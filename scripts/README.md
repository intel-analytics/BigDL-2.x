# How to Use Scripts
## Set Environment Variables

Before using scripts, two environment variables should be set.

* If you download Analytics Zoo from the [Release Page](../docs/docs/release-download.md):
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
```

* If you build Analytics Zoo by yourself:
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the dist directory of Analytics Zoo
```

The ```dist``` directory can be found in following hierarchy after you build Analytics Zoo.

```
analytics-zoo 
 |---apps                   (directory)
 |---backend                (directory)
 |---dist                  *(directory)*
       |---bin              (directory)
       |---conf             (directory)
       |---lib              (directory)
       |---extra-resources  (directory)
 |---docs                   (directory)
 |---pyzoo                  (directory)
 |---scripts                (directory)
 |---zoo                    (directory)
 |---LICENSE                (file)
 |---README.md              (file)
 |---make-dist.sh           (file)
 |---pom.xml                (file)
```

## Run Scripts
After setting the necessary environment variables above, you can run those scripts. One example is shown as following.
```bash
spark-submit-python-with-zoo.sh \
    --master your_master_of_spark \
    --driver-cores cores_number_of_driver  \
    --driver-memory memory_size_of_driver  \
    --total-executor-cores total_cores_number_of_executor  \
    --executor-cores cores_number_of_executor  \
    --executor-memory memory_size_of_executor \
    path_to_your_python_script_of_model
```
Note that not all the parameters are required. You only need to set the necessary ones.

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/#for-yarn-cluster) for instructions if you want to run on yarn cluster.