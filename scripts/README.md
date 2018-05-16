# How to Use Scripts
## Set Environment Variables

Before using scripts, two environment variables should be set.

If you download Zoo from the Release Page
```bash
export SPARK_HOME=directory path where you extract the spark package
export ANALYTICS_ZOO_HOME=directory path where you extract the zoo package
```

If you build Zoo by yourself
```bash
export SPARK_HOME=directory path where you extract the spark package
export ANALYTICS_ZOO_HOME=dist directory
```
The ```dist``` directory can be found in following hierarchy after you build the zoo.
```
zoo 
 |---backend      (directory)
 |---data         (directory)
 |---dist         *(directory)*
       |---conf   (directory)
       |---lib    (directory)
 |---docs         (directory)
 |---notebook     (directory)
 |---pyzoo        (directory)
 |---scripts      (directory)
 |---zoo          (directory)
 |---LICENSE      (file)
 |---README.md    (file)
 |---make-dist.sh (file)
 |---pom.xml      (file)
```
## Run Scripts
After setting necessary environment variables, you can run those scripts. One example is shown as following.
```bash
./spark-submit-with-zoo.sh \
    --master your_master_of_spark \
    --driver-cores cores_number_of_driver  \
    --driver-memory memory_size_of_driver  \
    --total-executor-cores total_cores_number_of_executor  \
    --executor-cores cores_number_of_executor  \
    --executor-memory memory_size_of_executor \
    path_to_your_python_script_of_model
```
Note that not all parameters are required. You only need to set necessary ones.
