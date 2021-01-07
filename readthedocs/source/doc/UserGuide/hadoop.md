# Hadoop/YARN User Guide

---

You can run Analytics Zoo programs on standard Hadoop/YARN clusters without any changes to the cluster (i.e., no need to pre-install Analytics Zoo or any Python libraries in the cluster).

### **1. Prepare Python Environment**

1) You need to first use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the Python environment _on the driver node_, and then install all the needed Python libraries (including Analytics Zoo) in the conda environment:

```bash
conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
conda activate zoo

pip install analytics-zoo
# Use conda or pip to install all the needed Python dependencies.
``` 

2) You need to install JDK in the environment. __JDK8__ is highly recommended.
You can download JDK8 and set the environment variable `JAVA_HOME` or install JDK8 via conda:

```bash
conda install -c anaconda openjdk=8.0.152
```

View the [Python User Guide](python.html) for more details.

---
### **2. For YARN client mode**

We recommend using `init_orca_context` in your code to run on standard Hadoop/YARN clusters in [YARN client mode](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn). 

```python
from zoo.orca import init_orca_context

sc = init_orca_context(cluster_mode="yarn-client", ...)
```

By specifying cluster_mode to be "yarn-client", `init_orca_context` would automatically prepare the runtime Python environment and initiate the distributed execution engine on the underlying YARN cluster. 
Then you can simply write and run your application as a normal Python script or in a Jupyter notebook.

View the [Orca Context](../../Orca/Overview/orca-context.html) for more details.

---
### **3. For YARN cluster mode**

If you need to run in [YARN cluster mode](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn), you need to first manually package your current conda environment:

```bash
conda pack -o environment.tar.gz
```

Then in your Python script, you can call `init_orca_context` and specify cluster_mode to be "spark_submit":

```python
from zoo.orca import init_orca_context

sc = init_orca_context(cluster_mode="spark-submit", ...)
```

View the [Orca Context](../../Orca/Overview/orca-context.html) for more details.
