# Databricks User Guide

---

You can run Analytics Zoo program on [Databricks](https://databricks.com/) Spark cluster.

### **1. Create a Databricks Spark cluster**

You can create either [AWS Databricks](https://docs.databricks.com/getting-started/try-databricks.html) workspace or [Azure Databricks](https://docs.microsoft.com/en-us/azure/azure-databricks/) workspace. Then create a Databricks Spark [clusters](https://docs.databricks.com/clusters/create.html) using the UI. Choose Databricks runtime version. This guide is tested on Runtime 5.5 LTS (includes Apache Spark 2.4.3, Scala 2.11).

### **2. Installing Analytics Zoo libraries**

1. In the left pane, click **Clusters** and select your cluster.

![Pic1](./images/Databricks1.PNG)

2. Install Analytics Zoo python environment using PyPI. Click **Libraries > Install New > PyPI**. Install official released version by texting "analytics-zoo" library. Or install latest nightly build of [Analytics Zoo](https://pypi.org/project/analytics-zoo/#history) with the specified version.

![Pic2](./images/Databricks2.PNG)

3. Install Analytics Zoo prebuilt jar package. Click **Libraries > Install New > Upload > Jar**. Download Analytics Zoo prebuilt package from this [Release Page](../release.md). Please note that you should choose the same spark version of package as your Databricks runtime version. Unzip it. Find jar named "analytics-zoo-bigdl_*-spark_*-jar-with-dependencies.jar" in the lib directory. Drop the jar on Databricks.

![Pic3](./images/Databricks3.PNG)

4. Make sure the jar file and analytics-zoo (with PyPI) are installed on all clusters. In **Libraries** tab of your cluster, check installed libraries and click “Install automatically on all clusters” option in **Admin Settings**.

![Pic4](./images/Databricks4.PNG)

### **3. Setting Spark configuration**

On the cluster configuration page, click the **Advanced Options** toggle. Click the **Spark** tab. You can provide custom [Spark configuration properties](https://spark.apache.org/docs/latest/configuration.html) in a cluster configuration. Please set it according to your cluster resource and program needs.

![Pic5](./images/Databricks5.PNG)

See below for an example of Spark config setting needed by Analytics Zoo. Here it sets 1 core and 6g memory per executor and driver. Note that "spark.cores.max" needs to be properly set below.

```
spark.shuffle.reduceLocality.enabled false
spark.serializer org.apache.spark.serializer.JavaSerializer
spark.shuffle.blockTransferService nio
spark.databricks.delta.preview.enabled true
spark.executor.cores 1
spark.executor.memory 6g
spark.speculation false
spark.driver.memory 6g
spark.scheduler.minRegisteredResourcesRatio 1.0
spark.cores.max 4
spark.driver.cores 1
```

### **4. Running Analytics Zoo on Databricks**

Open a new notebook. First call `init_nncontext()` at the beginning of your code. This will create a SparkContext with optimized performance configuration and initialize the BigDL engine.

```python
from zoo.orca import init_orca_context, stop_orca_context
init_orca_context(cluster_mode="spark-submit")
```

Output on Databricks:
![Pic6](./images/Databricks6.PNG)
