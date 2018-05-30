For Python users, Analytics Zoo can be installed either from pip or without pip.

**NOTE**: Only __Python 2.7__, __Python 3.5__ and __Python 3.6__ are supported for now.

---
## Install from pip

Analytics Zoo can be installed via pip easily using the following command.

***Install analytics-zoo-0.1.0.dev0***

* Note that you might need to add `sudo` if you don't have the permission for installation.

```bash
pip install --upgrade pip
pip install analytics-zoo==0.1.0.dev0     # for Python 2.7
pip3 install analytics-zoo==0.1.0.dev0    # for Python 3.5 and Python 3.6
```

**Remarks:**

1. Pip install supports __Mac__ and __Linux__ platforms.
2. Pip install only supports __local__ mode. Cluster mode might be supported in the future. For those who want to use Analytics Zoo in cluster mode, please try to [install without pip](#install-without-pip).
3. If you use pip install, it is __not__ necessary to set `SPARK_HOME`.
4. You need to install Java __>= JDK8__ before running Analytics Zoo, which is required by __pyspark__.
5. We've tested this package with pip 9.0.1.
6. `bigdl==0.5.0` and its dependencies (including `pyspark>=2.2`, `numpy>=1.7` and `six>=1.10.0`) will be automatically installed first before installing analytics-zoo if they haven't been detected in the current Python environment.


---
## Install without pip

If you choose to install Analytics Zoo without pip, you need to prepare Spark and the necessary dependencies.

**Steps:**
1. [Download Spark](https://spark.apache.org/downloads.html)

2. You can download the Analytics Zoo release and nightly build from the [Release Page](../release-download.md)
  or build the Analytics Zoo package from [source](../ScalaUserGuide/install-build-src.md).