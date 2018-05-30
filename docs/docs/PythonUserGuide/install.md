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
6. `bigdl==0.5.0` and its dependencies will be automatically installed first before installing analytics-zoo if they haven't been detected in the current Python environment.


---
## Install without pip

If you choose to install Analytics Zoo without pip, you need to prepare Spark and the necessary dependencies.

**Steps:**
1. [Download Spark](https://spark.apache.org/downloads.html)

- Note that __Python 3.6__ is only compatible with Spark 1.6.4, 2.0.3, 2.1.1, 2.2.0 and onwards. See [this issue](https://issues.apache.org/jira/browse/SPARK-19019) for more discussion.


2. You can download the Analytics Zoo release and nightly build from the [Release Page](../release-download.md)
  or build the Analytics Zoo package from [source](../ScalaUserGuide/install.md/#build-with-script-recommended).

3. Install Python dependencies. Analytics Zoo only depends on `numpy` and `six` for now.

*For Spark __standalone__ cluster*:

* __Remark__: If you're running in cluster mode, you need to install Python dependencies on both client and each worker node.
* Install numpy: 
```sudo apt-get install python-numpy``` (Ubuntu)
* Install six: 
```sudo apt-get install python-six``` (Ubuntu)

*For __Yarn__ cluster*:

You can run Analytics Zoo Python programs on Yarn clusters without changes to the cluster (i.e., no need to pre-install the Python dependencies).

You can first package all the required Python dependencies into a virtual environment on the local node (where you will run the spark-submit command),
and then directly use spark-submit to run the Analytics Zoo Python program on the Yarn cluster (using that virtual environment). Please follow the steps below: 
   
* Make sure you already installed such libraries(python-setuptools, python-dev, gcc, make, zip, pip) for creating virtual environment. If not, please install them first.
For example, on Ubuntu, run these commands to install:
```
apt-get update
apt-get install -y python-setuptools python-dev
apt-get install -y gcc make
apt-get install -y zip
easy_install pip
```
* Create dependency virtualenv package.
    * Under $ANALYTICS_ZOO_HOME (the dist directory under the Analytics Zoo project), you can find ```bin/python_package.sh```. Run this script to create dependency virtual environment according to dependency descriptions in `requirements.txt`. You can add your own dependencies in `requirements.txt`. The current requirements only contain the dependencies for running Analytics Zoo python examples and models.

    * After running this script, there will be `venv.zip` and `venv` directory generated in current directory. You can use them to submit your python jobs. Please refer to [example](run-without-pip.md#yarn.example) script of submitting bigdl python job with virtual environment in Yarn cluster.