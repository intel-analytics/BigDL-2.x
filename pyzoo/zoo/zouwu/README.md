# Project Zouwu - Telco Solution on Analytis Zoo


## Requirements
* PySpark verison 2.4.3
* Python 3.6 or 3.7
* Keras verison 1.2.2
* Tensorflow version 1.15.0

## Install 
  * Download the Analytics Zoo source code (master). 
  * Build a local .whl wih for Spark2.4.3 using command below. For detailed explaination of the options for ```bulid.sh``` script, refer to [AnalytisZoo Python Developer's guide](https://analytics-zoo.github.io/master/#DeveloperGuide/python/#build-whl-package-for-pip-install)
```bash
bash analytics-zoo/pyzoo/dev/build.sh linux default -Dspark.version=2.4.3 -Dbigdl.artifactId=bigdl-SPARK_2.4 -P spark_2.4+
```
  * The succesfully built whl locates in directory ```analytics-zoo/pyzoo/dist/```. Install the local .whl using below command. 
```
pip install analytics-zoo/pyzoo/dist/analytics_zoo-VERSION-py2.py3-none-PLATFORM_x86_64.whl
```

## Usage

