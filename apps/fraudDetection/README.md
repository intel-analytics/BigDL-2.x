BigDL has been used in the Fraud Detection system for one of the top payment companies. As a highly simplified
demo, this notebook uses the public data set to build a fraud detection example on Apache Spark.

The notebook is developed using Scala with Apache Spark 2.1 and BigDL 0.4. Refer to
https://github.com/intel-analytics/analytics-zoo/tree/master/pipeline/fraudDetection for the extra feature transformers.

How to run the notebook:

1. Refer to [Apache Toree](https://github.com/apache/incubator-toree/blob/master/README.md) for
how to use scala in Jupyter notebook.

An outline is:
```bash
pip install https://dist.apache.org/repos/dist/dev/incubator/toree/0.2.0/snapshots/dev1/toree-pip/toree-0.2.0.dev1.tar.gz
```

2. Build BigDL jar file or download the pre-built version from https://bigdl-project.github.io/master/#release-download/ 

3. To support the training for imbalanced data set in fraud detection, some Transformers and algorithms are developed in source folder,
https://github.com/intel-analytics/analytics-zoo/tree/master/pipeline/fraudDetection. We provided a pre-built jar file in this folder. Feel
free to go to the source folder and run "mvn clean package" to build from source.

4. Start the notebook.

```
SPARK_OPTS='--master=local[1] --jars /path/to/bigdl/jar/file,/.../analytics-zoo/pipeline/fraudDetection/fraud.jar' TOREE_OPTS='--nosparkcontext' jupyter notebook
```
