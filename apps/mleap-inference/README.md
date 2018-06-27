# Overview

This is a Scala example for low-latency inference with Spark ML pipeline and Analytics-Zoo. We
use MLeap to run inference with a trained BigDL model, which helps 
reduce the overhead of Spark Context and DataFrame, and achieves low latency execution (1e-4 
second) for production deployment of Spark Pipeline Model.

# Image Model Inference

You can run the full ModelInference example by following steps.

1. In this direcotry (mleap-inference), run
 
    `mvn clean package`

2. Run this example

Command to run the example in Spark local mode:
```
$SPARK_HOME/bin/spark-submit \
--class com.intel.analytics.zoo.apps.mleap.NNClassifierTest \
./target/MLeapTest-1.0.1-SNAPSHOT-jar-with-dependencies.jar
```

expected output:
```
Spark DataFrame transform benchmark:
 1 time: 0.152900738
 10 time: 0.634077539
 100 time: 4.921373392

leap frame transform result:
Row(WrappedArray(4.7, 3.2, 1.6, 0.2),1.0,1.0)
Row(WrappedArray(4.8, 3.4, 1.9, 0.2),1.0,1.0)
Row(WrappedArray(4.9, 2.4, 3.3, 1.0),2.0,2.0)
Row(WrappedArray(4.9, 3.1, 1.5, 0.1),1.0,1.0)

MLeap Frame transform benchmark:
 1 time: 0.001134866
 10 time: 0.00678356
 100 time: 0.027279505
 ```
 