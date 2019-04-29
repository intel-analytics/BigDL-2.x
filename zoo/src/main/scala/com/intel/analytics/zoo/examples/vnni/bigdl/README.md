# Summary
This folder contains three examples to illustrate the support of [VNNI](https://en.wikichip.org/wiki/x86/avx512vnni) using BigDL MKL-DNN in Analytics Zoo, which aims at accelerating inference by utilizing low numerical precision (Int8) computing. 
Int8 quantized models can give you better performance on Intel Xeon scalable processors.

- You can run the [Predict](#predict) example to do inference on images with the pre-trained model.
- You can run the [ImageNetInference](#imagenetinference) example to evaluate the pre-trained model using no-resize sequence files for images.
- You can run the [Perf](#perf) example to collect performance data for the pre-trained model.

## Download Analytics Zoo and pre-trained model
- You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.
- Download pre-trained int8 quantized ResNet50 model.


## Predict
This example demonstrates how to do image classification with the pre-trained int8 quantized model.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=local[*]
imagePath=the local folder path which contains images to be predicted 
modelPath=the path to the downloaded int8 model

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master ${MASTER} \
    --class com.intel.analytics.zoo.examples.vnni.bigdl.Predict \
    -f ${imagePath} -m ${modelPath}
```

__Options:__


## ImageNetInference
This example evaluates the pre-trained int8 model using Hadoop SequenceFiles for ImageNet no-resize images.

You may refer to this [script](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/utils/ImageNetSeqFileGenerator.scala) to generate sequence files for images.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
imagePath=the folder path which contains sequence files for ImageNet no-resize images.
modelPath=the path to the downloaded int8 model

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master ${MASTER} \
    --class --class com.intel.analytics.zoo.examples.vnni.bigdl.ImageNetInference \
    -f ${imagePath} -m ${modelPath}
```


## Perf
This example runs in local mode and calculates performance data (i.e. throughput and latency) for the pre-trained int8 model.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
export ANALYTICS_ZOO_JAR=export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
modelPath=the path to the downloaded int8 model

java -cp ${ANALYTICS_ZOO_JAR}:${SPARK_HOME}/jars/* com.intel.analytics.zoo.examples.vnni.bigdl.Perf -m ${modelPath} -b 64
```