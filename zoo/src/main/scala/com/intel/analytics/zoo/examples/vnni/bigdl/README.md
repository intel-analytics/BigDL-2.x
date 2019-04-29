# Summary
We hereby illustrate the support of [VNNI](https://en.wikichip.org/wiki/x86/avx512vnni) using BigDL [MKL-DNN](https://github.com/intel/mkl-dnn) as backend in Analytics Zoo, which aims at accelerating inference by utilizing low numerical precision (Int8) computing. 

Int8 quantized models can give you better performance on Intel Xeon scalable processors.

## Download Analytics Zoo and pre-trained model
- You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.
- Download pre-trained int8 quantized ResNet50 model from [here](https://drive.google.com/file/d/1xAXX6wHHMlVZU5TlmFANGFsna83Tbnpk/view?usp=sharing).

## Examples
This folder contains three examples for BigDL VNNI support:
- [Predict](#predict)
- [ImageNetInference](#imagenetinference)
- [Perf](#perf)

__Remarks:__
- If you are using an int8 quantized model pre-trained in Analytics Zoo, you can find the following info in console log:
```
INFO  ImageModel$:133 - Loading an int8 convertible model. Quantize to an int8 model for better performance
```
- You may need to enlarge memory configurations depending on the size of your input data.
- You can set `-Dbigdl.mklNumThreads` if necessary.

### Predict
This example demonstrates how to do image classification with the pre-trained int8 model.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=local[*]
imagePath=the folder path containing images
modelPath=the path to the downloaded int8 model

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master ${MASTER} \
    --class com.intel.analytics.zoo.examples.vnni.bigdl.Predict \
    -f ${imagePath} -m ${modelPath}
```

__Options:__
- `-f` `--folder`: The folder path that contains image data for prediction.
- `-m` `--model`: The path to the downloaded int8 model.
- `--topN`: The top N classes with highest probabilities as output. Default is 5.
- `--partitionNum`: The number of partitions to cut the dataset into. Default is 4.

__Sample console log output__:
```
INFO  Predict$:67 - image : 1.jpg, top 5
INFO  Predict$:71 - 	 class: kelpie, credit: 0.96370447
INFO  Predict$:71 - 	 class: Rottweiler, credit: 0.026292335
INFO  Predict$:71 - 	 class: Eskimo dog, husky, credit: 0.0019479054
INFO  Predict$:71 - 	 class: German shepherd, German shepherd dog, German police dog, alsatian, credit: 0.001165287
INFO  Predict$:71 - 	 class: Doberman, Doberman pinscher, credit: 8.323631E-4
```

### ImageNetInference
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


### Perf
This example runs in local mode and calculates performance data (i.e. throughput and latency) for the pre-trained int8 model.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
export ANALYTICS_ZOO_JAR=export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
modelPath=the path to the downloaded int8 model

java -cp ${ANALYTICS_ZOO_JAR}:${SPARK_HOME}/jars/* com.intel.analytics.zoo.examples.vnni.bigdl.Perf -m ${modelPath} -b 64
```