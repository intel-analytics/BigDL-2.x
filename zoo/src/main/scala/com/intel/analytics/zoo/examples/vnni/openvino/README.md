# Analytics Zoo OpenVINO Int8 ResNet_v1_50 Example

## Summary
We hereby illustrate the support of [VNNI](https://en.wikichip.org/wiki/x86/avx512vnni) using [OpenVINO](https://software.intel.com/en-us/openvino-toolkit) as backend in Analytics Zoo, which aims at accelerating inference by utilizing low numerical precision (Int8) computing. Int8 quantized models can generally give you better performance on Intel Xeon scalable processors.
 
## Environment
* Apache Spark (This version needs to be same with the version you use to build Analytics Zoo)
* [Analytics Zoo](https://analytics-zoo.github.io/master/#ScalaUserGuide/install/)

## Datasets and pre-trained models
* Datasets: [ImageNet2012 Val](http://image-net.org/challenges/LSVRC/2012/index)
* Pre-trained model: [ResNet50 for OpenVINO]()
* Int8 Optimized model: [ResNet50 Int8 for OpenVINO]()


## Examples
This folder contains four examples for OpenVINO VNNI support:
- [Perf](#perf)
- [VINOPerf](#vinoperf)
- [ImageNetEvaluation](#imagenetevaluation)
- [Predict](#predict)

Environment Setting:
- Set `ZOO_NUM_THREADS` to determine cores used by OpenVINO, e.g, `export ZOO_NUM_THREADS=10`. If it is set to `all`, e.g., `export ZOO_NUM_THREADS=all`, then OpenVINO will utilize all physical cores for Prediction. 

---
### Perf
This example runs in local mode and calculates performance data (i.e. throughput and latency) for the pre-trained int8 model using dummy input.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
export ANALYTICS_ZOO_JAR=export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`

MASTER=...
modelPath=path of the downloaded int8 model
weightPath=path of the downloaded int8 model weight

{ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh  \
    --master ${MASTER} --driver-memory 2g \
    --class com.intel.analytics.zoo.examples.vnni.openvino.Perf \
    -m ${modelPath} -w ${weightPath}
```

__Options:__
- `-m` `--model`: The path to the downloaded int8 model.
- `-w` `--weight`: The path to the downloaded int8 model weight.
- `-b` `--batchSize`: The batch size of input data. Default is 4.
- `-i` `--iteration`: The number of iterations to run the performance test. Default is 1.

__Sample console log output__:
```
2019-05-17 10:19:33 INFO  InferenceSupportive$:45 - model predict for activity time elapsed [0 s, 5 ms].
2019-05-17 10:19:33 INFO  Perf$:101 - Iteration 1 latency is 5.433454 ms
[ INFO ] Start inference (1 iterations)

Total inference time: 2.47494 ms
Average running time of one iteration: 2.47494 ms
Throughput: 1616.2 FPS

JNI total predict time: 3.94415 ms
JNI Throughput: 1014.16 FPS
```

---
### VINOPerf
VINOPerf is a dependency-reduced Perf based on [Inference Model](https://analytics-zoo.github.io/0.4.0/#ProgrammingGuide/inference/#inference-model). It runs locally and calculates performance data (i.e. throughput and latency) for the pre-trained int8 model using dummy input.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
export ANALYTICS_ZOO_JAR=export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`

modelPath=path of the downloaded int8 model
weightPath=path of the downloaded int8 model weight

java -cp ${ANALYTICS_ZOO_JAR}:${SPARK_HOME}/jars/* \
    com.intel.analytics.zoo.examples.vnni.openvino.VINOPerf \
    -m ${modelPath} -w ${weightPath}
```

__Options:__
- `-m` `--model`: The path to the downloaded int8 model.
- `-w` `--weight`: The path to the downloaded int8 model weight.
- `-b` `--batchSize`: The batch size of input data. Default is 4.
- `-i` `--iteration`: The number of iterations to run the performance test. Default is 1.

__Sample console log output__:
```
2019-05-17 10:00:30 INFO  InferenceSupportive$:45 - model predict for batch 1 time elapsed [0 s, 5 ms].
2019-05-17 10:00:30 INFO  VINOPerf$:84 - Iteration 1 latency is 4.993594 ms
[ INFO ] Start inference (1 iterations)

Total inference time: 2.40128 ms
Average running time of one iteration: 2.40128 ms
Throughput: 1665.78 FPS

JNI total predict time: 4.0567 ms
JNI Throughput: 986.023 FPS
```

---
### ImageNetEvaluation
This example evaluates the pre-trained int8 model using Hadoop SequenceFiles of ImageNet no-resize validation images.

You may refer to this [script](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/utils/ImageNetSeqFileGenerator.scala) to generate sequence files for images.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
imagePath=folder path that contains sequence files of ImageNet no-resize validation images.
modelPath=the path of downloaded int8 model
weightPath=the path of downloaded int8 model weight

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master ${MASTER} \
    --class com.intel.analytics.zoo.examples.vnni.openvino.ImageNetEvaluation \
    -f ${imagePath} -m ${modelPath} -w {weightPath}
```

__Options:__
- `-m` `--model`: The path to the downloaded int8 model.
- `-w` `--weight`: The path to the downloaded int8 model weight.
- `-b` `--batchSize`: The batch size of input data. Default is 4.
- `-f` `--folder`: The folder path that contains sequence files of ImageNet no-resize validation images.
- `--partitionNum`: The partition number of the dataset. Default is 32.

__Sample console log output__:
```
Evaluation Results:
Top1Accuracy is Accuracy(correct: 36432, count: 50000, accuracy: 0.72864)
Top5Accuracy is Accuracy(correct: 45589, count: 50000, accuracy: 0.91178)
```
Note that: int8 model's accuracy is a bit lower than normal model, due to int8 related optimization.

---
### Predict
This example runs in local mode and demonstrates how to do image classification with the pre-trained int8 model. Note that current example uses local mode to evaluate throughput and latency. If you want to run prediction on Spark cluster, please refer to ImageNetEvaluation or change `toLocal` to `toDistributed`.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
imagePath=folder path that contains image files of ImageNet
modelPath=the path of downloaded int8 model
weightPath=the path of downloaded int8 model weight

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master ${MASTER} \
    --class com.intel.analytics.zoo.examples.vnni.openvino.Predict \
    -f ${imagePath} -m ${modelPath} -w {weightPath}
```

__Options:__
- `-m` `--model`: The path to the downloaded int8 model.
- `-w` `--weight`: The path to the downloaded int8 model weight.
- `-b` `--batchSize`: The batch size of input data. Default is 4.
- `-f` `--folder`: The folder path that contains sequence files of ImageNet no-resize validation images.

__Sample console log output__:
```
INFO  Predict$:129 - image : 1.jpg, top 5
Predict$:129 - 	 class: collie, credit: 0.0024525642
Predict$:129 - 	 class: Shetland sheepdog, Shetland sheep dog, Shetland, credit: 0.0010984923
Predict$:129 - 	 class: borzoi, Russian wolfhound, credit: 0.0010027748
Predict$:129 - 	 class: groenendael, credit: 9.985587E-4
Predict$:129 - 	 class: briard, credit: 9.98535E-4
```
