# Analytics Zoo OpenVINO Int8 ResNet_v1_50 Example

## Summary
We hereby illustrate the support of [VNNI](https://en.wikichip.org/wiki/x86/avx512vnni) using [OpenVINO](https://software.intel.com/en-us/openvino-toolkit) as backend in Analytics Zoo, which aims at accelerating inference by utilizing low numerical precision (Int8) computing. Int8 quantized models can generally give you better performance on Intel Xeon scalable processors.
 
## Environment
* Apache Spark (This version needs to be same with the version you use to build Analytics Zoo)
* [Analytics Zoo](https://analytics-zoo.github.io/master/#ScalaUserGuide/install/)

Environment Setting:
- Set `ZOO_NUM_MKLTHREADS` to determine cores used by OpenVINO, e.g, `export ZOO_NUM_MKLTHREADS=10`. If it is set to `all`, e.g., `export ZOO_NUM_MKLTHREADS=all`, then OpenVINO will utilize all physical cores for Prediction.
- Set `KMP_BLOCKTIME=200`, i.e., `export KMP_BLOCKTIME=200`

## Datasets and pre-trained models
* Datasets: [ImageNet2012 Val](http://image-net.org/challenges/LSVRC/2012/index)
* Pre-trained model: [TensorFlow ResNet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)
* Optimized Pre-trained model with [OpenVINO](#prepare-openvino-model)

Note that all paths used in this example should be full path, e.g., `/root/model`.

---
### Prepare OpenVINO Model
TensorFlow models cannot be directly loaded by OpenVINO. It should be converted to OpenVINO optimized model and int8 optimized model first. You can use [OpenVINO toolkit](https://docs.openvinotoolkit.org/2020.2/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) to finish this job. With the help of [OpenVINO Open Model Zoo](https://github.com/opencv/open_model_zoo), you can download and optimize Tensorflow models into OpenVINO model, then further convert them into OpenVINO int8 models.

__Sample Result files in MODEL_PATH__:
```
resnet_v1_50.xml
resnet_v1_50.bin
resnet_v1_50_i8.xml
resnet_v1_50_i8.bin
```

Among them, `resnet_v1_50.xml` and `resnet_v1_50.bin` are OpenVINO optimized ResNet_v1_50 model and weight, `resnet_v1_50_i8.xml` and `resnet_v1_50_i8.bin` are OpenVINO int8 optimized ResNet_v1_50 model and weight. Both of them can be loaded by OpenVINO or Zoo.

__Note that int8 optimized model promises better performance (~2X) with slightly lower accuracy. When using int8 optimized model in `Perf` `ImageNetEvaluation` and `Predict`.__


## Examples
This folder contains 3 examples for OpenVINO VNNI support:
- [Perf](#perf)
- [ImageNetEvaluation](#imagenetevaluation)
- [Predict](#predict)

---
### Perf
This example runs in local mode and calculates performance data (i.e. throughput and latency) for the pre-trained int8 model using dummy input.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`

MASTER=...
modelPath=path of OpenVINO optimized model or int8 optimized model
weightPath=path of OpenVINO optimized model weight or int8 optimized model weight

# Pure JAVA
java -cp ${ANALYTICS_ZOO_JAR}:${SPARK_HOME}/jars/* \
    com.intel.analytics.zoo.examples.vnni.openvino.Perf \
    -m ${modelPath} -w ${weightPath}

# On Spark
${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh  \
    --master ${MASTER} --driver-memory 4g \
    --class com.intel.analytics.zoo.examples.vnni.openvino.Perf \
    -m ${modelPath} -w ${weightPath} --onSpark
```

__Options:__
- `-m` `--model`: The path to OpenVINO optimized model or int8 optimized model.
- `-w` `--weight`: The path to OpenVINO optimized model weight or int8 optimized model weight.
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
### ImageNetEvaluation
This example evaluates the pre-trained int8 model using Hadoop SequenceFiles of ImageNet no-resize validation images.

You may refer to this [script](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/utils/ImageNetSeqFileGenerator.scala) to generate sequence files for images.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
imagePath=folder path that contains sequence files of ImageNet no-resize validation images.
modelPath=the path of OpenVINO optimized model or int8 optimized model
weightPath=the path of OpenVINO optimized model weight or int8 optimized model weight

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master ${MASTER} --driver-memory 100g \
    --class com.intel.analytics.zoo.examples.vnni.openvino.ImageNetEvaluation \
    -f ${imagePath} -m ${modelPath} -w ${weightPath}
```

__Options:__
- `-m` `--model`: The path to OpenVINO optimized model or int8 optimized model.
- `-w` `--weight`: The path to OpenVINO optimized model weight or int8 optimized model weight.
- `-b` `--batchSize`: The batch size of input data. Default is 4.
- `-f` `--folder`: The folder path that contains sequence files of ImageNet no-resize validation images.
- `--partitionNum`: The partition number of the dataset. Default is 32.

__Sample console log output__:
```
Evaluation Results:
Top1Accuracy is Accuracy(correct: 36432, count: 50000, accuracy: 0.72864)
Top5Accuracy is Accuracy(correct: 45589, count: 50000, accuracy: 0.91178)
```

---
### Predict
This example runs in local mode and demonstrates how to do image classification with the pre-trained int8 model. Note that current example uses local mode to evaluate throughput and latency. If you want to run prediction on Spark cluster, please refer to ImageNetEvaluation or change `toLocal` to `toDistributed`.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
imagePath=folder path that contains image files of ImageNet
modelPath=the path of OpenVINO optimized model or int8 optimized model
weightPath=the path of OpenVINO optimized model weight or int8 optimized model weight

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master ${MASTER} --driver-memory 10g \
    --class com.intel.analytics.zoo.examples.vnni.openvino.Predict \
    -f ${imagePath} -m ${modelPath} -w ${weightPath}
```

__Options:__
- `-m` `--model`: The path to OpenVINO optimized model or int8 optimized model.
- `-w` `--weight`: The path to OpenVINO optimized model weight or int8 optimized model weight.
- `-b` `--batchSize`: The batch size of input data. Default is 4.
- `-f` `--folder`: The folder path that contains sequence files of ImageNet no-resize validation images.

__Sample console log output__:
```
INFO  Predict$:129 - image : 1.jpg, top 5
Predict$:129 - 	 class: sea snake, credit: 0.8946274
Predict$:129 - 	 class: water snake, credit: 0.077680744
Predict$:129 - 	 class: hognose snake, puff adder, sand viper, credit: 0.008171927
Predict$:129 - 	 class: rock python, rock snake, Python sebae, credit: 0.006575753
Predict$:129 - 	 class: Indian cobra, Naja naja, credit: 0.0048251627
```
