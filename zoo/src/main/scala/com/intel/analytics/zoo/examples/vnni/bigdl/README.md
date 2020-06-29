# Summary
We hereby illustrate the support of [VNNI](https://en.wikichip.org/wiki/x86/avx512vnni) using BigDL [MKL-DNN](https://github.com/intel/mkl-dnn) as backend in Analytics Zoo, which aims at accelerating inference by utilizing low numerical precision (Int8) computing. 

Int8 quantized models can generally give you better performance on Intel Xeon scalable processors.

## Download Analytics Zoo and the pre-trained model
- You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.
- Download pre-trained int8 quantized ResNet50 model from [here](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/image-classification/analytics-zoo_resnet-50-int8_imagenet_0.5.0.model).

## Examples
This folder contains three examples for BigDL VNNI support:
- [Predict](#predict)
- [ImageNetEvaluation](#imagenetevaluation)
- [Perf](#perf)

__Remarks:__
- If you are using an int8 quantized model pre-trained in Analytics Zoo, you can find the following info in console log:
```
INFO  ImageModel$:132 - Loading an int8 convertible model. Quantize to an int8 model for better performance
```
- You may need to enlarge memory configurations depending on the size of your input data.
- You can set `-Dbigdl.mklNumThreads` if necessary.

---
### Predict
This example demonstrates how to do image classification with the pre-trained int8 model.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
imagePath=the folder path containing images
modelPath=the path to the downloaded int8 model

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master local[*] \
    --class com.intel.analytics.zoo.examples.vnni.bigdl.Predict \
    -f ${imagePath} -m ${modelPath}
```

__Options:__
- `-f` `--folder`: The local folder path that contains images for prediction.
- `-m` `--model`: The path to the downloaded int8 model.
- `--topN`: The top N classes with highest probabilities as output. Default is 5.

__Sample console log output__:
```
INFO  Predict$:64 - image : 1.jpg, top 5
INFO  Predict$:68 - 	 class: kelpie, credit: 0.96370447
INFO  Predict$:68 - 	 class: Rottweiler, credit: 0.026292335
INFO  Predict$:68 - 	 class: Eskimo dog, husky, credit: 0.0019479054
INFO  Predict$:68 - 	 class: German shepherd, German shepherd dog, German police dog, alsatian, credit: 0.001165287
INFO  Predict$:68 - 	 class: Doberman, Doberman pinscher, credit: 8.323631E-4
```

---
### ImageNetEvaluation
This example evaluates the pre-trained int8 model using Hadoop SequenceFiles of ImageNet no-resize validation images.

You may refer to this [script](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/utils/ImageNetSeqFileGenerator.scala) to generate sequence files for images.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
imagePath=the folder path that contains sequence files of ImageNet no-resize validation images.
modelPath=the path to the downloaded int8 model

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master ${MASTER} \
    --class com.intel.analytics.zoo.examples.vnni.bigdl.ImageNetEvaluation \
    -f ${imagePath} -m ${modelPath}
```

__Options:__
- `-f` `--folder`: The folder path that contains sequence files of ImageNet no-resize validation images.
- `-m` `--model`: The path to the downloaded int8 model.
- `--partitionNum`: The partition number of the dataset. Default is 32.

__Sample console log output__:
```
Top1Accuracy is Accuracy(correct: 37912, count: 50000, accuracy: 0.75824)
Top5Accuracy is Accuracy(correct: 46332, count: 50000, accuracy: 0.92664)
```

---
### Perf
This example runs in local mode and calculates performance data (i.e. throughput and latency) for the pre-trained int8 model using dummy input.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
modelPath=the path to the downloaded int8 model

java -cp ${ANALYTICS_ZOO_JAR}:${SPARK_HOME}/jars/* com.intel.analytics.zoo.examples.vnni.bigdl.Perf -m ${modelPath} -b 64
```

__Options:__
- `-m` `--model`: The path to the downloaded int8 model.
- `-b` `--batchSize`: The batch size of input data. Default is 32.
- `-i` `--iteration`: The number of iterations to run the performance test. Default is 1000. The result should be the average of each iteration time cost.

__Sample console log output__:
```
......
INFO  Perf$:67 - Iteration 796, batch 64, takes 29802474 ns, throughput is 2147.47 imgs/sec
INFO  Perf$:67 - Iteration 797, batch 64, takes 30284076 ns, throughput is 2113.32 imgs/sec
INFO  Perf$:67 - Iteration 798, batch 64, takes 29884174 ns, throughput is 2141.60 imgs/sec
......
INFO  Perf$:82 - Iteration 928, latency for a single image is 1.683318 ms
INFO  Perf$:82 - Iteration 929, latency for a single image is 1.748185 ms
INFO  Perf$:82 - Iteration 930, latency for a single image is 1.622709 ms
......
```
