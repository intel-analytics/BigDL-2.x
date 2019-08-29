# Analytics-Zoo InferenceModel with OpenVINO accelerating on Flink Streaming
Analytics-zoo provides Inference Model package for speeding up prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).

This is the example of streaming with Flink and Resnet50 model, as well as using Analytics-Zoo InferenceModel and OpenVINO backend to accelerate prediction. See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/model-inference-examples/model-inference-flink/src/main/scala/com/intel/analytics/zoo/apps/model/inference/flink/Resnet50ImageClassification) for the whole program.

There are four parts in this tutorial.
- [Dataset and pre-trained models](#dataset-and-pre-trained-models)
- [Getting started Analytics-Zoo InferenceModel](#getting-started-analytics-zoo-inferencemodel)
- [Getting started Flink program](#getting-started-flink-program)
- [Running the example on a local machine or a cluster](#running-the-example-on-a-local-machine-or-a-cluster)

## Dataset and pre-trained models
### Extract dataset and pre-trained model
In this example, we extract image from ImageNet and save it in `resources` folder. The pre-trained model ResNet50 can be obtained from [here](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz).
### Data preparation and pre-processing
Get image from `resources` folder, and return an input stream using `classLoader.getResourceAsStream`
```
val classLoader = this.getClass.getClassLoader
val content = classLoader.getResourceAsStream("n02110063_11239.JPEG")
```
Then, you may pre-process data as you need. In this sample, `trait ImageProcessing`  provides approaches to convert format, resize and normalize. The input stream is supposed to be converted as below:
```
val imageBytes = Stream.continually(content.read).takeWhile(_ != -1).map(_.toByte).toArray
val imageMat = byteArrayToMat(imageBytes)
val imageCent = centerCrop(imageMat, 224, 224)
val imageTensor = matToNCHWAndRGBTensor(imageCent)
val imageNormlized = channelScaledNormalize(imageTensor)
```
## Getting started Analytics-Zoo InferenceModel
Define a class extended analytics-zoo `InferenceModel`. As we use OpenVINO backend in this example, it allows passing and loading parameters to convert to OpenVINO model.
This is the sample of defining a `Resnet50InferenceModel` class. See more details [here](https://github.com/intel-analytics/analytics-zoo/blob/master/apps/model-inference-examples/model-inference-flink/src/main/scala/com/intel/analytics/zoo/apps/model/inference/flink/Resnet50ImageClassification/Resnet50InferenceModel.scala).

```
class Resnet50InferenceModel(var concurrentNum: Int = 1, modelType: String, modelBytes: Array[Byte], inputShape: Array[Int], ifReverseInputChannels: Boolean, meanValues: Array[Float], scale: Float)
extends InferenceModel(concurrentNum) with Serializable {
  doLoadTF(null, modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale)
  }
}
```

## Getting started Flink program
### Obtain an execution environment
The first step is to create an execution environment. The `StreamExecutionEnvironment` is the context in which a streaming program is executed. `getExecutionEnvironment` is the typical function creating an environment to execute your program when the program is invoked on your local machine or a cluster.
```
val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment
```
### Create and transform DataStreams
`StreamExecutionEnvironment` provides several stream sources function. As we simulate one hundred inputs and use `List` to hold them, we can create a DataStream from a collection using `fromCollection()` method.
```
val dataStream: DataStream[Array[Float]] =  env.fromCollection(inputs)
val tensorStream: DataStream[JList[JList[JTensor]]] = dataStream.map(value => {
  val input = new JTensor(value, Array(1, 224, 224, 3))
  val data = Arrays.asList(input)
  List(data).asJava
})
```
### Specify Transformation Functions
Define a class extends `RichMapFunction`. Three main methods of rich function in this example are open, close and map. `open()` is initialization method. `close()` is called after the last call to the main working methods. `map()` is the user-defined function, mapping an element from the input data set and to one exact element, ie, `JList[JList[JTensor]]`.
```
class ModelPredictionMapFunction(modelType: String, modelBytes: Array[Byte], inputShape: Array[Int], ifReverseInputChannels: Boolean, meanValues: Array[Float], scale: Float)
extends RichMapFunction[JList[JList[JTensor]], JList[JList[JTensor]]] {
  var resnet50InferenceModel: Resnet50InferenceModel = _
  override def open(parameters: Configuration): Unit = {
    resnet50InferenceModel = new Resnet50InferenceModel(1, modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale)
  }
  override def close(): Unit = {
    resnet50InferenceModel.release()
  }
  override def map(in: JList[JList[JTensor]]): JList[JList[JTensor]] = {
    resnet50InferenceModel.doPredict(in)
  }
}
```
Pass the `RichMapFunctionn` function to a `map` transformation.
```
val resultStream = tensorStream.map(new ModelPredictionMapFunction(modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale))
```
### Trigger the program execution
The program is actually executed only when calling `execute()` on the `StreamExecutionEnvironment`. Whether the program is executed locally or submitted on a cluster depends on the type of execution environment.
```
env.execute()
```
### Collect final results
Create an iterator to iterate over the elements of the DataStream.
```
val results = DataStreamUtils.collect(resultStream.javaStream).asScala
```
## Running the example on a local machine or a cluster
### Build the project
Build the project using Maven because we need the jar file for running on the cluster. Go to the root directory of model-inference-flink and execute the mvn clean package command, which prepares the jar file for model-inference-flink:
```
mvn clean package
```
The resulting jar file will be in the target subfolder: target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar. We’ll use this later.

### Start and stop Flink
You may start a flink cluster if there is no runing one. Go to the location where you installed Flink :
```
./bin/start-cluster.sh
```
Check the Dispatcher's web frontend at http://localhost:8081 and make sure everything is up and running.
To stop Flink when you're done type:
```
./bin/stop-cluster.sh
```

### Run the Example
* Run `export FLINK_HOME=the root directory of flink`.
* Run `export ANALYTICS_ZOO_HOME=the folder of Analytics Zoo project`.
* Download [resnet_v1_50 model](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz). Run `export MODEL_PATH=path to the downloaded model`.
* Edit flink-conf.yaml to set heap size or the number of task slots as you need, ie,  `jobmanager.heap.size: 10g`
* Run the follwing command with arguments to submit the Flink program. Change parameter settings as you need.

```bash
${FLINK_HOME}/bin/flink run \
    -m localhost:8181 -p 2 \
    -c com.intel.analytics.zoo.apps.model.inference.flink.ImageClassificationStreaming  \
    ${ANALYTICS_ZOO_HOME}/apps/model-inference-examples/model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
    --modelType resnet_v1_50 --checkpointPathcheckpointPath ${MODEL_PATH}  \
    --inputShape "1,224,224,3" --ifReverseInputChannels true --meanValues "123.68,116.78,103.94" --scale 1
```
### The result
The output of that command should look similar to this, if everything went according to plan:
```bash
Starting execution of program
start ImageClassificationStreaming job...
(params resolved,resnet_v1_50,/root/to/models/resnet_v1_50.ckpt,1,224,224,3,true,123.68,116.78,103.94,1.0)
org.apache.flink.api.common.ExecutionConfig@86941f8b
 Printing result to stdout.
[[JTensor{data=[6.978136E-5, 9.844725E-4, 2.3672989E-4, ...], shape=[1, 1000]}]]
Program execution finished
Job with JobID e6c2eefb0eaae22c3fda0bfb4dff4078 has finished.
Job Runtime: 15103 ms
```
You can also get an overview of your cluster resources and running jobs by checking out the Flink dashboard at http://localhost:8081.
