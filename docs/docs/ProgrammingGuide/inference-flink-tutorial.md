# Analytics-Zoo InferenceModel with OpenVINO accelerating on Flink Streaming

Analytics-Zoo provides Inference Model package for speeding up prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR). 

Now, we will use an example to introduce how to use Analytics-Zoo Inference Model with Tensorflow Resnet50 model, as well as applying OpenVINO backend to accelerate prediction on Flink streaming. See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/model-inference-examples/model-inference-flink/src/main/scala/com/intel/analytics/zoo/apps/model/inference/flink/Resnet50ImageClassification) for the whole program. 

### Data

In this tutorial, we will use the **ImageNet** dataset.  For fun, we extract just one image and save it in `resources` folder. Later, we will create a list to hold one hundred repeated inputs to simulate the prediction of one hundred images. Certainly, you are encouraged to extract as many various images as you need for practicing.

The dataset will be loaded directly from `resources`  folder. The program directory structure and image path should be:

```
/path/to/model-inference-flink
├── pom.xml
├── src
│   └── main
│       ├── resources
│       │   └── n02110063_11239.JPEG
│       └── scala
```

Next, get image from `resources` folder, and return an input stream.

```
# Load data from relative path of the resources folder
# classLoader
val classLoader = this.getClass.getClassLoader
val content = classLoader.getResourceAsStream("n02110063_11239.JPEG")
```

Then, you may pre-process data as you need. In this sample, `trait ImageProcessing`  is prepared to provide approaches to convert format, resize and normalize.

- Image is loaded as streaming. Target is `Tensor[Float]`
- `trait ImageProcessing` :
  - `byteArrayToMat(Array[Byte])`: convert `Array[byte]` to `OpenCVMat`
  - `centerCrop(mat,W,H,normalized,isClip) `: do a center crop by resizing a square. `normalized`and `isclip` are optional.
  - `matToNCHWAndRGBTensor(mat)`: convert `OpenCVMat` to `Tensor[Float]`
  - `channelScaledNormalize(tensor, meanR, meanG, meanB, scale)`: normalize with the input RGB mean and scale

The input stream is supposed to be converted as below:

```
# Convert image stream to Array[byte]
val imageBytes = Stream.continually(content.read).takeWhile(_ !=-1).map(_.toByte).toArray

# apply methods from trait ImageProcessing  
val imageMat = byteArrayToMat(imageBytes)
val imageCent = centerCrop(imageMat, 224, 224)
val imageTensor = matToNCHWAndRGBTensor(imageCent)
val tensorNormalized = channelScaledNormalize(imageTensor, 123, 116, 103, 1)
println(tensorNormalized)
```

Out:

```
(1,.,.) =
-101.0  -110.0  -114.0  ...     15.0    72.0    43.0
-85.0   -93.0   -109.0  ...     47.0    79.0    24.0
-94.0   -76.0   -83.0   ...     6.0     84.0    18.0
...
79.0    -55.0   -77.0   ...     -39.0   101.0   24.0
80.0    -88.0   -72.0   ...     80.0    61.0    48.0
57.0    -102.0  -107.0  ...     101.0   25.0    47.0

(2,.,.) =
-94.0   -103.0  -107.0  ...     -9.0    68.0    30.0
-70.0   -81.0   -104.0  ...     50.0    61.0    6.0
-77.0   -64.0   -67.0   ...     29.0    63.0    8.0
...
58.0    -30.0   -50.0   ...     -35.0   95.0    -3.0
59.0    -79.0   -60.0   ...     83.0    38.0    22.0
51.0    -82.0   -100.0  ...     92.0    8.0     32.0

(3,.,.) =
-79.0   -90.0   -94.0   ...     9.0     59.0    19.0
-55.0   -64.0   -86.0   ...     64.0    54.0    -1.0
-62.0   -48.0   -59.0   ...     32.0    54.0    5.0
...
59.0    -15.0   -38.0   ...     -39.0   89.0    -2.0
51.0    -71.0   -54.0   ...     84.0    15.0    0.0
47.0    -87.0   -95.0   ...     80.0    -15.0   22.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x224x224])
```

Since we simulate one hundred images as inputs, we use a `List` to hold the repeated  `Array[Float]` 

```
# Convert tensor to Array[Float] 
val input = new Array[Float](imageTensor.nElement())

# Create data list containing one hundred same image Array
val inputs = List.fill(100)(input)
println(inputs)
```

Out:

```
List([F@194bcebf, [F@194bcebf, [F@194bcebf, [F@194bcebf, [F@194bcebf, ... [F@194bcebf)
```

### Defining an InferenceModel

Define a class extended analytics-zoo `InferenceModel`. As we use OpenVINO backend in this example, it allows passing and loading parameters to convert to OpenVINO model. The pre-trained model ResNet50 can be obtained from [here](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz).

Let's define the input parameters of the class:

- **concurrentNum**-the number of requests a model can accept concurrently.
- **modelType**-the pre-trained model type.
- **modelBytes**-the bytes of the model.
- **inputShape**-array which contains the dimensions of the tensor. It should be fed to an input node(s) of the model.
- **ifReverseInputChannels**-the boolean value of if need reverse input channels. Switch the input channels order from RGB to BGR (or vice versa).
- **meanValues**- the required mean values to convert models from [TensorFlow*-Slim Image Classification Model Library](https://github.com/tensorflow/models/tree/master/research/slim/README.md)  All input values coming from original network inputs will be divided by this value.
- **scale**-to be used for the input image per channel. 

```
# concurrentNum
var concurrentNum = 1

# modelType
var modelType = "resnet_v1_50"

# ifReverseInputChannels
var ifReverseInputChannels = true

# inputShape
var inputShape = Array(1, 224, 224, 3)

# meanValues to convert resnet_v1_50 model
var meanValues = Array(123.68f, 116.78f, 103.94f)

# scale to convert resnet_v1_50 model
var scale = 1.0f

# To obtain model bytes, there are several steps
# abstract path name of the model
var checkpointPath: String = "/path/to/models/resnet_v1_50.ckpt"

# length of the file in bytes
val fileSize = new File(checkpointPath).length()

# Create a file inputStream to read streams of bytes
val inputStream = new FileInputStream(checkpointPath)

#  convert fileSize to Array[Byte]
val modelBytes = new Array[Byte](fileSize.toInt)
inputStream.read(modelBytes)
```

Let's define a `Resnet50InferenceModel` class to extend analytics-zoo `InferenceModel`. 

```
class Resnet50InferenceModel(var concurrentNum: Int = 1, modelType: String, modelBytes: Array[Byte], inputShape: Array[Int], ifReverseInputChannels: Boolean, meanValues: Array[Float], scale: Float)
extends InferenceModel(concurrentNum) with Serializable {

  doLoadTF(null, modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale)
  
  }
}
```

### Getting started the Flink program

We will do the following steps in order:

1. Obtain an execution environment 
2. Create and transform DataStreams
3. Specify Transformation Functions
4. Trigger the program execution
5. Collect final results
6. Running the example on a local machine or a cluster

#### 1. Obtain an execution environment

The first step is to create an execution environment. The `StreamExecutionEnvironment` is the context in which a streaming program is executed. `getExecutionEnvironment` is the typical function creating an environment to execute your program when the program is invoked on your local machine or a cluster.

```
val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment
```

#### 2. Create and transform DataStreams

`StreamExecutionEnvironment` provides several stream sources function. As we simulate one hundred inputs and use `List` to hold them, we can create a DataStream from a collection using `fromCollection()` method.

```
# dataStream
val dataStream: DataStream[Array[Float]] =  env.fromCollection(inputs)

# map to DataStream[JList[JList[JTensor]]]
val tensorStream: DataStream[JList[JList[JTensor]]] = dataStream.map(value => {
  val input = new JTensor(value, Array(1, 224, 224, 3))
  val data = Arrays.asList(input)
  List(data).asJava
})
```

Out:

```
(tensorStream,org.apache.flink.streaming.api.scala.DataStream@4808bc9b)
```

#### 3. Specify transformation functions

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

Out:

```
(resultStream,org.apache.flink.streaming.api.scala.DataStream@4b2c5e02)
```

#### 4. Trigger the program execution

The program is actually executed only when calling `execute()` on the `StreamExecutionEnvironment`. Whether the program is executed locally or submitted on a cluster depends on the type of execution environment.

```
env.execute()
```

#### 5. Collect final results

Finally, create an iterator to iterate over the elements of the DataStream. 

```
val results = DataStreamUtils.collect(resultStream.javaStream).asScala
```

At this step, we complete the whole program. Let's start how to run the example. 

#### 6. Running the example on a local machine or a cluster

- ##### Build the project

Build the project using Maven because we need the jar file for running on the cluster. Go to the root directory of model-inference-flink and execute the mvn clean package command, which prepares the jar file for model-inference-flink:

```
mvn clean package
```

The resulting jar file will be in the target subfolder: target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar. We’ll use this later.

- ##### Start and stop Flink

You may start a flink cluster if there is no runing one. Go to the location where you installed Flink :

```
./bin/start-cluster.sh
```

Check the Dispatcher's web frontend at http://localhost:8081 and make sure everything is up and running.
To stop Flink when you're done type:

```
./bin/stop-cluster.sh
```

- ##### Run the Example

  - Run `export FLINK_HOME=the root directory of flink`.
  - Download [resnet_v1_50 model](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz). Run `export MODEL_PATH = path to the downloaded model`.
  - Edit flink-conf.yaml to set heap size or the number of task slots as you need, ie, `jobmanager.heap.size: 10g`

All are ready! Let's run the following command with arguments to submit the Flink program. Change parameter settings as you need.

```bash
${FLINK_HOME}/bin/flink run \
    -m localhost:8081 -p 2 \
    -c com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification.
    ImageClassificationStreaming  \
    /path/to/model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
    --modelType resnet_v1_50 --checkpointPath ${MODEL_PATH}  \
    --inputShape "1,224,224,3" --ifReverseInputChannels true --meanValues "123.68,116.78,103.94" --scale 1
```

The output of that command should look similar to this, if everything went according to plan:

```bash
Starting execution of program
start ImageClassificationStreaming job...
(params,resnet_v1_50,/path/to/models/resnet_v1_50.ckpt,1,224,224,3,true,123.68,116.78,103.94,1.0)
org.apache.flink.api.common.ExecutionConfig@86941f8b
Printing result to stdout. Since we play with one hundred repeated inputs, there are the same number of equal results.
[[JTensor{data=[6.978136E-5, 9.844725E-4, 2.3672989E-4, 2.969411E-4, 4.7597036E-4, 1.715969E-4, 1.1608376E-4, 1.8288662E-4, 7.620713E-5, ...], shape=[1, 1000]}]]
[[JTensor{data=[6.978136E-5, 9.844725E-4, 2.3672989E-4, 2.969411E-4, 4.7597036E-4, 1.715969E-4, 1.1608376E-4, 1.8288662E-4, 7.620713E-5, ...], shape=[1, 1000]}]]
...
Program execution finished
Job with JobID f0bedd54bd81db640833c283a9283289 has finished.
Job Runtime: 14830 ms
```

#### Wrapping up

we have reached the end of the tutorial. In this tutorial, you have learned how to create the Analytics-Zoo `InferenceModel` class for loading and prediction with a deep learning model. With that, you defined your own `RichMapFunction` and started with the Flink streaming. 

What go for next? You could take a practice. Load the data and model you need to see what speedup you get.
