## Object Detection example
This example illustrates how to detect objects in image with pre-trained model.

### Run steps
#### 1. Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

#### 2. Install OpenCV
The example uses OpenCV library to save image. Please install it before run this example.

#### 3. Prepare pre-trained models

Download pre-trained models from [Object Detection](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/object-detection.md)

#### 4. Prepare predict dataset

Put your image data for prediction in the ./image folder.

#### 5. Run the example

modelPath=... // model path. Local file system/HDFS/Amazon S3 are supported

imagePath=... // image path. Local file system/HDFS are supported. With local file system, the files need to be available on all nodes in the cluster and please use file:///... for local files.

outputPath=... // output path. Currently only support local file system.

partitionNum=... // Optional, a suggestion value of the minimal partition number

##### * Run after pip install

You can easily use the following commands to run this example:

```bash
export SPARK_DRIVER_MEMORY=10g
python path/to/predict.py ${model_path} ${image_path} ${output_path} ${partitionNum}
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.

##### * Run with prebuilt package

```bash
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master local[4] \
    --driver-memory 10g \
    --executor-memory 10g \
    path/to/predict.py ${model_path} ${image_path} ${output_path} ${partitionNum}
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.


### Results
You can find new generated images stored in output_path, and the objects in the images are with a box around them [labeled "name"]
