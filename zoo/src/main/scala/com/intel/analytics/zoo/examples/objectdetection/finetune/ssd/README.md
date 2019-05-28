This example shows how to fine tune a SSD(Single Shot MultiBox Detector) model with your own dataset.
[SSD](https://research.google.com/pubs/pub44872.html) is one of the state-of-the-art
 object detection pipeline.

## Prepare custom training dataset
Please make sure your dataset has folder structure like:
Custom dataset
+-- images/
+-- annotations/
+-- ImageSets/
+-- classname.txt

* images: folder that contains set of images
* annotations: folder that contains set of annotation files
annotations are saved as XML files in PASCAL VOC format. One XML for one image. Eg:
<annotation>
	<folder>messi</folder>
	<filename>100.jpg</filename>
	<path>100.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>1280</width>
		<height>720</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>messi</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>691</xmin>
			<ymin>186</ymin>
			<xmax>767</xmax>
			<ymax>300</ymax>
		</bndbox>
	</object>
</annotation>

* ImageSets: folder that contains image-annotation mappings. Each line contains a mapping of image to its annotation.
 Eg, contain two files, one for training set and the other for validation set. for training set,
images/114.png annotations/114.xml
images/233.png annotations/233.xml
images/323.png annotations/323.xml
images/254.png annotations/254.xml

* classname.txt: class name for label mapping. See pascal_classname.txt as an example.
 Eg,
 __background__
 object1
 object2

## Generate Sequence Files

```bash
export ANALYTICS_ZOO=where the zoo is located
export ZOO_JAR_SPARK_PATH=${ANALYTICS_ZOO}/zoo/target/analytics-zoo-bigdl_BIGDL_VERSION-spark_SPARK_VERSION-ZOO_VERSION-jar-with-dependencies-and-spark.jar
export ZOO_JAR_PATH=${ANALYTICS_ZOO}/dist/lib/analytics-zoo-bigdl_BIGDL_VERSION-spark_SPARK_VERSION-ZOO_VERSION-jar-with-dependencies.jar

java -cp ${ZOO_JAR_SPARK_PATH} \
         com.intel.analytics.zoo.models.image.objectdetection.common.dataset.RoiImageSeqGenerator \
     -f $data -o output -i $name -p $num
```

```-f``` is your image folder where you put the images,annotations,ImageSets,classname.txt
```-o``` is the output folder
```-i``` is the imageset name, eg. if you have train.txt under ImageSet, name should be "train"
```-p``` is the parallel number

note that a spark jar is needed as dependency.

## Download pretrained model
Download pre-trained ssd models from [Object Detection](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/object-detection.md)

## Run the train example
Example command for running in Spark cluster (eg. spark local)

```
spark-submit \
    --master local[4] \
    --driver-memory 100g \
    --executor-memory 100g \
    --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" \
    --class com.intel.analytics.zoo.examples.objectdetection.finetune.ssd.Train \
    ${ZOO_JAR_PATH} --resolution 300 --model analytics-zoo_ssd-vgg16-300x300_PASCAL_0.1.0.model \
    --class xxx -f xxx -v xxx -e 1 -b 8 --modelDir folder -l 0.0001 --learningRateDecay 0.0005
```
* resolution: input resolution 300 or 512
* model: model snapshot location
* class : class file which contains object class names
* f: url of hdfs or local folder store the train hadoop sequence image files.
* v: url of hdfs or local folder store the validation hadoop sequence image files.
* e: max epochs to run
* b: mini-batch size. Should be multiple of total cores
* modelDir: checkpoint directory, and related summary director
* l: learning rate
* learningRateDecay: learning rate decay
