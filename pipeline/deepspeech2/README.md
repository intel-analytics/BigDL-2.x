
This demo code contains the inference based on pre-trained Deep Speech 2 model on BigDL 0.1.
(Soon to be updated to 0.3). The example runs on Spark 2.0+

### ds2 model (~387MB):
https://drive.google.com/open?id=0B9zID9CU9HQeU1luc2ZKSHA1MjA


### Run inference with example:

1. Download model file "dp2.bigdl" from the link above.

2. Import the project into IDE or build with "mvn clean package".

3. script to run ds2 inference:

```shell
   spark-submit --master local[1] \
   --conf spark.driver.memory=20g \
   --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" \
   --class com.intel.analytics.zoo.pipeline.deepspeech2.example.InferenceExample \
   deepspeech2-0.1-SNAPSHOT-jar-with-dependencies.jar  \
   -m /path/to/dp2.bigdl -d /path/data/1462-170145-0004.flac -n 1 -p 1 -s 30
   ```



