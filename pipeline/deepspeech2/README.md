
This demo code contains the inference based on pre-trained Deep Speech 2 model on BigDL 0.3. The example runs on Spark 2.0+

### ds2 model for branch-0.3:

Download [deepspeech2 model](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/deepspeech2/dp2-0.3.bigdl)


### Run inference with example:

1. Download model file "dp2-0.3.bigdl" from the link above.

2. Import the project into IDE or build with "mvn clean package".

3. script to run ds2 inference:

```shell
   spark-submit --master local[1] \
   --conf spark.driver.memory=20g \
   --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" \
   --driver-class-path deepspeech2-0.3-SNAPSHOT-jar-with-dependencies.jar \
   --class com.intel.analytics.zoo.pipeline.deepspeech2.example.InferenceExample \
   deepspeech2-0.3-SNAPSHOT-jar-with-dependencies.jar  \
   -m /path/to/dp2-0.3.bigdl -d /path/data/1462-170145-0004.flac -n 1 -p 1 -s 30
   ```

### Run inference evaluate:

Following the above 1 and 2 steps to prepare environment. Then:

- Prepare evaluate data: 

Make sure your audio files in one folder, eg. ```/path/data```, and create a mapping.txt to keep audio files and responding text contents, format like this:
```
6313-66129-0000 HE NO DOUBT WOULD BRING FOOD OF SOME KIND WITH HIM
6313-66129-0001 WITH A SHOUT THE BOYS DASHED PELL MELL TO MEET THE PACK TRAIN AND FALLING IN BEHIND THE SLOW MOVING BURROS URGED THEM ON WITH DERISIVE SHOUTS AND SUNDRY RESOUNDING SLAPS ON THE ANIMALS FLANKS
6313-66129-0002 COLD WATER IS THE MOST NOURISHING THING WE'VE TOUCHED SINCE LAST NIGHT
```
Then the structure of `/path/data` directory looks like:
```{r, engine='sh'}
$ [/path/data]$ tree . -L 1
.
├── 6313-66125-0000.flac
├── 6313-66125-0001.flac
├── 6313-66125-0002.flac
└── mapping.txt
```

- script to run ds2 inference evaluate:

```shell
   spark-submit --master local[1] \
   --conf spark.driver.memory=20g \
   --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" \
   --driver-class-path deepspeech2-0.3-SNAPSHOT-jar-with-dependencies.jar \
   --class com.intel.analytics.zoo.pipeline.deepspeech2.example.InferenceEvaluate \
   deepspeech2-0.3-SNAPSHOT-jar-with-dependencies.jar  \
   -m /path/to/dp2-0.3.bigdl -d /path/data -n 1 -p 1 -s 30
   ```

where 

 ```-m``` is the path to the model.
 ```-d``` is the evaluate data folder.
 ```-n``` is file number to do evaluation, default 8.
 ```-p``` is partition number, default 4.
 ```-s``` is audio segment length in seconds. Default is 30.