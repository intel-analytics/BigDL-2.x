

### ds2 model (~387MB):
https://drive.google.com/open?id=0B_s7AwBOnuD-ckRqQWM3WFctZmM

### sample audio files:

Please upload your samples audio to HDFS. For instance, ```hdfs://127.0.0.1:9001/deepspeech/data/dev-clean```

The sample audio must include a mapping file which records the paths of audio and their corresponding transcripts. You can download sample audio files from [librispeech](http://www.openslr.org/12/).

For instance (mapping.txt):

```
6313-66129-0000 HE NO DOUBT WOULD BRING FOOD OF SOME KIND WITH HIM
6313-66129-0001 WITH A SHOUT THE BOYS DASHED PELL MELL TO MEET THE PACK TRAIN AND FALLING IN BEHIND THE SLOW MOVING BURROS URGED THEM ON WITH DERISIVE SHOUTS AND SUNDRY RESOUNDING SLAPS ON THE ANIMALS FLANKS
6313-66129-0002 COLD WATER IS THE MOST NOURISHING THING WE'VE TOUCHED SINCE LAST NIGHT
6313-66129-0003 WE DID NOT IT MUST HAVE COME TO LIFE SOME TIME DURING THE NIGHT AND DUG ITS WAY OUT LAUGHED TAD
...
```

Please save your ds2.model to HDFS. For instance, ```hdfs://127.0.0.1:9001/deepspeech/data/ds2.model```


### script to run ds2 on clusters:
If the input audio sample is much too long (> 10 seconds), it is suggested to segment your sample into smaller clips. Please setting the -s or --segment argument to be true to indicate whether to segment your input sample or not. The default segment length will be 4 seconds.


#### Spark local mode
```shell
./bigdl.sh -- spark-submit --master local[4] \
--conf spark.driver.memory=20g \
--conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" \
--class com.intel.analytics.deepspeech2.example.InferenceEvaluate \
deepspeech2-0.1-SNAPSHOT-jar-with-dependencies.jar  \
-d hdfs://127.0.0.1:9001/deepspeech/data/dev-clean \
-m hdfs://127.0.0.1:9001/deepspeech/data -p 4 -n 4 -s false
```

#### Spark standalone mode
```shell
./bigdl.sh -- spark-submit \
--master spark://... \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" \
--class com.intel.analytics.deepspeech2.example.InferenceEvaluate \
deepspeech2-0.1-SNAPSHOT-jar-with-dependencies.jar \
-d hdfs://Gondolin-Node-002:9000/deepspeech/data/dev-clean \
-m hdfs://Gondolin-Node-002:9000/deepspeech/data/ds2.model -p 9 -n 27 -s false
```