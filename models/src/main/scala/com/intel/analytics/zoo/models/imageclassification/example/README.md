## Image Classification example With Spark and Storm

Users can apply bigdl predictor in both Spark and Storm, for batching and streaming scenario respectively

### Image Classification with Spark

Run Spark example with below command

master=spark://xxx.xxx.xxx.xxx:xxxx # please set your own spark master
executor-memory=...
executor-cores=...
total-executor-cores=...
imageFoler=...  #please set image folder you want to predict on
modelPath =... #Bigdl model, you could find pre-trained models in [Model Zoo](https://github.com/intel-analytics/analytics-zoo/tree/master/models)
topN = ... #How many top results you want to get
partition = ... #partition number

spark-submit --driver-memory 30g --master $master --executor-memory 100g                 \

             --executor-cores $executor-cores                                                      \
             --total-executor-cores $total-executor-cores                                              \
             --driver-class-path models-0.1-SNAPSHOT-jar-with-dependencies.jar \
             --class com.intel.analytics.zoo.models.imageclassification.example.Predict          \
                       models-0.1-SNAPSHOT-jar-with-dependencies.jar           \
             -f  $imageFoler \
             --model $modelPath\
             --topN $topN -p $partition

### Image Classification with Storm

In order to run Storm example, we should build the `models` module with `all-in-one`

Run Storm example with below command


imageFoler=...  #please set image folder you want to predict on
modelPath =... #Bigdl model, you could find pre-trained models in [Model Zoo](https://github.com/intel-analytics/analytics-zoo/tree/master/models)
topN = ... #How many top results you want to get
$localMode=... #if run in local model or not
storm jar   models-0.1-SNAPSHOT-jar-with-dependencies.jar \
            com.intel.analytics.zoo.models.imageclassification.example.PredictStreaming \
            -f  $imageFoler \
            --model $modelPath\
            --topN $topN
            --localMode $localMode 
