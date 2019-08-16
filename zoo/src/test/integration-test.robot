*** Settings ***
Documentation    Zoo Integration Test
Resource         common.robot
Suite Setup      Prepare DataSource And Verticals
Suite Teardown   Delete All Sessions
Test template    Zoo Test

*** Variables ***
@{verticals}  ${spark_210_3_vid}    ${hdfs_264_3_vid}

*** Test Cases ***   SuiteName                             VerticalId
1                    Spark2.1 Test Suite                   ${spark_210_3_vid}
2                    Yarn Test Suite                       ${hdfs_264_3_vid}

*** Keywords ***
Build SparkJar
   [Arguments]       ${spark_profile}
   ${build}=         Catenate                        SEPARATOR=/    ${curdir}    make-dist.sh
   Log To Console    ${spark_profile}
   Log To Console    start to build jar
   Log To Console    make-dist.sh -P ${spark_profile} -Dbigdl.version=${bigdl_version}...
   Run               ${build} -P ${spark_profile} -Dbigdl.version=${bigdl_version}
   Remove File       ${jar_path}
   Move File         zoo/target/analytics-zoo-bigdl_${bigdl_version}-spark_${spark_version}-${version}-jar-with-dependencies.jar    ${jar_path}
   Log To Console    build jar finished

Run Spark Test
   [Arguments]                      ${submit}                   ${spark_master}
   Create Directory                 /tmp/objectdetection/output
   Log To Console                   begin session recommender
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 10g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.recommendation.SessionRecExp ${jar_path} --input ${public_hdfs_master}:9000/ecommerce --outputDir ./output/
   Log To Console                   begin text classification
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 2g --executor-memory 2g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.textclassification.TextClassification ${jar_path} --batchSize 128 --dataPath ${integration_data_dir}/text_data/20news-18828 --embeddingPath ${integration_data_dir}/text_data/glove.6B --partitionNum 32 --nbEpoch 2
   Log To Console                   begin image classification
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 10g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.imageclassification.Predict ${jar_path} -f ${public_hdfs_master}:9000/kaggle/train_100 --topN 1 --model ${integration_data_dir}/models/analytics-zoo_squeezenet-quantize_imagenet_0.1.0.model --partition 32
   Log To Console                   begin object detection
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 1g --executor-memory 1g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.objectdetection.inference.Predict ${jar_path} --image ${public_hdfs_master}:9000/kaggle/train_100 --output /tmp/objectdetection/output --modelPath ${integration_data_dir}/models/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model --partition 32
   Log To Console                   begin recommendation wideAndDeep on ml-1m
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 5g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.recommendation.WideAndDeepExample ${jar_path} --inputDir ${public_hdfs_master}:9000/ml-1m --dataset ml-1m
   Log To Console                   begin recommendation wideAndDeep on census
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 5g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.recommendation.WideAndDeepExample ${jar_path} --inputDir ${public_hdfs_master}:9000/census --dataset census
   Log To Console                   begin recommendation NCF
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 5g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.recommendation.NeuralCFexample ${jar_path} --inputDir ${public_hdfs_master}:9000/ml-1m --batchSize 8000
   Log To Console                   begin anomalydetection AnomalyDetection
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 10g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.anomalydetection.AnomalyDetection ${jar_path} --inputDir ${public_hdfs_master}:9000/NAB/nyc_taxi
   Log To Console                   begin finetune
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 10g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.nnframes.finetune.ImageFinetune ${jar_path} --modelPath ${integration_data_dir}/models/analytics-zoo_inception-v1_imagenet_0.1.0.model --batchSize 32 --imagePath ${public_hdfs_master}:9000/dogs_cats/samples --nEpochs 2
   Log To Console                   begin imageInference
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 10g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.nnframes.imageInference.ImageInferenceExample ${jar_path} --caffeDefPath ${integration_data_dir}/models/nnframes/deploy.prototxt --caffeWeightsPath ${integration_data_dir}/models/nnframes/bvlc_googlenet.caffemodel --batchSize 32 --imagePath ${public_hdfs_master}:9000/dogs_cats/samples
   Log To Console                   begin imageTransferLearning
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 10g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.nnframes.imageTransferLearning.ImageTransferLearning ${jar_path} --caffeDefPath ${integration_data_dir}/models/nnframes/deploy.prototxt --caffeWeightsPath ${integration_data_dir}/models/nnframes/bvlc_googlenet.caffemodel --batchSize 32 --imagePath ${public_hdfs_master}:9000/dogs_cats/samples --nEpochs 20
   Log To Console                   begin QARanker
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 3g --executor-memory 3g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.qaranker.QARanker ${jar_path} --dataPath ${public_hdfs_master}:9000/WikiQAProcessed --embeddingFile ${integration_data_dir}/text_data/glove.6B/glove.6B.100d.txt --batchSize 192 --nbEpoch 2
   Log To Console                   begin train inceptionv1
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 3g --executor-memory 50g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.inception.TrainInceptionV1 ${jar_path} -f ${public_hdfs_master}:9000/imagenet-small -b 128 -i 20 --learningRate 0.001
   Log To Console                   begin train inceptionv1 opencv
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 3g --executor-memory 50g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.inception.TrainInceptionV1 ${jar_path} -f ${public_hdfs_master}:9000/imagenet-small -b 128 -i 20 --opencv --learningRate 0.001
   Remove Directory                 /tmp/objectdetection        recursive=True

Spark2.1 Test Suite
   Log To Console                   (1/2) Start the Spark2.1 Test Suite
   Build SparkJar                   spark_2.x
   Set Environment Variable         SPARK_HOME               /opt/work/spark-2.1.0-bin-hadoop2.7
   ${submit}=                       Catenate                 SEPARATOR=/    /opt/work/spark-2.1.0-bin-hadoop2.7/bin    spark-submit
   Run Spark Test                   ${submit}                ${spark_210_3_master}

Yarn Test Suite
   Log To Console                   (2/2) Start the Yarn Test Suite
   Create Directory                 /tmp/objectdetection/output
   Build SparkJar                   spark_2.x
   Set Environment Variable         SPARK_HOME               /opt/work/spark-2.1.0-bin-hadoop2.7
   Set Environment Variable         HADOOP_CONF_DIR          /opt/work/hadoop-2.6.5/etc/hadoop
   Set Environment Variable         http_proxy               ${http_proxy}
   Set Environment Variable         https_proxy              ${https_proxy}
   ${submit}=                       Catenate                 SEPARATOR=/    /opt/work/spark-2.1.0-bin-hadoop2.7/bin    spark-submit
   Log To Console                   begin session recommender
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 5g --executor-memory 10g --class com.intel.analytics.zoo.examples.recommendation.SessionRecExp ${jar_path} --input ${public_hdfs_master}:9000/ecommerce --outputDir ./output/
   Log To Console                   begin text classification
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 5g --executor-memory 10g --class com.intel.analytics.zoo.examples.textclassification.TextClassification ${jar_path} --batchSize 128 --dataPath ${integration_data_dir}/text_data/20news-18828 --embeddingPath ${integration_data_dir}/text_data/glove.6B --partitionNum 8 --nbEpoch 2
   Log To Console                   begin image classification
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 5g --executor-memory 10g --class com.intel.analytics.zoo.examples.imageclassification.Predict ${jar_path} -f ${public_hdfs_master}:9000/kaggle/train_100 --topN 1 --model ${integration_data_dir}/models/analytics-zoo_squeezenet-quantize_imagenet_0.1.0.model --partition 32
   Log To Console                   begin object detection
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 1g --executor-memory 1g --class com.intel.analytics.zoo.examples.objectdetection.inference.Predict ${jar_path} --image ${public_hdfs_master}:9000/kaggle/train_100 --output /tmp/objectdetection/output --modelPath ${integration_data_dir}/models/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model --partition 32
   Log To Console                   begin recommendation wideAndDeep on ml-1m
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 5g --executor-memory 10g --class com.intel.analytics.zoo.examples.recommendation.WideAndDeepExample ${jar_path} --inputDir ${public_hdfs_master}:9000/ml-1m --dataset ml-1m
   Log To Console                   begin recommendation wideAndDeep on census
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 5g --executor-memory 10g --class com.intel.analytics.zoo.examples.recommendation.WideAndDeepExample ${jar_path} --inputDir ${public_hdfs_master}:9000/census --dataset census
   Log To Console                   begin recommendation NCF
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 5g --executor-memory 10g --class com.intel.analytics.zoo.examples.recommendation.NeuralCFexample ${jar_path} --inputDir ${public_hdfs_master}:9000/ml-1m --batchSize 8000
   Log To Console                   begin anomalydetection AnomalyDetection
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 5g --executor-memory 10g --class com.intel.analytics.zoo.examples.anomalydetection.AnomalyDetection ${jar_path} --inputDir ${public_hdfs_master}:9000/NAB/nyc_taxi
   Log To Console                   begin finetune
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 5g --executor-memory 10g --class com.intel.analytics.zoo.examples.nnframes.finetune.ImageFinetune ${jar_path} --modelPath ${integration_data_dir}/models/analytics-zoo_inception-v1_imagenet_0.1.0.model --batchSize 32 --imagePath ${public_hdfs_master}:9000/dogs_cats/samples --nEpochs 2
   Log To Console                   begin imageInference
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 5g --executor-memory 10g --class com.intel.analytics.zoo.examples.nnframes.imageInference.ImageInferenceExample ${jar_path} --caffeDefPath ${integration_data_dir}/models/nnframes/deploy.prototxt --caffeWeightsPath ${integration_data_dir}/models/nnframes/bvlc_googlenet.caffemodel --batchSize 32 --imagePath ${public_hdfs_master}:9000/dogs_cats/samples
   Log To Console                   begin imageTransferLearning
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 5g --executor-memory 10g --class com.intel.analytics.zoo.examples.nnframes.imageTransferLearning.ImageTransferLearning ${jar_path} --caffeDefPath ${integration_data_dir}/models/nnframes/deploy.prototxt --caffeWeightsPath ${integration_data_dir}/models/nnframes/bvlc_googlenet.caffemodel --batchSize 32 --imagePath ${public_hdfs_master}:9000/dogs_cats/samples --nEpochs 20
   Log To Console                   begin QARanker
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 3g --executor-memory 3g --class com.intel.analytics.zoo.examples.qaranker.QARanker ${jar_path} --dataPath ${public_hdfs_master}:9000/WikiQAProcessed --embeddingFile ${integration_data_dir}/text_data/glove.6B/glove.6B.100d.txt --batchSize 192 --nbEpoch 2
   Log To Console                   begin train inceptionv1
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 3g --executor-memory 50g --class com.intel.analytics.zoo.examples.inception.TrainInceptionV1 ${jar_path} -f ${public_hdfs_master}:9000/imagenet-small -b 128 -i 20 --learningRate 0.001
   Log To Console                   begin train inceptionv1 opencv
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 3g --executor-memory 50g --class com.intel.analytics.zoo.examples.inception.TrainInceptionV1 ${jar_path} -f ${public_hdfs_master}:9000/imagenet-small -b 128 -i 20 --opencv --learningRate 0.001
   Remove Environment Variable      http_proxy                https_proxy              PYSPARK_DRIVER_PYTHON            PYSPARK_PYTHON
   Remove Directory                 /tmp/objectdetection        recursive=True
