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
   [Arguments]       ${spark_version}
   ${build}=         Catenate                        SEPARATOR=/    ${curdir}    make-dist.sh
   Log To Console    ${spark_version}
   Log To Console    start to build jar
   Log To Console    make-dist.sh -P ${spark_version} ...
   Run               ${build} -P ${spark_version}
   Remove File       ${jar_path}
   Move File         zoo/target/analytics-zoo-${version}-jar-with-dependencies.jar    ${jar_path}
   Log To Console    build jar finished

DownLoad Input
   ${hadoop}=                       Catenate       SEPARATOR=/    /opt/work/hadoop-2.6.5/bin    hadoop
   Run                              ${hadoop} fs -get ${mnist_data_source} ./
   Log To Console                   got mnist data !!
   Run                              ${hadoop} fs -get ${cifar_data_source} ./
   Log To Console                   got cifar data !!
   Run                              ${hadoop} fs -get ${public_hdfs_master}:9000/text_data /tmp/
   Run                              tar -zxvf /tmp/text_data/20news-18828.tar.gz -C /tmp/text_data
   Log To Console                   got textclassifier data !!
   Set Environment Variable         http_proxy                                                  ${http_proxy}
   Set Environment Variable         https_proxy                                                 ${https_proxy}
   Run                              wget ${tiny_shakespeare}
   Set Environment Variable         LANG                                                        en_US.UTF-8
   Run                              head -n 8000 input.txt > val.txt
   Run                              tail -n +8000 input.txt > train.txt
   Run                              wget ${simple_example}
   Run                              tar -zxvf simple-examples.tgz
   Log To Console                   got examples data!!
   Create Directory                 model
   Create Directory                 models

   Run                              wget https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_squeezenet-quantize_imagenet_0.1.0 -P /tmp/imageclassification/
   Log To Console                   got image data!!
   Run                              wget https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model -P /tmp/objectdetection/
   Log To Console                   got image data!!
   Create Directory                 /tmp/objectdetection/output
   Remove Environment Variable      http_proxy                  https_proxy                     LANG

Remove Input
   Remove Directory                 model                       recursive=True
   Remove Directory                 models                      recursive=True
   Remove Directory                 mnist                       recursive=True
   Remove File                      input.txt
   Remove Directory                 simple-examples             recursive=True
   Remove File                      simple-examples.tgz
   Remove Directory                 /tmp/text-data              recursive=True
   Remove Directory                 /tmp/imageclassification    recursive=True
   Remove Directory                 /tmp/objectdetection        recursive=True

Run Spark Test
   [Arguments]                      ${submit}                   ${spark_master}
   DownLoad Input
   Log To Console                   begin text classification
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 5g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.textclassification.TextClassification ${jar_path} --batchSize 128 --baseDir /tmp/text_data --partitionNum 32 --nbEpoch 2
   Log To Console                   begin image classification
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 5g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.imageclassification.Predict ${jar_path} -f ${public_hdfs_master}:9000/kaggle/train_100 --topN 1 --model /tmp/imageclassification/analytics-zoo_squeezenet-quantize_imagenet_0.1.0 --partition 32
   Log To Console                   begin object detection
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 1g --executor-memory 1g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.objectdetection.Predict ${jar_path} --image ${public_hdfs_master}:9000/kaggle/train_100 --output /tmp/objectdetection/output --model /tmp/objectdetection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model --partition 32
   Log To Console                   begin recommendation wideAndDeep
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 5g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.recommendation.WideAndDeepExample ${jar_path} --inputDir ${public_hdfs_master}:9000/ml-1m
   Log To Console                   begin recommendation NCF
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 5g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.zoo.examples.recommendation.NeuralCFexample ${jar_path} --inputDir ${public_hdfs_master}:9000/ml-1m
   Remove Input

Spark2.1 Test Suite
   Log To Console                   (1/2) Start the Spark2.1 Test Suite
   Build SparkJar                   spark_2.x
   Set Environment Variable         SPARK_HOME               /opt/work/spark-2.1.0-bin-hadoop2.7
   ${submit}=                       Catenate                 SEPARATOR=/    /opt/work/spark-2.1.0-bin-hadoop2.7/bin    spark-submit
   Run Spark Test                   ${submit}                ${spark_210_3_master}

Yarn Test Suite
   Log To Console                   (2/2) Start the Yarn Test Suite
   DownLoad Input
   Build SparkJar                   spark_2.x
   Set Environment Variable         SPARK_HOME               /opt/work/spark-2.1.0-bin-hadoop2.7
   Set Environment Variable         http_proxy               ${http_proxy}
   Set Environment Variable         https_proxy              ${https_proxy}
   ${submit}=                       Catenate                 SEPARATOR=/    /opt/work/spark-2.1.0-bin-hadoop2.7/bin    spark-submit
   Log To Console                   begin text classification
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 5g --executor-memory 10g --class com.intel.analytics.zoo.examples.textclassification.TextClassification ${jar_path} --batchSize 128 --baseDir /tmp/text_data --partitionNum 8 --nbEpoch 2
   Log To Console                   begin image classification
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 5g --executor-memory 5g --class com.intel.analytics.zoo.examples.imageclassification.Predict ${jar_path} -f ${public_hdfs_master}:9000/kaggle/train_100 --topN 1 --model /tmp/imageclassification/analytics-zoo_squeezenet-quantize_imagenet_0.1.0 --partition 32
   Log To Console                   begin object detection
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 1g --executor-memory 1g --class com.intel.analytics.zoo.examples.objectdetection.Predict ${jar_path} --image ${public_hdfs_master}:9000/kaggle/train_100 --output /tmp/objectdetection/output --model /tmp/objectdetection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model --partition 32
   Log To Console                   begin recommendation wideAndDeep
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 5g --executor-memory 10g --class com.intel.analytics.zoo.examples.recommendation.WideAndDeepExample ${jar_path} --inputDir ${public_hdfs_master}:9000/ml-1m
   Log To Console                   begin recommendation NCF
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 8 --num-executors 4 --driver-memory 5g --executor-memory 10g --class com.intel.analytics.zoo.examples.recommendation.NeuralCFexample ${jar_path} --inputDir ${public_hdfs_master}:9000/ml-1m
   Remove Environment Variable      http_proxy                https_proxy              PYSPARK_DRIVER_PYTHON            PYSPARK_PYTHON
   Remove Input
