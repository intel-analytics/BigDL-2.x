export SPARK_HOME=/home/yuhao/workspace/lib/spark-2.1.1-bin-hadoop2.7
export ANALYTICS_ZOO_HOME=/home/yuhao/workspace/github/hhbyyh/analytics-zoo/dist

$ANALYTICS_ZOO_HOME/bin/spark-scala-with-zoo.sh \
--master local[2] \
--driver-memory 12g \
--class com.intel.analytics.zoo.examples.imageclassification.LocalPredict \
/home/yuhao/workspace/github/hhbyyh/analytics-zoo/dist/lib/analytics-zoo-bigdl_0.7.2-spark_2.1.0-0.5.0-SNAPSHOT-jar-with-dependencies.jar \
/home/yuhao/workspace/model/analytics-zoo_resnet-50_imagenet_0.1.0.model \
/home/yuhao/workspace/github/hhbyyh/Test/ZooExample/input \
distributed 8
