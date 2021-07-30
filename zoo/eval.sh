JAR=~/summer_intern/analytics-zoo/zoo/target/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar
EXECUTORS=1
CORES=1
MEM_SIZE=20g
master=local[1]
#ARGS="--dataset=ml-1m--inputDir=/Users/hyunsoo_kim/summer_intern/analytics-zoo/zoo/data/ml-1m"
#export SPARK_HOME=/Users/hyunsoo_kim/summer_intern/spark
#export SPARK_SUBMIT_OPTS=-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=4000
#export JAVA_HOME=/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home

$SPARK_HOME/bin/spark-submit \
  --class com.intel.analytics.zoo.examples.recommendation.WideAndDeepExample \
  --master $master \
  --num-executors $EXECUTORS --executor-cores $CORES --executor-memory $MEM_SIZE \
  --conf "spark.driver.memory=4000m" \
  --conf "spark.driver.cores=8" \
  --conf "spark.caching.dir=/Users/hyunsoo_kim/summer_intern/cache" \
  $JAR --dataset=ml-1m \
	--inputDir=/Users/hyunsoo_kim/summer_intern/analytics-zoo/zoo/data/ml-1m


#export SPARK_SUBMIT_OPTS=-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5000
#export ANALYTICS_ZOO_HOME=/opt/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.10.0-dist-all
#master=local[1]
#args="--inputDir=./data/ml-1m,--dataset=ml-1m"
#${ANALYTICS_ZOO_HOME}/bin/spark-submit-scala-with-zoo.sh \
#    --master $master \
#    --driver-cores 2 \
#    --driver-memory 4000m  \
#    --total-executor-cores 1  \
#    --executor-cores 1 \
#    --executor-memory 4g \
#    --class com.intel.analytics.zoo.examples.recommendation.WideAndDeepExample \
#    --inputDir ./data/ml-1m \
#    --dataset ml-1m