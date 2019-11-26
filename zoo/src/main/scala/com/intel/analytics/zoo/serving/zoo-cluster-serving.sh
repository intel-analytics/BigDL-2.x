#!/bin/bash

# --------------config

/opt/work/redis-5.0.5/src/redis-server --port $REDIS_PORT > /opt/work/redis.log &
echo "redis server started, please check log in /opt/work/redis.log"

function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

parse_yaml config.yaml
eval $(parse_yaml config.yaml)

if [ -z "${data_src}" ]; then
    echo "You have not specify data source, using localhost:6379"
    data_src=localhost:6379
fi
if [ -z "${data_shape}" ]; then
    echo "You have not specify data shape, using 3,224,224"
    data_shape=3,224,224
fi



if [ -z "${spark_master}" ]; then
    echo "master of spark cluster not set, using default value local[*]"
    spark_master=local[*]
fi
if [ -z "${spark_driver_memory}" ]; then
    echo "spark driver memory not set, using default value 4g"
    spark_driver_memory=4g
fi
if [ -z "${spark_executor_memory}" ]; then
    echo "spark executor memory not set, using default value 1g"
    spark_executor_memory=1g
fi
if [ -z "${spark_num_executors}" ]; then
    echo "spark num-executors not set, using default value 1"
    spark_num_executors=1
fi
if [ -z "${spark_executor_cores}" ]; then
    echo "spark executor-cores not set, using default value 4"
    spark_executor_cores=4
fi
if [ -z "${spark_total_executor_cores}" ]; then
    echo "spark executor-cores not set, using default value 4"
    spark_total_executor_cores=4
fi

if [ -z "${params_batch_size}" ]; then
    echo "batch size of inference not set, using default value 4"
    params_batch_size=4
fi
if [ -z "${params_mkl_threads}" ]; then    
    params_mkl_threads=${spark_executor_cores}
fi
if [ -z "${params_engine_type}" ]; then    
    params_engine_type=mklblas
fi

if [ -z "${params_top_n}" ]; then  
    echo "top-n not set, using default value 1"  
    params_top_n=1
fi


if [ -z "${model_path}" ]; then
    echo "Please set model path"
    exit 1
fi


${SPARK_HOME}/bin/spark-submit --master ${spark_master} --driver-memory ${spark_driver_memory} --executor-memory ${spark_executor_memory} --num-executors ${spark_num_executors} --executor-cores ${spark_executor_cores} --total-executor-cores ${spark_total_executor_cores} --conf "spark.executor.extraJavaOptions=-Dbigdl.engineType=${params_engine_type}" --jars /home/litchy/pro/spark-redis-2.4.0-SNAPSHOT-jar-with-dependencies.jar --class com.intel.analytics.zoo.serving.ClusterServing /home/litchy/pro/analytics-zoo/dist/lib/analytics-zoo-bigdl_0.10.0-spark_2.4.0-0.7.0-SNAPSHOT-jar-with-dependencies.jar
# -f ${model_path} -b ${params_batch_size} -n ${params_top_n} -r ${data_src} -s ${data_shape}


