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

if [ -z "${model_path}" ]; then
    echo "Please set model path"
    exit 1
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
if [ -z "${params_batch_size}" ]; then
    echo "batch size of inference not set, using default value 4"
    params_batch_size=4
fi
if [ -z "${params_mkl_threads}" ]; then    
    params_mkl_threads=4
fi
if [ -z "${params_engine_type}" ]; then    
    params_engine_type=mkldnn
fi



docker run --net=host -v $(pwd)/${model_path}:/opt/work/resources -e ModelFolder="resources" -e Master=${spark_master} -e DriverMemory=${spark_driver_memory} -e ExecutorMemory=${spark_executor_memory} -e NumExecutors=${spark_num_executors} -e ExecutorCores=${spark_executor_cores} -e BatchSize=${params_batch_size} -e MklThreads=${params_mkl_threads} -e EngineType=${params_engine_type} intelanalytics/analytics-zoo-serving-pub-sub:0.5.1-spark_2.4.0
