#export MASTER=local[4]
export FTP_URI=$FTP_URI
#export BIGDL_HOME=$BIGDL_HOME
#export BIGDL_CLASSPATH=$BIGDL_CLASSPATH
#export SPARK_HOME=$SPARK_HOME
#export PYSPARK_ZIP=$PYSPARK_ZIP
#export DL_PYTHON_HOME=$DL_PYTHON_HOME
#export PYTHONPATH=$PYTHONPATH


set -e
mkdir -p result
echo "#1 start example test for dien"
#timer
start=$(date "+%s")
if [ -f data/test.json ]; then
  echo "data/test.json already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/test.json -P data
fi
if [ -f data/test ]; then
  echo "data/test already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/test -P data
fi

python ../example/dien/dien_preprocessing.py \
    --executor_cores 6 \
    --executor_memory 50g \
    --input_meta ./data/test \
    --input_transaction ./data/test.json \
    --output ./result/

now=$(date "+%s")
time1=$((now - start))

echo "#2 start example test for dlrm"
#timer
start=$(date "+%s")
if [ -d data/day_0.parquet ]; then
  echo "data/day_0.parquet already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/day0.tar.gz -P data
  tar -xvzf data/day0.tar.gz -C data
fi
if [ -d data/day_1.parquet ]; then
  echo "data/day_1.parquet already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/day1.tar.gz -P data
  tar -xvzf data/day1.tar.gz -C data
fi

python ../example/dlrm/dlrm_preprocessing.py \
    --cores 6 \
    --memory 50g \
    --days 0-1 \
    --input_folder ./data \
    --output_folder ./result \
    --frequency_limit 15
now=$(date "+%s")
time2=$((now - start))

echo "#3 start example test for wnd"
#timer
start=$(date "+%s")

python ../example/wnd/wnd_preprocessing.py \
    --executor_cores 6 \
    --executor_memory 50g \
    --days 0-1 \
    --input_folder ./data \
    --output_folder ./result \
    --frequency_limit 15 \
    --cross_sizes 10000,10000
now=$(date "+%s")
time3=$((now - start))


rm -rf data
rm -rf result

echo "#1 dien time used: $time1 seconds"
echo "#2 dlrm time used: $time2 seconds"
echo "#3 wnd time used: $time3 seconds"
