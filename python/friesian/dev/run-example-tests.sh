export FTP_URI=$FTP_URI


set -e
mkdir -p result
echo "#1 start example test for dien preprocessing"
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

echo "#2 start example test for dlrm preprocessing"
#timer
start=$(date "+%s")
if [ -d data/day_0.parquet ]; then
  echo "data/day_0.parquet already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/day0.tar.gz -P data
  tar -xvzf data/day0.tar.gz -C data
fi

python ../example/dlrm/dlrm_preprocessing.py \
    --cores 6 \
    --memory 50g \
    --days 0-0 \
    --input_folder ./data \
    --output_folder ./result \
    --frequency_limit 15
now=$(date "+%s")
time2=$((now - start))

echo "#3 start example test for wnd preprocessing"
#timer
start=$(date "+%s")
if [ -d data/day_1.parquet ]; then
  echo "data/day_1.parquet already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/day1.tar.gz -P data
  tar -xvzf data/day1.tar.gz -C data
fi
python ../example/wnd/wnd_preprocessing.py \
    --executor_cores 6 \
    --executor_memory 50g \
    --days 1-1 \
    --input_folder ./data \
    --output_folder ./result \
    --frequency_limit 15 \
    --cross_sizes 10000,10000
now=$(date "+%s")
time3=$((now - start))


rm -rf data
rm -rf result

echo "#1 dien preprocessing time used: $time1 seconds"
echo "#2 dlrm preprocessing time used: $time2 seconds"
echo "#3 wnd preprocessing time used: $time3 seconds"
