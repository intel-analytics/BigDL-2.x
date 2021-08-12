#!/bin/bash

# chronos test
echo "Chronos test start"

dir=${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/chronos
pytorchFiles=("chronos_nyc_taxi_tsdataset_forecaster" "chronos_minn_traffic_anomaly_detector" "chronos_experimental_autots_nyc_taxi")
index=1

set -e

for f in "${pytorchFiles[@]}"
do
   
	filename="${dir}/${f}"
	echo "#${index} start example for ${f}"
	#timer
	start=$(date "+%s")

	# chronos_nyc_taxi_tsdataset_forecaster data download
	if [ ! -f nyc_taxi.csv ]; then
		wget https://raw.githubusercontent.com/numenta/NAB/v1.0/data/realKnownCause/nyc_taxi.csv
	fi

	# chronos_minn_traffic_anomaly_detector data download
	if [ ! -f speed_7578.csv ]; then
		wget https://raw.githubusercontent.com/numenta/NAB/master/data/realTraffic/speed_7578.csv
	fi

	${ANALYTICS_ZOO_HOME}/apps/ipynb2py-mac.sh ${filename}
	sed -i '' -e "s/get_ipython()/#/g"  ${filename}.py
	sed -i '' -e "s/import os/#import os/g" ${filename}.py
	sed -i '' -e "s/import sys/#import sys/g" ${filename}.py
	sed -i '' -e 's/^[^#].*environ*/#&/g' ${filename}.py
	sed -i '' -e 's/^[^#].*__future__ */#&/g' ${filename}.py
	sed -i '' -e "s/_ = (sys.path/#_ = (sys.path/g" ${filename}.py
	sed -i '' -e "s/.append/#.append/g" ${filename}.py
	sed -i '' -e 's/^[^#].*site-packages*/#&/g' ${filename}.py
	sed -i '' -e 's/version_info/#version_info/g' ${filename}.py
	sed -i '' -e 's/python_version/#python_version/g' ${filename}.py
	sed -i '' -e 's/exit()/#exit()/g' ${filename}.py
	sed -i '' -e 's/plt.show()/#plt.show()/g' ${filename}.py

	python ${filename}.py

	now=$(date "+%s")
	time=$((now-start))
	echo "Complete #${index} with time ${time} seconds"	
	index=$((index+1))
done

# orca test
echo "orca test start"

dir=${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart
pytorchFiles=("pytorch_lenet_mnist_data_creator_func" "pytorch_lenet_mnist" "pytorch_distributed_lenet_mnist" "autoestimator_pytorch_lenet_mnist" "autoxgboost_regressor_sklearn_boston")
index=1

set -e

for f in "${pytorchFiles[@]}"
do
   
	filename="${dir}/${f}"
	echo "#${index} start example for ${f}"
	#timer
	start=$(date "+%s")

	${ANALYTICS_ZOO_HOME}/apps/ipynb2py-mac.sh ${filename}
	sed -i '' -e "s/get_ipython()/#/g"  ${filename}.py
	sed -i '' -e "s/import os/#import os/g" ${filename}.py
	sed -i '' -e "s/import sys/#import sys/g" ${filename}.py
	sed -i '' -e 's/^[^#].*environ*/#&/g' ${filename}.py
	sed -i '' -e 's/^[^#].*__future__ */#&/g' ${filename}.py
	sed -i '' -e "s/_ = (sys.path/#_ = (sys.path/g" ${filename}.py
	sed -i '' -e "s/.append/#.append/g" ${filename}.py
	sed -i '' -e 's/^[^#].*site-packages*/#&/g' ${filename}.py
	sed -i '' -e 's/version_info/#version_info/g' ${filename}.py
	sed -i '' -e 's/python_version/#python_version/g' ${filename}.py
	sed -i '' -e 's/epochs=30/epochs=1/g' ${filename}.py

	python ${filename}.py

	now=$(date "+%s")
	time=$((now-start))
	echo "Complete #${index} with time ${time} seconds"	
	index=$((index+1))
done

# orca example test
echo "orca examples test start"

dir=${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/examples
pytorchFiles=("fashion_mnist_bigdl" "super_resolution")
index=1

set -e

if [ ! -f BSDS300-images.tgz ]; then
  wget $FTP_URI/analytics-zoo-data/BSDS300-images.tgz
fi
if [ ! -d sr_dataset/BSDS300/images ]; then
  mkdir sr_dataset
  tar -xzf BSDS300-images.tgz -C sr_dataset
fi

for f in "${pytorchFiles[@]}"
do
   
	filename="${dir}/${f}"
	echo "#${index} start example for ${f}"
	#timer
	start=$(date "+%s")

	${ANALYTICS_ZOO_HOME}/apps/ipynb2py-mac.sh ${filename}
	sed -i '' -e "s/get_ipython()/#/g"  ${filename}.py
	sed -i '' -e "s/import os/#import os/g" ${filename}.py
	sed -i '' -e "s/import sys/#import sys/g" ${filename}.py
	sed -i '' -e 's/^[^#].*environ*/#&/g' ${filename}.py
	sed -i '' -e 's/^[^#].*__future__ */#&/g' ${filename}.py
	sed -i '' -e "s/_ = (sys.path/#_ = (sys.path/g" ${filename}.py
	sed -i '' -e "s/.append/#.append/g" ${filename}.py
	sed -i '' -e 's/^[^#].*site-packages*/#&/g' ${filename}.py
	sed -i '' -e 's/version_info/#version_info/g' ${filename}.py
	sed -i '' -e 's/python_version/#python_version/g' ${filename}.py
	sed -i '' -e 's/batch_size = 32/batch_size = 320/g' ${filename}.py
	sed -i '' -e 's/epochs = 30/epochs = 1/g' ${filename}.py

	python ${filename}.py

	now=$(date "+%s")
	time=$((now-start))
	echo "Complete #${index} with time ${time} seconds"	
	index=$((index+1))
done
