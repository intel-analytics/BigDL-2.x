#!/usr/bin/env bash
clear_up() {
	echo "Clearing up environment. Uninstalling analytics-zoo"
	pip uninstall -y analytics-zoo
	pip uninstall -y bigdl
	pip uninstall -y pyspark
}

set -e

names=(ray/quickstart/ray_sharded_parameter_server)
len=${#names[@]}
runtime=()
for (( i=0; i<$len; ++i));
do
	name=${names[$i]}
	echo "#$((i+1)) start test for $name.ipynb"
	start=$(date "+%s")
	${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/$name
	sed -i '/get_ipython/s/^/#/' ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/$name.py
	python ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/$name.py

	exit_status=$?
	if [ $exit_status -ne 0 ]; then
	  clear_up
	  echo "$name failed"
	  exit $exit_status
	fi

	now=$(date "+%s")
	runtime+=$((now - start))

	rm ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/$name.py
done

for (( i=0; i<$len; ++i));
do
	echo "#$((i+1)) ${names[$i]} time used: ${runtime[$i]} seconds"
done	
