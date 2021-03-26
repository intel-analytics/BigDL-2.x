#!/usr/bin/env bash
clear_up() {
  echo "Clearing up environment. Uninstalling analytics-zoo"
  pip uninstall -y analytics-zoo
  pip uninstall -y bigdl
  pip uninstall -y pyspark
}

set -e 

echo "#1 start test for tf_lenet_mnist.ipynb "
#replace '!pip install --pre' to '#pip install --pre', here we test pr with built whl package. In nightly-build job, we test only use "ipython notebook" for pre-release Analytics Zoo
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/tf_lenet_mnist
sed -i '/get_ipython/s/^/#/' ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/tf_lenet_mnist.py
python ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/tf_lenet_mnist.py

exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "tf_lenet_mnist failed"
  exit $exit_status
fi

now=$(date "+%s")
time1=$((now - start))

echo "#2 start test for keras_lenet_mnist.ipynb "
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/keras_lenet_mnist
sed -i '/get_ipython/s/^/#/' ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/keras_lenet_mnist.py
python ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/keras_lenet_mnist.py

exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "keras_lenet_mnist failed"
  exit $exit_status
fi

now=$(date "+%s")
time2=$((now - start))

echo "#1 tf_lenet_mnist time used: $time1 seconds"
echo "#2 keras_lenet_mnist time used: $time2 seconds"
