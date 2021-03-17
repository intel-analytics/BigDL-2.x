# run with ipython

echo "#1 start test for tf_lenet_mnist.ipynb "
#replace '!pip install --pre' to '#pip install --pre', here we test pr with built whl package. In nightly-build job, we test only use "ipython notebook" for pre-release Analytics Zoo
start=$(date "+%s")
sed -i 's/\!/#/g' ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/tf_lenet_mnist.ipynb
ipython ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/tf_lenet_mnist.ipynb
now=$(date "+%s")
time1=$((now - start))

echo "#2 start test for keras_lenet_mnist.ipynb "
start=$(date "+%s")
sed -i 's/\!/#/g' ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/keras_lenet_mnist.ipynb
ipython ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/keras_lenet_mnist.ipynb
now=$(date "+%s")
time2=$((now - start))

echo "#3 start test for zouwu_autots_nyc_taxi.ipynb"
start=$(date "+%s")
sed -i 's/\!/#/g' ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/zouwu/zouwu_autots_nyc_taxi.ipynb
ipython ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/zouwu/zouwu_autots_nyc_taxi.ipynb
now=$(date "+%s")
time3=$((now - start))

echo "#1 tf_lenet_mnist time used: $time1 seconds"
echo "#2 keras_lenet_mnist time used: $time2 seconds"
echo "#3 zouwu_autots_nyc_taxi time used: $time3 seconds"

