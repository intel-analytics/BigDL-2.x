# run with ipython

echo "#1 start test for tf2_lenet_mnist.ipynb"
#replace '!pip install --pre' to '#pip install --pre', here we test pr with built whl package. In nightly-build job, we test only use "ipython notebook" for pre-release Analytics Zoo
start=$(date "+%s")
{ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/tf2_keras_lenet_mnist
sed -i 's/\!/#/g' $ANALYTICS_ZOO_HOME/docs/docs/colab-notebook/orca/quickstart/tf2_keras_lenet_mnist.py
python ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/tf2_keras_lenet_mnist.py
now=$(date "+%s")
time1=$((now - start))

echo "#2 start test for ncf_dataframe.ipynb"
start=$(date "+%s")
{ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/ncf_dataframe
sed -i 's/\!/#/g' ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/ncf_dataframe.py
python ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/ncf_dataframe.py
now=$(date "+%s")
time2=$((now - start))

echo "#1 tf2_keras_lenet_mnist time used: $time1 seconds"
echo "#2 ncf_dataframe time used: $time2 seconds"
