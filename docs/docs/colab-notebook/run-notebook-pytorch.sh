echo "#2 start example for pytorch minist"
#timer
start=$(date "+%s")	
#if [ -f analytics-zoo-data/data/MNIST ]	
#then	
#    echo "MNIST already exists"	
#else	
#    wget -nv $FTP_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz -P analytics-zoo-data/data/MNIST/raw	
#    wget -nv $FTP_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz -P analytics-zoo-data/data/MNIST/raw	
#    wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz -P analytics-zoo-data/data/MNIST/raw	
#    wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz -P analytics-zoo-data/data/MNIST/raw	
#fi	
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist
sed -i "s/get_ipython()/#/g"  ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py
sed -i "s/import os/#import os/g" ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py
sed -i "s/import sys/#import sys/g" ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py
sed -i 's/^[^#].*environ*/#&/g' ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py
sed -i 's/^[^#].*__future__ */#&/g' ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py
sed -i "s/_ = (sys.path/#_ = (sys.path/g" ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py
sed -i 's/^[^#].*site-packages*/#&/g' ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py

python ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py

now=$(date "+%s")	
time2=$((now-start))	

