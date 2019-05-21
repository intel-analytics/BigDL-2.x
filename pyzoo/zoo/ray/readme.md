# How to run Ray On Yarn

1) You should install Conda first and create a conda-env named "ray36"

2) Install some essential dependencies on the conda env
pip install pyspark==2.4
pip install BigDL
pip install ray[debug]
pip install conda-pack
pip install psutil
pip install aiohttp
# You can also download and set environment variable(JAVA_HOME) manually.
conda install -c anaconda openjdk=8.0.152

4) start jupyter notebook and run the following code:
Please refer: path/to/analytics-zoo/pyzoo/test/zoo/ray/integration/yarn_ps_gondolin.py

