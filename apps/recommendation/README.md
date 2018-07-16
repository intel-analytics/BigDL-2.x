# Recommendation
These two notebooks demonstrate how to build neural network recommendation system (Neural Collaborative Filtering, Wide and Deep) with explict feedback using Analytics Zoo and BigDL on Spark. 

## Environment
* Python 2.7/3.5/3.6
* JDK 8
* Spark 1.6.0/2.1.1/2.1.2/2.2.0(This version needs to be same with the version you use to build Analytics Zoo)
(wide_n_deep needs to use Spark version >= 2.1.2)
* Jupyter Notebook 4.1

## Steps to run the notebook
* Download Analytics Zoo and build it.
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project`.
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
MASTER=local[*]
bash ${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 4  \
    --driver-memory 22g  \
    --total-executor-cores 4  \
    --executor-cores 4  \
    --executor-memory 22g
```
