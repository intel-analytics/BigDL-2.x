SPARK_OPTS='--master=local[1] --jars ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-VERSION-jar-with-dependencies.jar,./fraud-1.0.1-SNAPSHOT.jar' TOREE_OPTS='--nosparkcontext' jupyter notebook
