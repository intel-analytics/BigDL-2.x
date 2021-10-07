import tensorflow as tf

from zoo.orca import init_orca_context, OrcaContext
from zoo.orca.learn.tf2.estimator import Estimator

conf = {"spark.network.timeout": "10000000",
        "spark.sql.broadcastTimeout": "7200",
        "spark.sql.shuffle.partitions": "2000",
        "spark.locality.wait": "0s",
        "spark.sql.hive.filesourcePartitionFileCacheSize": "4096000000",
        "spark.sql.crossJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryo.unsafe": "true",
        "spark.kryoserializer.buffer.max": "1024m",
        "spark.task.cpus": "1",
        "spark.executor.heartbeatInterval": "200s",
        "spark.driver.maxResultSize": "40G",
        "spark.driver.memoryOverhead": "5G",
        "spark.executor.memoryOverhead": "5g"}


def create_model(config):
    user = tf.keras.layers.Input(shape=[1])
    item = tf.keras.layers.Input(shape=[1])

    feat = tf.keras.layers.concatenate([user, item], axis=1)
    predictions = tf.keras.layers.Dense(2, activation='softmax')(feat)

    model = tf.keras.models.Model(inputs=[user, item], outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# model = create_model()
sc = init_orca_context(cluster_mode="yarn",
                       cores=2,
                       num_nodes=2,
                       driver_cores=2,
                       driver_memory="10g",
                       conf=conf)
spark = OrcaContext.get_spark_session()
file_path = "/opt/work/jwang/analytics-zoo-jennie/pyzoo/test/zoo/resources/orca/learn/ncf.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)
from pyspark.sql.functions import array

df = df.withColumn('user', array('user')) \
    .withColumn('item', array('item'))

est = Estimator.from_keras(model_creator=create_model,
                           backend="spark")
# est.set_tensorboard("hdfs://172.16.0.105:8020/user/root/jwang/log_dir", app_name="test")
res = est.fit(data=df,
              batch_size=8,
              epochs=4,
              steps_per_epoch=4,
              feature_cols=['user', 'item'],
              label_cols=['label'],
              validation_data=df,
              validation_steps=2)

est.save("hdfs://172.16.0.105:8020/user/root/jwang/test.model")
est.load("hdfs://172.16.0.105:8020/user/root/jwang/test.model")
est.save_weights("hdfs://172.16.0.105:8020/user/root/jwang/test.h5")
est.load_weights("hdfs://172.16.0.105:8020/user/root/jwang/test.h5")

result = est.evaluate(data=df, batch_size=8, num_steps=2,
                      feature_cols=['user', 'item'],
                      label_cols=['label']
                      )

print("result: ", result)

pred_df = est.predict(data=df, batch_size=8, feature_cols=['user', 'item'])

pred_df.show()
