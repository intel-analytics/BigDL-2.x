
import tensorflow as tf
meta_file="/home/ding/pretrain_model/bert/google/uncased_L-12_H-768_A-12/bert_model.ckpt.meta"
checkpoint="/home/ding/pretrain_model/bert/google/uncased_L-12_H-768_A-12/bert_model.ckpt"
with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
        saver.restore(sess, checkpoint)
        tvars = tf.trainable_variables()
        tvars_vals = sess.run(tvars)

        # np.savetxt("/tmp/numpy/t.out", tvars_vals[1])
        # t=np.loadtxt("/tmp/numpy/t.out", dtype = np.float64)

        for var, val in zip(tvars, tvars_vals):
            # print(var.name, val)  # Prints the name of the variable alongside its value.
            path = "/tmp/numpy/"
            path += var.name.replace(":", "_").replace("/", "_").encode('ascii', 'ignore')
            path += ".out"
            np.savetxt(path, val)
