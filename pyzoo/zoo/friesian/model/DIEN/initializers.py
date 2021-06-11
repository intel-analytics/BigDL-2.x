import tensorflow as tf
ver = tf.__version__
is_v2 = False
if (ver[0] == '2'):
  is_v2 = True
if is_v2:
    from tensorflow.keras.initializers import GlorotNormal, GlorotUniform


def get_embeddings_initializer():
    if is_v2:
        return GlorotUniform()
    else:
        return None

def get_dense_initializer():
    return None

def get_gru_initializer():
    return None
