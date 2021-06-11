import numpy
import os
import random
from model_dien import *

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
SEED = 3
USE_BF16 = False
use_bf16 = USE_BF16

def build_model(model_type, n_uid, n_mid, n_cat, lr, use_bf16, training):
    if model_type == 'DNN':
        model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr,
                          use_bf16=use_bf16,
                          training=training)
    elif model_type == 'PNN':
        model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr,
                          use_bf16=use_bf16,
                          training=training)
    elif model_type == 'Wide':
        model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr,
                               use_bf16=use_bf16,
                               training=training)
    elif model_type == 'DIN':
        model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr,
                          use_bf16=use_bf16,
                          training=training)
    elif model_type == 'DIN-V2-gru-att-gru':
        model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                         ATTENTION_SIZE, lr,
                                         use_bf16=use_bf16, training=training)
    elif model_type == 'DIN-V2-gru-gru-att':
        model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                         ATTENTION_SIZE, lr,
                                         use_bf16=use_bf16, training=training)
    elif model_type == 'DIN-V2-gru-qa-attGru':
        model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                           ATTENTION_SIZE, lr,
                                           use_bf16=use_bf16, training=training)
    elif model_type == 'DIN-V2-gru-vec-attGru':
        model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                            ATTENTION_SIZE, lr,
                                            use_bf16=use_bf16, training=training)
    elif model_type == 'DIEN':
        model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                                ATTENTION_SIZE, lr,
                                                use_bf16=use_bf16, training=training)
    else:
        print("Invalid model_type: %s", model_type)
        sys.exit(1)
    return model

def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    tf.set_random_seed(seed)

def align_input_features(model):
    input_phs = [model.uid_batch_ph, model.mid_his_batch_ph, model.cat_his_batch_ph, model.mask,
                 model.seq_len_ph, model.mid_batch_ph, model.cat_batch_ph]
    feature_cols = ['user', 'item_hist_seq', 'category_hist_seq', 'item_hist_seq_mask',
                    'item_hist_seq_length', 'item', 'category']
    if model.use_negsampling:
        input_phs.extend([model.noclk_mid_batch_ph, model.noclk_cat_batch_ph])
        feature_cols.extend(['neg_item_hist_seq', 'neg_category_hist_seq'])
    return [input_phs, feature_cols]

def load_dien_data(spark, data_dir):
    train_data = spark.read.parquet(data_dir + "/data/train")
    test_data = spark.read.parquet(data_dir + "/data/test")
    userdf = spark.read.parquet(data_dir + "/user_index/*")
    itemdf = spark.read.parquet(data_dir + "/item_index/*")
    catdf = spark.read.parquet(data_dir + "/category_index/*")
    n_uid = userdf.select("id").agg({"id": "max"}).collect()[0][0] + 1
    n_mid = itemdf.select("id").agg({"id": "max"}).collect()[0][0] + 1
    n_cat = catdf.select("id").agg({"id": "max"}).collect()[0][0] + 1
    return train_data, test_data, n_uid, n_mid, n_cat

