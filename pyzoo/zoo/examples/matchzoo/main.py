#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import print_function
import os
import sys
import time
import json
import argparse
import random
random.seed(49999)
import numpy
numpy.random.seed(49999)
import tensorflow
tensorflow.set_random_seed(49999)

from collections import OrderedDict

from zoo.pipeline.api.autograd import *
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import *
from bigdl.keras.converter import WeightsConverter
from test.zoo.pipeline.utils.test_utils import ZooTestCase
import numpy as np

from utils import *
import inputs
import metrics

np.random.seed(1330)

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
sess = tensorflow.Session(config = config)

def load_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    model_config = config['model']['setting']
    model_config.update(config['inputs']['share'])
    sys.path.insert(0, config['model']['model_path'])

    model = import_object(config['model']['model_py'], model_config)
    mo = model.build()
    return mo

def load_keras2_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    model_config = config['keras2model']['setting']
    model_config.update(config['inputs']['share'])
    sys.path.insert(0, config['keras2model']['model_path'])

    model = import_object(config['keras2model']['model_py'], model_config)
    mo = model.build()
    return mo


# def train(config):
#
#     print(json.dumps(config, indent=2), end='\n')
#     # read basic config
#     global_conf = config["global"]
#     optimizer = global_conf['optimizer']
#     optimizer=optimizers.get(optimizer)
#     K.set_value(optimizer.lr, global_conf['learning_rate'])
#     weights_file = str(global_conf['weights_file']) + '.%d'
#     display_interval = int(global_conf['display_interval'])
#     num_iters = int(global_conf['num_iters'])
#     save_weights_iters = int(global_conf['save_weights_iters'])
#
#     # read input config
#     input_conf = config['inputs']
#     share_input_conf = input_conf['share']
#
#
#     # collect embedding
#     if 'embed_path' in share_input_conf:
#         embed_dict = read_embedding(filename=share_input_conf['embed_path'])
#         _PAD_ = share_input_conf['vocab_size'] - 1
#         embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
#         embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
#         share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
#     else:
#         embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
#         share_input_conf['embed'] = embed
#     print('[Embedding] Embedding Load Done.', end='\n')
#
#     # list all input tags and construct tags config
#     input_train_conf = OrderedDict()
#     input_eval_conf = OrderedDict()
#     for tag in input_conf.keys():
#         if 'phase' not in input_conf[tag]:
#             continue
#         if input_conf[tag]['phase'] == 'TRAIN':
#             input_train_conf[tag] = {}
#             input_train_conf[tag].update(share_input_conf)
#             input_train_conf[tag].update(input_conf[tag])
#         elif input_conf[tag]['phase'] == 'EVAL':
#             input_eval_conf[tag] = {}
#             input_eval_conf[tag].update(share_input_conf)
#             input_eval_conf[tag].update(input_conf[tag])
#     print('[Input] Process Input Tags. %s in TRAIN, %s in EVAL.' % (input_train_conf.keys(), input_eval_conf.keys()), end='\n')
#
#     # collect dataset identification
#     dataset = {}
#     for tag in input_conf:
#         if tag != 'share' and input_conf[tag]['phase'] == 'PREDICT':
#             continue
#         if 'text1_corpus' in input_conf[tag]:
#             datapath = input_conf[tag]['text1_corpus']
#             if datapath not in dataset:
#                 dataset[datapath], _ = read_data(datapath)
#         if 'text2_corpus' in input_conf[tag]:
#             datapath = input_conf[tag]['text2_corpus']
#             if datapath not in dataset:
#                 dataset[datapath], _ = read_data(datapath)
#     print('[Dataset] %s Dataset Load Done.' % len(dataset), end='\n')
#
#     # initial data generator
#     train_gen = OrderedDict()
#     eval_gen = OrderedDict()
#
#     for tag, conf in input_train_conf.items():
#         print(conf, end='\n')
#         conf['data1'] = dataset[conf['text1_corpus']]
#         conf['data2'] = dataset[conf['text2_corpus']]
#         generator = inputs.get(conf['input_type'])
#         train_gen[tag] = generator( config = conf )
#
#     for tag, conf in input_eval_conf.items():
#         print(conf, end='\n')
#         conf['data1'] = dataset[conf['text1_corpus']]
#         conf['data2'] = dataset[conf['text2_corpus']]
#         generator = inputs.get(conf['input_type'])
#         eval_gen[tag] = generator( config = conf )
#
#     ######### Load Model #########
#     model = load_model(config)
#
#     loss = []
#     for lobj in config['losses']:
#         if lobj['object_name'] in mz_specialized_losses:
#             loss.append(rank_losses.get(lobj['object_name'])(lobj['object_params']))
#         else:
#             loss.append(rank_losses.get(lobj['object_name']))
#     eval_metrics = OrderedDict()
#     for mobj in config['metrics']:
#         mobj = mobj.lower()
#         if '@' in mobj:
#             mt_key, mt_val = mobj.split('@', 1)
#             eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
#         else:
#             eval_metrics[mobj] = metrics.get(mobj)
#     model.compile(optimizer=optimizer, loss=loss)
#     print('[Model] Model Compile Done.', end='\n')
#
#     for i_e in range(num_iters):
#         for tag, generator in train_gen.items():
#             genfun = generator.get_batch_generator()
#             print('[%s]\t[Train:%s] ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag), end='')
#             history = model.fit_generator(
#                     genfun,
#                     steps_per_epoch = display_interval,
#                     epochs = 1,
#                     shuffle=False,
#                     verbose = 0
#                 ) #callbacks=[eval_map])
#             print('Iter:%d\tloss=%.6f' % (i_e, history.history['loss'][0]), end='\n')
#
#         for tag, generator in eval_gen.items():
#             genfun = generator.get_batch_generator()
#             print('[%s]\t[Eval:%s] ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag), end='')
#             res = dict([[k,0.] for k in eval_metrics.keys()])
#             num_valid = 0
#             for input_data, y_true in genfun:
#                 y_pred = model.predict(input_data)
#                 if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
#                     list_counts = input_data['list_counts']
#                     for k, eval_func in eval_metrics.items():
#                         for lc_idx in range(len(list_counts)-1):
#                             pre = list_counts[lc_idx]
#                             suf = list_counts[lc_idx+1]
#                             res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])
#                     num_valid += len(list_counts) - 1
#                 else:
#                     for k, eval_func in eval_metrics.items():
#                         res[k] += eval_func(y_true = y_true, y_pred = y_pred)
#                     num_valid += 1
#             generator.reset()
#             print('Iter:%d\t%s' % (i_e, '\t'.join(['%s=%f'%(k,v/num_valid) for k, v in res.items()])), end='\n')
#             sys.stdout.flush()
#         if (i_e+1) % save_weights_iters == 0:
#             model.save_weights(weights_file % (i_e+1))


def standardize_input_data(data, names, shapes=None,
                           check_batch_axis=True,
                           exception_prefix=''
                           ):
    if not names:
        if data is not None and hasattr(data, '__len__') and len(data):
            raise ValueError('Error when checking model ' +
                             exception_prefix + ': '
                             'expected no data, but got:', data)
        return []
    if data is None:
        return [None for _ in range(len(names))]

    if isinstance(data, dict):
        try:
            data = [data[x].values if data[x].__class__.__name__ == 'DataFrame' else data[x] for x in names]
        except KeyError as e:
            raise ValueError(
                'No data provided for "' + e.args[0] + '". Need data '
                'for each key in: ' + str(names))
    elif isinstance(data, list):
        if len(names) == 1 and data and isinstance(data[0], (float, int)):
            data = [np.asarray(data)]
        else:
            data = [x.values if x.__class__.__name__ == 'DataFrame' else x for x in data]
    else:
        data = data.values if data.__class__.__name__ == 'DataFrame' else data
        data = [data]
    data = [np.expand_dims(x, 1) if x is not None and x.ndim == 1 else x for x in data]

    if len(data) != len(names):
        if data and hasattr(data[0], 'shape'):
            raise ValueError(
                'Error when checking model ' + exception_prefix +
                ': the list of Numpy arrays that you are passing to '
                'your model is not the size the model expected. '
                'Expected to see ' + str(len(names)) + ' array(s), '
                'but instead got the following list of ' +
                str(len(data)) + ' arrays: ' + str(data)[:200] + '...')
        elif len(names) > 1:
            raise ValueError(
                'Error when checking model ' + exception_prefix +
                ': you are passing a list as input to your model, '
                'but the model expects a list of ' + str(len(names)) +
                ' Numpy arrays instead. The list you passed was: ' +
                str(data)[:200])
        elif len(data) == 1 and not hasattr(data[0], 'shape'):
            raise TypeError(
                'Error when checking model ' + exception_prefix +
                ': data should be a Numpy array, or list/dict of '
                'Numpy arrays. Found: ' + str(data)[:200] + '...')
        elif len(names) == 1:
            data = [np.asarray(data)]

    # Check shapes compatibility.
    if shapes:
        for i in range(len(names)):
            if shapes[i] is not None:
                data_shape = data[i].shape
                shape = shapes[i]
                if data[i].ndim != len(shape):
                    raise ValueError(
                        'Error when checking ' + exception_prefix +
                        ': expected ' + names[i] + ' to have ' +
                        str(len(shape)) + ' dimensions, but got array '
                        'with shape ' + str(data_shape))
                if not check_batch_axis:
                    data_shape = data_shape[1:]
                    shape = shape[1:]
                for dim, ref_dim in zip(data_shape, shape):
                    if ref_dim != dim and ref_dim:
                        raise ValueError(
                            'Error when checking ' + exception_prefix +
                            ': expected ' + names[i] + ' to have shape ' +
                            str(shape) + ' but got array with shape ' +
                            str(data_shape))
    return data


def predict(config):
    ######## Read input config ########

    print(json.dumps(config, indent=2), end='\n')
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.02, 0.02, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print('[Embedding] Embedding Load Done.', end='\n')

    # list all input tags and construct tags config
    input_predict_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'PREDICT':
            input_predict_conf[tag] = {}
            input_predict_conf[tag].update(share_input_conf)
            input_predict_conf[tag].update(input_conf[tag])
    print('[Input] Process Input Tags. %s in PREDICT.' % (input_predict_conf.keys()), end='\n')

    # collect dataset identification
    dataset = {}
    for tag in input_conf:
        if tag == 'share' or input_conf[tag]['phase'] == 'PREDICT':
            if 'text1_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text1_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath)
            if 'text2_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text2_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath)
    print('[Dataset] %s Dataset Load Done.' % len(dataset), end='\n')

    # initial data generator
    predict_gen = OrderedDict()

    for tag, conf in input_predict_conf.items():
        print(conf, end='\n')
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        predict_gen[tag] = generator(
                                    #data1 = dataset[conf['text1_corpus']],
                                    #data2 = dataset[conf['text2_corpus']],
                                     config = conf )

    ######## Read output config ########
    output_conf = config['outputs']

    ######## Load Model ########
    global_conf = config["global"]
    weights_file = str(global_conf['weights_file']) + '.' + str(global_conf['test_weights_iters'])

    model = load_model(config)
    # model.load_weights(weights_file)
    keras2_model = load_keras2_model(config)

    ######## Get and Set Weights ########
    query_embedding = keras2_model.get_layer("query_embedding")
    query_weights = query_embedding.get_weights()
    zquery_weights = WeightsConverter.to_bigdl_weights(query_embedding, query_weights)
    zquery_embedding = [l for l in model.layers if l.name() == "query_embedding"][0]
    zquery_embedding.set_weights(zquery_weights)

    doc_embedding = keras2_model.get_layer("doc_embedding")
    doc_weights = doc_embedding.get_weights()
    zdoc_weights = WeightsConverter.to_bigdl_weights(doc_embedding, doc_weights)
    zdoc_embedding = [l for l in model.layers if l.name() == "doc_embedding"][0]
    zdoc_embedding.set_weights(zdoc_weights)
    #
    # dense = keras2_model.get_layer("dense")
    # dense_weights = dense.get_weights()
    # zdense_weights = WeightsConverter.to_bigdl_weights(dense, dense_weights)
    # zdense = [l for l in model.layers if l.name() == "dense"][0]
    # zdense.set_weights(zdense_weights)

    eval_metrics = OrderedDict()
    for mobj in config['metrics']:
        mobj = mobj.lower()
        if '@' in mobj:
            mt_key, mt_val = mobj.split('@', 1)
            eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
        else:
            eval_metrics[mobj] = metrics.get(mobj)
    res = dict([[k,0.] for k in eval_metrics.keys()])

    batch_size = 20
    query_data = np.random.randint(0, 10000, [batch_size, 10])
    doc_data = np.random.randint(0, 10000, [batch_size, 40])
    input_data = [query_data, doc_data]
    keras2_y_pred = keras2_model.predict(input_data, batch_size=batch_size)
    y_pred = model.predict(input_data, distributed=False)
    equal = np.allclose(y_pred, keras2_y_pred, rtol=1e-5, atol=1e-5)
   # 16
    for tag, generator in predict_gen.items():
        genfun = generator.get_batch_generator()
        print('[%s]\t[Predict] @ %s ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag), end='')
        num_valid = 0
        res_scores = {}
        for input_data, y_true in genfun:
            keras2_y_pred = keras2_model.predict(input_data, batch_size=len(y_true))
            names = ['query', 'doc']
            shapes = [(None, 10), (None, 40)]
            list_input_data = standardize_input_data(input_data, names, shapes, check_batch_axis=False)
           # list_input_data = [data[0:2, :] for data in list_input_data]
            sout = model.get_output_shape()
            fout = model.forward(list_input_data)
            y_pred = model.predict(list_input_data, distributed=False) #y_pred = model.predict(input_data, batch_size=len(y_true) )
            equal = np.allclose(y_pred, keras2_y_pred, rtol=1e-5, atol=1e-5)
            print(equal)

            if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                list_counts = input_data['list_counts']
                for k, eval_func in eval_metrics.items():
                    for lc_idx in range(len(list_counts)-1):
                        pre = list_counts[lc_idx]
                        suf = list_counts[lc_idx+1]
                        res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])

                y_pred = np.squeeze(y_pred)
                for lc_idx in range(len(list_counts)-1):
                    pre = list_counts[lc_idx]
                    suf = list_counts[lc_idx+1]
                    for p, y, t in zip(input_data['ID'][pre:suf], y_pred[pre:suf], y_true[pre:suf]):
                        if p[0] not in res_scores:
                            res_scores[p[0]] = {}
                        res_scores[p[0]][p[1]] = (y, t)

                num_valid += len(list_counts) - 1
            else:
                for k, eval_func in eval_metrics.items():
                    res[k] += eval_func(y_true = y_true, y_pred = y_pred)
                for p, y, t in zip(input_data['ID'], y_pred, y_true):
                    if p[0] not in res_scores:
                        res_scores[p[0]] = {}
                    res_scores[p[0]][p[1]] = (y[1], t[1])
                num_valid += 1
        generator.reset()

        if tag in output_conf:
            if output_conf[tag]['save_format'] == 'TREC':
                with open(output_conf[tag]['save_path'], 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d:d[1][0], reverse=True)
                        for inum,(did, (score, gt)) in enumerate(dinfo):
                            f.write('%s\tQ0\t%s\t%d\t%f\t%s\t%s\n'%(qid, did, inum, score, config['net_name'], gt))
            elif output_conf[tag]['save_format'] == 'TEXTNET':
                with open(output_conf[tag]['save_path'], 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d:d[1][0], reverse=True)
                        for inum,(did, (score, gt)) in enumerate(dinfo):
                            f.write('%s %s %s %s\n'%(gt, qid, did, score))

        print('[Predict] results: ', '\t'.join(['%s=%f'%(k,v/num_valid) for k, v in res.items()]), end='\n')
        sys.stdout.flush()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', help='Phase: Can be train or predict, the default value is train.')
    parser.add_argument('--model_file', default='./models/knrm_wikiqa.config', help='Model_file: MatchZoo model file for the chosen model.')
    args = parser.parse_args()
    model_file =  args.model_file
    with open(model_file, 'r') as f:
        config = json.load(f)
    phase = args.phase
    # if args.phase == 'train':
    #     train(config)
    # elif args.phase == 'predict':
    #     predict(config)
    # else:
    #     print('Phase Error.', end='\n')
    predict(config)
    return

if __name__=='__main__':
    main(sys.argv)
