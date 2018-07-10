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

import zoo.pipeline.api.autograd as A
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras2.layers import *
from zoo.pipeline.api.keras.models import *
from model import BasicModel
from bigdl.keras.converter import WeightsConverter
from zoo.examples.matchzoo.utils.utility import *

class KNRM(BasicModel):
    def __init__(self, config):
        super(KNRM, self).__init__(config)
        self._name = 'KNRM'
        self.check_list = [ 'text1_maxlen', 'kernel_num','sigma','exact_sigma',
                            'embed', 'embed_size', 'vocab_size']
        self.setup(config)
        if not self.check():
            raise TypeError('[KNRM] parameter check wrong')
        print('[KNRM] init done')

    def setup(self, config):
        self.set_default('kernel_num', 11)
        self.set_default('sigma', 0.1)
        self.set_default('exact_sigma', 0.001)
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
        self.config.update(config)


    def build(self):
        def Kernel_layer(mu,sigma):
            def kernel(x):
                return A.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)
            return A.Lambda(lambda x: kernel(x)) #Activation(kernel)

        # batch_size = 2
        # query = Input(name='query', shape=(10,))
        # doc = Input(name='doc', shape=(40,))
        # query_data = np.random.uniform(0, 1, [batch_size, 10])
        # doc_data = np.random.uniform(0, 1, [batch_size, 40])
        # input_data = [query_data, doc_data]
        # embedding1 = Embedding(self.config['vocab_size'], self.config['embed_size'], name="query_embedding")  # trainable=self.config['train_embed'] weights=[self.config['embed']]
        # q_embed = embedding1(query)
        # embedding2 = Embedding(self.config['vocab_size'], self.config['embed_size'], name="doc_embedding")
        # d_embed = embedding2(doc)
        # mm = A.dot(q_embed, d_embed, axes=[2, 2], normalize=True)
        #
        # from keras.layers import Dot
        # from keras.layers import Embedding as KEmbedding
        # from keras.engine.topology import Input as KInput
        # query = KInput(name='query', shape=(10,))
        # doc = KInput(name='doc', shape=(40,))
        # embeddingk = KEmbedding(self.config['vocab_size'], self.config['embed_size'])
        # q_embedk = embeddingk(query)
        # d_embedk = embeddingk(doc)
        # kmm = Dot(axes=[2, 2], normalize=True)([q_embedk, d_embedk])
        # model = Model(input=[query, doc], output=mm)
        # fout = model.forward(input_data)
        # fout

        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        #show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        #show_layer_info('Input', doc)
        # import zoo.pipeline.api.keras.engine.topology as k1
        # mm = k1.merge(query, doc)
        # test_out = Dense(1)(doc)
        #we do not support sharing weights in one embedding
        embedding1 = Embedding(self.config['vocab_size'], self.config['embed_size'], name="query_embedding") #trainable=self.config['train_embed'] weights=[self.config['embed']]
        q_embed = embedding1(query)
        #show_layer_info('Embedding', q_embed)
        embedding2 = Embedding(self.config['vocab_size'], self.config['embed_size'], name="doc_embedding")
        d_embed = embedding2(doc)
        #show_layer_info('Embedding', d_embed)
        mm = A.dot(q_embed, d_embed, axes=[2, 2], normalize=False)
        #mm = A.sum(q_embed * d_embed, axis=2) # Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])
        #show_layer_info('Dot', mm)
        tout_ = mm

        KM = []
        for i in range(self.config['kernel_num']):
            mu = 1. / (self.config['kernel_num'] - 1) + (2. * i) / (self.config['kernel_num'] - 1) - 1.0
            sigma = self.config['sigma']
            if mu > 1.0:
                sigma = self.config['exact_sigma']
                mu = 1.0
            mm_exp = Kernel_layer(mu, sigma)(mm)
            # show_layer_info('Exponent of mm:', mm_exp)
            mm_doc_sum = A.Lambda(lambda x: A.sum(x, 2))(mm_exp) #reduce_sum(x, 2)
            # show_layer_info('Sum of document', mm_doc_sum)
            mm_log = A.Lambda(lambda x: A.log(x + 1.0))(mm_doc_sum)
            # show_layer_info('Logarithm of sum', mm_log)
            mm_sum = A.Lambda(lambda x: A.sum(x, 1))(mm_log)
            # show_layer_info('Sum of all exponent', mm_sum)
            KM.append(mm_sum)
       # # KM = [A.expand_dims(k, 1) for k in KM]
        KMStack = A.stack(KM, 1)
        Phi = A.Lambda(lambda x: x)(KMStack)
        # show_layer_info('Stack', Phi)
        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax', bias_initializer='zero', name="dense")(Phi)
            #we may need to add support for kernel_initializer=RandomUniform
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1, bias_initializer='zero', name="dense")(Phi)
        # show_layer_info('Dense', out_)
        # model = Model(input=[doc], output=[test_out])
        # model = Model(input=[query, doc], output=[out_])
        model = Model(input=[query, doc], output=[tout_])

        # query_embedding = [l for l in model.layers if l.name() == "query_embedding"][0]
        # query_embedding.get_weights()
        return model
