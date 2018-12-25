from nlp_architect.models.intent_extraction import IntentExtractionModel
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from nlp_architect.contrib.tensorflow.python.keras.layers.crf import CRF
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM, \
#     TimeDistributed, concatenate


class MultiTaskIntentModel(IntentExtractionModel):
    """
    Multi-Task Intent and Slot tagging model (using tf.keras)

    Args:
        use_cudnn (bool, optional): use GPU based model (CUDNNA cells)
    """

    def __init__(self, use_cudnn=False):
        super().__init__()
        self.model = None
        self.word_length = None
        self.num_labels = None
        self.num_intent_labels = None
        self.word_vocab_size = None
        self.char_vocab_size = None
        self.word_emb_dims = None
        self.char_emb_dims = None
        self.char_lstm_dims = None
        self.tagger_lstm_dims = None
        self.dropout = None
        self.use_cudnn = use_cudnn

    def build(self,
              word_length,
              num_labels,
              num_intent_labels,
              word_vocab_size,
              char_vocab_size,
              word_emb_dims=100,
              char_emb_dims=30,
              char_lstm_dims=30,
              tagger_lstm_dims=100,
              dropout=0.2):
        """
        Build a model

        Args:
            word_length (int): max word length (in characters)
            num_labels (int): number of slot labels
            num_intent_labels (int): number of intent classes
            word_vocab_size (int): word vocabulary size
            char_vocab_size (int): character vocabulary size
            word_emb_dims (int, optional): word embedding dimensions
            char_emb_dims (int, optional): character embedding dimensions
            char_lstm_dims (int, optional): character feature LSTM hidden size
            tagger_lstm_dims (int, optional): tagger LSTM hidden size
            dropout (float, optional): dropout rate
        """
        self.word_length = word_length
        self.num_labels = num_labels
        self.num_intent_labels = num_intent_labels
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.word_emb_dims = word_emb_dims
        self.char_emb_dims = char_emb_dims
        self.char_lstm_dims = char_lstm_dims
        self.tagger_lstm_dims = tagger_lstm_dims
        self.dropout = dropout

        words_input = Input(shape=(30,), name='words_input')
        embedding_layer = Embedding(self.word_vocab_size,
                                    self.word_emb_dims, name='word_embedding')
        word_embeddings = embedding_layer(words_input)
        word_embeddings = Dropout(self.dropout)(word_embeddings)

        # create word character input and embeddings layer
        word_chars_input = Input(shape=(30, self.word_length), name='word_chars_input')
        char_embedding_layer = Embedding(self.char_vocab_size, self.char_emb_dims,
                                         input_length=self.word_length, name='char_embedding')
        # # apply embedding to each word
        # char_embeddings = char_embedding_layer(word_chars_input)
        # # feed dense char vectors into BiLSTM
        # char_embeddings = TimeDistributed(
        #     Bidirectional(self._rnn_cell(self.char_lstm_dims)))(char_embeddings)
        # char_embeddings = Dropout(self.dropout)(char_embeddings)

        # first BiLSTM layer (used for intent classification)
        first_bilstm_layer = Bidirectional(
            self._rnn_cell(self.tagger_lstm_dims, return_sequences=True, return_state=True))
        first_lstm_out = first_bilstm_layer(word_embeddings)

        lstm_y_sequence = first_lstm_out[:1][0]  # save y states of the LSTM layer
        states = first_lstm_out[1:]
        hf, _, hb, _ = states  # extract last hidden states
        h_state = concatenate([hf, hb], axis=-1)
        intents = Dense(self.num_intent_labels, activation='softmax',
                        name='intent_classifier_output')(h_state)

        # # create the 2nd feature vectors
        # combined_features = concatenate([lstm_y_sequence, char_embeddings], axis=-1)
        #
        # # 2nd BiLSTM layer for label classification
        # second_bilstm_layer = Bidirectional(self._rnn_cell(self.tagger_lstm_dims,
        #                                                    return_sequences=True))(
        #     combined_features)
        # second_bilstm_layer = Dropout(self.dropout)(second_bilstm_layer)
        # bilstm_out = Dense(self.num_labels)(second_bilstm_layer)
        #
        # # feed BiLSTM vectors into CRF
        # crf = CRF(self.num_labels, name='intent_slot_crf')
        # labels = crf(bilstm_out)

        # compile the model
        model = Model(words_input, intents)

        # define losses and metrics
        loss_f = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']

        model.compile(loss=loss_f,
                      optimizer=Adam(),
                      metrics=metrics)
        self.model = model

    def _rnn_cell(self, units, **kwargs):
        return LSTM(units, **kwargs)
