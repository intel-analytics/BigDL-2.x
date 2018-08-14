from keras.models import Model
from keras.layers import Input, Dense, Flatten, LSTM
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import ThresholdedReLU
from keras.layers import Dropout
from keras.callbacks import TensorBoard


class CharCNNZhang(object):
    """
    Class to implement the Character Level Convolutional Neural Network for Text Classification,
    as described in Zhang et al., 2015 (http://arxiv.org/abs/1509.01626)
    """
    def __init__(self, input_size, alphabet_size, embedding_size,
                 conv_layers, fully_connected_layers, num_of_classes,
                 threshold, dropout_p,
                 optimizer='adam', loss='categorical_crossentropy'):
        """
        Initialization for the Character Level CNN model.

        Args:
            input_size (int): Size of input features
            alphabet_size (int): Size of alphabets to create embeddings for
            embedding_size (int): Size of embeddings
            conv_layers (list[list[int]]): List of Convolution layers for model
            fully_connected_layers (list[list[int]]): List of Fully Connected layers for model
            num_of_classes (int): Number of classes in data
            threshold (float): Threshold for Thresholded ReLU activation function
            dropout_p (float): Dropout Probability
            optimizer (str): Training optimizer
            loss (str): Loss function
        """
        self.input_size = input_size
        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size
        self.conv_layers = conv_layers
        self.fully_connected_layers = fully_connected_layers
        self.num_of_classes = num_of_classes
        self.threshold = threshold
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.loss = loss
        self._build_model()  # builds self.model variable

    def _build_model(self):
        """
        Build and compile the Character Level CNN model

        Returns: None

        """
        # Input first layer
        char_inputs = Input(shape=(self.input_size,), name='sent_char_input', dtype='int64')
        # Input second layer
        word_inputs = input(shape=(500, 200), name='sent_word_input', dtype='float')
        # Embedding first layers
        y = Embedding(self.alphabet_size + 1, self.embedding_size, input_length=self.input_size)(char_inputs)
        # Concat two layers
        x = keras.layers.concatenate([y, word_inputs], axis=1)
        # Convolution layers
        for cl in self.conv_layers:
            x = Convolution1D(cl[0], cl[1])(x)
            x = ThresholdedReLU(self.threshold)(x)
            if cl[2] != -1:
                x = MaxPooling1D(cl[2])(x)
        x = Flatten()(x)
        # Fully connected layers
        for fl in self.fully_connected_layers:
            x = Dense(fl)(x)
            x = ThresholdedReLU(self.threshold)(x)
            x = Dropout(self.dropout_p)(x)
        # Output layer
        predictions = Dense(self.num_of_classes, activation='softmax')(x)
        # Build and compile model
        model = Model(inputs=[char_inputs, word_inputs], outputs=predictions)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        self.model = model
        print("CharCNNZhang model built: ")
        self.model.summary()

    def train(self, training_char_inputs, training_word_inputs,
              validation_char_inputs, validation_word_inputs,
              training_labels, validation_labels,
              epochs, batch_size, checkpoint_every=100):
        """
        Training function

        Args:
            training_char_inputs (numpy.ndarray): Training set char inputs
            training_word_inputs (numpy.ndarray): Training set word inputs
            training_labels (numpy.ndarray): Training set labels
            validation_char_inputs (numpy.ndarray): Validation set char inputs
            validation_word_inputs (numpy.ndarray): Validation set word inputs
            validation_labels (numpy.ndarray): Validation set labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            checkpoint_every (int): Interval for logging to Tensorboard

        Returns: None

        """
        # Create callbacks
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=checkpoint_every, batch_size=batch_size,
                                  write_graph=False, write_grads=True, write_images=False,
                                  embeddings_freq=checkpoint_every,
                                  embeddings_layer_names=None)
        # Start training
        print("Training CharCNNZhang model: ")
        self.model.fit([training_char_inputs, training_word_inputs], training_labels,
                       validation_data=([validation_char_inputs, validation_word_inputs], validation_labels),
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=1,
                       callbacks=[tensorboard])

    def test(self, testing_char_inputs, testing_word_input, testing_labels, batch_size):
        """
        Testing function

        Args:
            testing_char_inputs (numpy.ndarray): Testing set char inputs
            testing_word_input (numpy.ndarray): Testing set word inputs
            testing_labels (numpy.ndarray): Testing set labels
            batch_size (int): Batch size

        Returns: None

        """
        # Evaluate inputs
        scores = self.model.evaluate([testing_char_inputs, testing_word_input],
                                     testing_labels, batch_size=batch_size, verbose=1)
        # self.model.predict([testing_char_inputs,testing_word_input], batch_size=batch_size, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
