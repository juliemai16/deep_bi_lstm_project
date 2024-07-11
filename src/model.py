import tensorflow as tf
from src.custom_layers import CustomLSTM

class DeepBidirectionalLSTM(tf.keras.Model):
    def __init__(self, units, embedding_size, vocab_size, input_length, num_layers):
        super(DeepBidirectionalLSTM, self).__init__()
        self.num_layers = num_layers
        self.input_length = input_length
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=input_length)

        self.forward_layers = [CustomLSTM(units, embedding_size if i == 0 else units) for i in range(num_layers)]
        self.backward_layers = [CustomLSTM(units, embedding_size if i == 0 else units) for i in range(num_layers)]

        self.classification_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_shape=(units * 2,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, sentence):
        batch_size = tf.shape(sentence)[0]

        pre_layer_fwd = tf.stack([tf.zeros([batch_size, self.units]), tf.zeros([batch_size, self.units])])
        pre_layer_bwd = tf.stack([tf.zeros([batch_size, self.units]), tf.zeros([batch_size, self.units])])

        embedded_sentence = self.embedding(sentence)

        for i in range(self.input_length):
            word_fwd = embedded_sentence[:, i, :]
            pre_layer_fwd = self.forward_layers[0](pre_layer_fwd, word_fwd)
            
            word_bwd = embedded_sentence[:, self.input_length - i - 1, :]
            pre_layer_bwd = self.backward_layers[0](pre_layer_bwd, word_bwd)

        h_fwd, _ = tf.unstack(pre_layer_fwd)
        h_bwd, _ = tf.unstack(pre_layer_bwd)

        h_concat = tf.concat([h_fwd, h_bwd], axis=1)

        return self.classification_model(h_concat)
