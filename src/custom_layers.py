import tensorflow as tf

class CustomLSTM(tf.keras.layers.Layer):
    def __init__(self, units, inp_shape):
        super(CustomLSTM, self).__init__()
        self.units = units
        self.inp_shape = inp_shape
        self.W = self.add_weight("W", shape=(4, self.units, self.inp_shape))
        self.U = self.add_weight("U", shape=(4, self.units, self.units))

    def call(self, pre_layer, x):
        pre_h, pre_c = tf.unstack(pre_layer)
        
        i_t = tf.nn.sigmoid(tf.matmul(x, tf.transpose(self.W[0])) + tf.matmul(pre_h, tf.transpose(self.U[0])))
        f_t = tf.nn.sigmoid(tf.matmul(x, tf.transpose(self.W[1])) + tf.matmul(pre_h, tf.transpose(self.U[1])))
        o_t = tf.nn.sigmoid(tf.matmul(x, tf.transpose(self.W[2])) + tf.matmul(pre_h, tf.transpose(self.U[2])))
        n_c_t = tf.nn.tanh(tf.matmul(x, tf.transpose(self.W[3])) + tf.matmul(pre_h, tf.transpose(self.U[3])))
        
        c = tf.multiply(f_t, pre_c) + tf.multiply(i_t, n_c_t)
        h = tf.multiply(o_t, tf.nn.tanh(c))
        
        return tf.stack([h, c])
