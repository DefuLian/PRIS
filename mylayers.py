import tensorflow as tf
from utils import Misc
import numpy as np
class RCEncoding(tf.keras.layers.Layer):
    # att_mode: additive  v^T tanh(Wx + U q)
    # att_mode: dot       x^T q
    # att_mode: scale_dot x^T q / sqrt(d)
    # att_mode: bilinear  x^T W q
    # att_mode: euclidean exp - 0.5|x - q|
    # att_mode: kd 
    # rnn_mode: res  x - f(x)
    # rnn_mode: rnn  W x - U f(x) = [W, U] * [x; -f(x)]
    # rnn_mode: none x
    def __init__(self, num_codewords, att_mode, rnn_mode, **kwargs):
        self.num_codewords = num_codewords
        self.T = 0.9
        self.prob = 0.5
        self.att_mode = att_mode
        self.rnn_mode = rnn_mode
        super(RCEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dim = input_shape.as_list()[-1]
        self.codebooks = [None] * len(self.num_codewords)

        if self.att_mode == 'additive':
            self.W_att_c = [None] * len(self.num_codewords)
            self.W_att_x = [None] * len(self.num_codewords)
            self.v_att = [None] * len(self.num_codewords)
        elif self.att_mode == 'bilinear':
            self.W_att = [None] * len(self.num_codewords)
        elif self.att_mode == 'kd':
            self.W_att = [None] * len(self.num_codewords)
        
        for i in range(len(self.num_codewords)):
            self.codebooks[i] = self.add_weight(name='codebook_%d' % i,
                                                shape=(self.num_codewords[i], self.dim),
                                                initializer=tf.keras.initializers.glorot_normal(),
                                                trainable=True)
            if self.att_mode == 'additive':
                self.W_att_c[i] = self.add_weight(name='weight_att_additive_c_%d' % i,
                                                  shape=(self.dim, self.dim),
                                                  initializer=tf.keras.initializers.glorot_normal(),
                                                  trainable=True)
                self.W_att_x[i] = self.add_weight(name='weight_att_additive_x_%d' % i,
                                                  shape=(self.dim, self.dim),
                                                  initializer=tf.keras.initializers.glorot_normal(),
                                                  trainable=True)
                self.v_att[i] = self.add_weight(name='v_att_additive_%d' % i,
                                                shape=(self.dim, ),
                                                initializer=tf.keras.initializers.glorot_normal(),
                                                trainable=True)
            elif self.att_mode == 'bilinear':
                self.W_att[i] = self.add_weight(name='weight_att_bilinear_%d' % i,
                                                shape=(self.dim, self.dim),
                                                initializer=tf.keras.initializers.glorot_normal(),
                                                trainable=True)
            elif self.att_mode == 'kd':
                self.W_att[i] = self.add_weight(name='weight_att_kd_%d' % i,
                                                shape=(self.dim, self.num_codewords[i]),
                                                initializer=tf.keras.initializers.glorot_normal(),
                                                trainable=True)
        if len(self.num_codewords) > 1:
            if self.rnn_mode == 'rnn':
                self.W_rnn = self.add_weight(name='weight_rnn',
                                             shape=(2*self.dim, self.dim),
                                             initializer=tf.keras.initializers.glorot_normal(),
                                             trainable=True)
            elif self.rnn_mode == 'res':
                self.W_rnn = self.add_weight(name='weight_rnn',
                                             shape=(self.dim, self.dim),
                                             initializer=tf.keras.initializers.glorot_normal(),
                                             trainable=True)
        super(RCEncoding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        output = [None] * len(self.num_codewords)
        indexes = [None] * len(self.num_codewords)
        weights = [None] * len(self.num_codewords)
        x = inputs
        for i in range(len(self.num_codewords)):
            if i>0:
                if self.rnn_mode == 'rnn':
                    x = tf.tensordot(tf.concat((x, output[i-1]), -1), self.W_rnn, axes=[-1, 0])
                elif self.rnn_mode == 'res':
                    #x = x - output[i-1]
                    x = x - tf.tensordot(output[i-1], self.W_rnn, axes=[-1, 0])
            if self.att_mode == 'additive':
                x_ = tf.tensordot(x, self.W_att_x[i], axes=[-1, 0])
                c_ = tf.tensordot(self.codebooks[i], self.W_att_c[i], axes=[-1, 0])
                xc = tf.tanh(tf.expand_dims(x_, -2) + c_)
                att_logit = tf.tensordot(xc, self.v_att[i], axes=[-1, 0])
            elif self.att_mode == 'dot':
                att_logit = tf.tensordot(x, self.codebooks[i], axes=[-1, -1])
            elif self.att_mode == 'scale_dot':
                att_logit = tf.tensordot(x, self.codebooks[i], axes=[-1, -1]) / tf.sqrt(tf.cast(self.dim,dtype=tf.float32))
            elif self.att_mode == 'bilinear':
                x_ = tf.tensordot(x, self.W_att[i], axes=[-1, 0])
                c = tf.tensordot(self.codebooks[i], self.W_att[i], axes=[-1, 0])
                att_logit = tf.tensordot(x_, c, axes=[-1, -1])
            elif self.att_mode == 'euclidean':
                x_ = tf.expand_dims(x, -2)
                dist = tf.reduce_sum(tf.squared_difference(x_, self.codebooks[i]), axis=-1, keepdims=False)
                att_logit = tf.exp(-dist / 2)
            elif self.att_mode == 'kd':
                att_logit = tf.tensordot(x, self.W_att[i], axes=[-1, 0])
            else:
                raise ValueError('Not supported attention mode')

            weight_soft = Misc.gumbel_softmax(att_logit, self.T, hard=False)
            weight_hard = Misc.gumbel_softmax(att_logit, self.T, hard=True)
            indexes[i] = tf.argmax(att_logit, -1)
            weights[i] = weight_soft
            weight = tf.stop_gradient(weight_hard - weight_soft) + weight_soft
            #weight = tf.keras.backend.in_train_phase(weight_soft, weight_hard)
            #weight = tf.keras.backend.in_train_phase(weight_, weight_hard)
            output[i] = tf.tensordot(weight, self.codebooks[i], axes=[-1, 0])
        #train_output = tf.nn.dropout(tf.add_n(output), rate=1-self.prob) + tf.nn.dropout(inputs, rate=self.prob)
        #test_output = tf.add_n(output)
        #return tf.keras.backend.in_train_phase(train_output,  test_output), tf.concat(indexes, -1)
        return tf.add_n(output), tf.concat(indexes, -1)

    def compute_output_shape(self, input_shape):
        output_shape = []
        output_shape.extend(input_shape[:-1])
        output_shape.append(len(self.num_codewords))
        return input_shape, output_shape


class EuclideanDistillation(tf.keras.layers.Layer):
    def __init__(self, coef=1, **kwargs):
        self.coef = coef
        super(EuclideanDistillation, self).__init__(**kwargs)
    def call(self, inputs):
        x, x_ = inputs
        self.add_loss(self.coef*tf.reduce_mean((x - x_)**2))
        return x_

class DistilledDot(tf.keras.layers.Layer):
    def __init__(self, ed_coef=1, id_coef=0, **kwargs):
        self.ed_coef = ed_coef
        self.id_coef = id_coef
        self.momentum = 0.99
        super(DistilledDot, self).__init__(**kwargs)
    def build(self, input_shape):
        dim = input_shape[0].as_list()[-1]
        self.moving_user = self.add_weight(name='moving_user', shape=(dim, dim), trainable=False, initializer='zeros')
        self.moving_user_ = self.add_weight(name='moving_user_', shape=(dim, dim), trainable=False, initializer='zeros')
    def call(self, inputs):
        user, user_, item, item_ = inputs
        if self.ed_coef>0:
            self.add_loss(self.ed_coef * tf.reduce_mean((item - item_)**2))
        if self.id_coef>0:
            x = tf.tensordot(user, user, axes=[[0,1], [0,1]])
            x_ = tf.tensordot(user_, user_, axes=[[0,1], [0,1]])
            self.add_update(tf.keras.backend.moving_average_update(self.moving_user, x, self.momentum))
            self.add_update(tf.keras.backend.moving_average_update(self.moving_user_, x_, self.momentum))
        return tf.tensordot(user_, item_, axes=-1)

