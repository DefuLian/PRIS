import tensorflow as tf
import time
from utils import Eval
import numpy as np
from sampler import *
from mylayers import RCEncoding, EuclideanDistillation


samplers = [SamplerModel, PopularSamplerModel, ClusterSamplerModel, ClusterPopularSamplerModel, ExactSamplerModel]


class EvaluateCallback(tf.keras.callbacks.Callback):

    def __init__(self, round_):
        self.round = round_
        super(EvaluateCallback, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.starttime
        print('Epoch={} - {}s - loss={:.4f}'.format(self.round + epoch + 1, int(elapsed), logs['loss']))


def compute_loss(pred, prob, weighted):
    if weighted:
      importance = tf.nn.softmax(tf.negative(pred) - tf.log(prob))
    else:
      importance = tf.nn.softmax(tf.ones_like(pred))
    weight_loss = tf.multiply(importance, tf.negative(tf.log_sigmoid(pred)))
    loss = tf.reduce_sum(weight_loss, -1, keepdims=True)
    return loss


def identity_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred - 0 * y_true)


class PerisModel:
    def __init__(self, config):
        user_id = tf.keras.Input(shape=(1,), name='user_id')
        pos_id = tf.keras.Input(shape=(1,), name='pos_id')
        neg_id = tf.keras.Input(shape=(config.neg_num,), name='neg_id')
        neg_prob = tf.keras.Input(shape=(config.neg_num,), name='neg_prob', dtype='float32')
        item_embed_layer = tf.keras.layers.Embedding(config.num_item, config.d, name='item_embedding',
                                                     embeddings_initializer=tf.keras.initializers.glorot_normal(),
                                                     activity_regularizer=tf.keras.regularizers.l2(config.coef / config.batch_size))
        user_embed = tf.keras.layers.Embedding(config.num_user, config.d, name='user_embedding',
                                               embeddings_initializer=tf.keras.initializers.glorot_normal(),
                                               activity_regularizer=tf.keras.regularizers.l2(config.coef / config.batch_size))(user_id)

        pos_item_embed = item_embed_layer(pos_id)
        neg_item_embed = item_embed_layer(neg_id)
        pos_score = tf.keras.layers.dot([user_embed, pos_item_embed], axes=-1)
        neg_score = tf.keras.layers.dot([user_embed, neg_item_embed], axes=-1)
        ruij = tf.keras.layers.Flatten()(tf.keras.layers.subtract([pos_score, neg_score]))
        loss = tf.keras.layers.Lambda(lambda x: compute_loss(*x, config.weighted))([ruij, neg_prob])
        self.model = tf.keras.Model(inputs=[user_id, pos_id, neg_id, neg_prob], outputs=loss)
        self.model.compile(loss=identity_loss, optimizer=tf.keras.optimizers.Adam(lr=config.learning_rate))
        self.config = config

    def get_uv(self):
        user_embed = self.model.get_layer('user_embedding')
        item_embed = self.model.get_layer('item_embedding')
        u = user_embed.get_weights()[0]
        v = item_embed.get_weights()[0]
        return u, v

    def fit(self, train):
        steps_per_epoch = int((train.nnz + self.config.batch_size - 1) / self.config.batch_size)
        opt_para = [{}, {'mode': self.config.mode}, {'num_clusters': self.config.num_clusters},
                          {'num_clusters': self.config.num_clusters, 'mode': self.config.mode}, {}]
        if self.config.sampler in {0, 1}:
            sampler = samplers[self.config.sampler](train, **opt_para[self.config.sampler])\
                .negative_sampler(neg=self.config.neg_num)
            dataset = IO.construct_dataset(sampler, self.config.neg_num).shuffle(50000)\
                .batch(self.config.batch_size).repeat(self.config.epochs)
            self.model.fit(dataset, epochs=self.config.epochs, steps_per_epoch=steps_per_epoch, verbose=0,
                           callbacks=[EvaluateCallback(0)])
        elif self.config.sampler in {2, 3, 4}:
            sampler = samplers[self.config.sampler].__bases__[0](train, **opt_para[self.config.sampler % 2])\
                .negative_sampler(neg=self.config.neg_num)
            for i in range(int(self.config.epochs / self.config.epochs_)):
                dataset = IO.construct_dataset(sampler, self.config.neg_num).shuffle(50000)\
                    .batch(self.config.batch_size).repeat(self.config.epochs_)
                self.model.fit(dataset, epochs=self.config.epochs_, steps_per_epoch=steps_per_epoch, verbose=0,
                               callbacks=[EvaluateCallback(i * self.config.epochs_)])
                u, v = self.get_uv()
                sampler = samplers[self.config.sampler](train, {'U': u, 'V': v}, **opt_para[self.config.sampler])\
                    .negative_sampler(self.config.neg_num)


    def evaluate(self, train, test):
        m, n = train.shape
        u, v = self.get_uv()
        users = np.random.choice(m, min(m, 50000), False)
        m = Eval.evaluate_item(train[users, :], test[users, :], u[users, :], v, topk=-1)
        return m


class PerisJointModel:
    def __init__(self, config):
        user_id = tf.keras.Input(shape=(1,), name='user_id')
        pos_id = tf.keras.Input(shape=(1,), name='pos_id')
        neg_id = tf.keras.Input(shape=(config.neg_num,), name='neg_id')
        neg_prob = tf.keras.Input(shape=(config.neg_num,), name='neg_prob', dtype='float32')
        item_embed_layer = tf.keras.layers.Embedding(config.num_item, config.d, name='item_embedding',
                                                     embeddings_initializer=tf.keras.initializers.glorot_normal(),
                                                     activity_regularizer=tf.keras.regularizers.l2(config.coef / config.batch_size))
        user_embed = tf.keras.layers.Embedding(config.num_user, config.d, name='user_embedding',
                                               embeddings_initializer=tf.keras.initializers.glorot_normal(),
                                               activity_regularizer=tf.keras.regularizers.l2(config.coef / config.batch_size))(user_id)

        pos_item_embed = item_embed_layer(pos_id)
        neg_item_embed = item_embed_layer(neg_id)
        pos_score = tf.keras.layers.dot([user_embed, pos_item_embed], axes=-1)
        neg_score = tf.keras.layers.dot([user_embed, neg_item_embed], axes=-1)
        ruij = tf.keras.layers.Flatten()(tf.keras.layers.subtract([pos_score, neg_score]))
        loss = tf.keras.layers.Lambda(lambda x: compute_loss(*x, config.weighted))([ruij, neg_prob])

        num_clusters = config.num_clusters
        reg = tf.keras.layers.ActivityRegularization(l2=config.coef2 / config.batch_size)
        dist = EuclideanDistillation(coef=config.coef_kd)

        def transform(x):
            return reg(dist(x))

        stop_grad = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x))
        num_codewords = [num_clusters]
        item_rce_layer = RCEncoding(num_codewords, att_mode='bilinear', rnn_mode='none', name='rcencoding')

        pos_item_embed_stop = stop_grad(pos_item_embed)
        neg_item_embed_stop = stop_grad(neg_item_embed)
        pos_item_embed_, pos_item_cluster_idx = item_rce_layer(pos_item_embed_stop)
        neg_item_embed_, _ = item_rce_layer(neg_item_embed_stop)
        user_embed_ = tf.keras.layers.Dense(config.d, use_bias=False, name='user_dense',
                                            activity_regularizer=tf.keras.regularizers.l2(config.coef2 / config.batch_size))(stop_grad(user_embed))
        pos_score_ = tf.keras.layers.dot([user_embed_, transform([pos_item_embed_stop, pos_item_embed_])], axes=-1)
        neg_score_ = tf.keras.layers.dot([user_embed_, transform([neg_item_embed_stop, neg_item_embed_])], axes=-1)
        ruij_ = tf.keras.layers.Flatten()(tf.keras.layers.subtract([pos_score_, neg_score_]))
        loss_ = tf.keras.layers.Lambda(lambda x: compute_loss(*x, config.weighted))([ruij_, neg_prob])
        loss2 = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([loss, loss_])

        self.model = tf.keras.Model(inputs=[user_id, pos_id, neg_id, neg_prob], outputs=loss2)
        self.model.compile(loss=identity_loss, optimizer=tf.keras.optimizers.Adam())
        self.config = config

    def get_cluster(self, m, n):
        cluster = self.model.get_layer('rcencoding')
        model_pred_item_cluster = tf.keras.Model(inputs=self.model.input[1], outputs=cluster.output[1])
        user_embed_layer = self.model.get_layer('user_dense')
        model_pred_user = tf.keras.Model(inputs=self.model.input[0], outputs=user_embed_layer.output)
        item_code = np.squeeze(model_pred_item_cluster.predict(np.arange(n)))
        item_center = cluster.get_weights()[0]
        U = np.squeeze(model_pred_user.predict(np.arange(m)))
        return U, item_code, item_center

    def get_uv(self):
        user_embed = self.model.get_layer('user_embedding')
        item_embed = self.model.get_layer('item_embedding')
        U = user_embed.get_weights()[0]
        V = item_embed.get_weights()[0]
        return U, V

    def fit(self, train):
        steps_per_epoch = int((train.nnz + self.config.batch_size - 1) / self.config.batch_size)
        opt_para = {} if self.config.sampler == 2 else {'mode': self.config.mode}
        if self.config.sampler in {2, 3}:
            sampler = samplers[self.config.sampler].__bases__[0](train, **opt_para)\
                .negative_sampler(neg=self.config.neg_num)
            for i in range(int(self.config.epochs / self.config.epochs_)):
                dataset = IO.construct_dataset(sampler, self.config.neg_num).shuffle(50000)\
                    .batch(self.config.batch_size).repeat(self.config.epochs_)
                self.model.fit(dataset, epochs=self.config.epochs_, steps_per_epoch=steps_per_epoch, verbose=0,
                               callbacks=[EvaluateCallback(i * self.config.epochs_)])
                u, code, center = self.get_cluster(self.config.num_user, self.config.num_item)
                sampler = ClusterPopularSamplerModel(train, {'U': u, 'code': code, 'center': center}, **opt_para)\
                    .negative_sampler(self.config.neg_num)


    def evaluate(self, train, test):
        m, n = train.shape
        u, v = self.get_uv()
        users = np.random.choice(m, min(m, 50000), False)
        m = Eval.evaluate_item(train[users, :], test[users, :], u[users, :], v, topk=-1)
        return m