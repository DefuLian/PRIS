import scipy as sp
import scipy.sparse
import numpy as np
import scipy.io as sio
from utils import IO, Eval, Misc
import tensorflow as tf
import os
import time
import sys
from sampler import *
import sampler
from sklearn.cluster import KMeans
from mylayers import RCEncoding, EuclideanDistillation

def evaluate_model(model, train, test):
    m, n = train.shape
    user_embed = model.get_layer('user_embedding')
    item_embed = model.get_layer('item_embedding')
    U = user_embed.get_weights()[0]
    V = item_embed.get_weights()[0]
    users = np.random.choice(m, min(m, 50000), False)
    m = Eval.evaluate_item(train[users, :], test[users, :], U[users,:], V)
    print(Eval.format(m))
    return m
def get_uv(model):
    user_embed = model.get_layer('user_embedding')
    item_embed = model.get_layer('item_embedding')
    U = user_embed.get_weights()[0]
    V = item_embed.get_weights()[0]
    return U, V
def get_cluster(model, m, n):
    cluster = model.get_layer('rcencoding')
    model_pred_item_cluster = tf.keras.Model(inputs=model.input[1], outputs=cluster.output[1])
    user_embed_layer = model.get_layer('user_dense')
    model_pred_user = tf.keras.Model(inputs=model.input[0], outputs=user_embed_layer.output)
    item_code = np.squeeze(model_pred_item_cluster.predict(np.arange(n)))
    item_center = cluster.get_weights()[0]
    U = np.squeeze(model_pred_user.predict(np.arange(m)))
    return U, item_code, item_center


class EvaluateCallback(tf.keras.callbacks.Callback):
    def __init__(self, train, test, round):
        num_users, num_items = train.shape
        self.train = train
        self.test = test
        self.round = round
        self.test_users = np.random.choice(num_users, min(num_users, 5000), False)
        self.test_users2 = np.random.choice(num_users, min(num_users, 50000), False)
        self.test_items = np.arange(num_items)
        super(EvaluateCallback, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = time.time()

    def on_epoch_end(self, epoch, logs=None):
        user_embed = self.model.get_layer('user_embedding')
        item_embed = self.model.get_layer('item_embedding')
        user_func = tf.keras.backend.function(inputs=user_embed.input, outputs=user_embed.output)
        item_func = tf.keras.backend.function(inputs=item_embed.input, outputs=item_embed.output)
        U = user_func(self.test_users)
        V = item_func(self.test_items)
        m = Eval.evaluate_item(self.train[self.test_users,:], self.test[self.test_users,:], U, V)
        elapsed = time.time() - self.starttime
        print('Epoch={} - {}s - ndcg@100={:.4f} - recall@100={:.4f} - loss={:.4f}'
              .format(self.round+epoch+1, int(elapsed), m['item_ndcg'][100], m['item_recall'][100], logs['loss']))


def compute_loss(pred, prob):
    importance = tf.nn.softmax(tf.negative(pred) - tf.log(prob))
    weight_loss = tf.multiply(importance, tf.negative(tf.log_sigmoid(pred)))
    loss = tf.reduce_sum(weight_loss, -1, keepdims=True)
    return loss

def identity_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred - 0 * y_true)


def main():
    Misc.set_seed()
    #mat = IO.load_matrix_from_file(sys.argv[1])
    mat = IO.load_matrix_from_file('C:/Users/USTC/Desktop/citeulikedata.mat')
    num_user, num_item = mat.shape
    d = 32
    batch_size = 128
    epochs = 60
    neg_num = 5
    coef = 2
    coef2 = 0.1
    user_id = tf.keras.Input(shape=(1,), name='user_id')
    pos_id = tf.keras.Input(shape=(1,), name='pos_id')
    neg_id = tf.keras.Input(shape=(neg_num,), name='neg_id')
    neg_prob = tf.keras.Input(shape=(neg_num,), name='neg_prob', dtype='float32')
    item_embed_layer = tf.keras.layers.Embedding(num_item, d, name='item_embedding',
                                                 embeddings_initializer=tf.keras.initializers.glorot_normal(),
                                                 activity_regularizer=tf.keras.regularizers.l2(coef / batch_size))
    user_embed = tf.keras.layers.Embedding(num_user, d, name='user_embedding',
                                           embeddings_initializer=tf.keras.initializers.glorot_normal(),  # )(user_id)
                                           activity_regularizer=tf.keras.regularizers.l2(coef / batch_size))(user_id)

    pos_item_embed = item_embed_layer(pos_id)
    neg_item_embed = item_embed_layer(neg_id)
    pos_score = tf.keras.layers.dot([user_embed, pos_item_embed], axes=-1)
    neg_score = tf.keras.layers.dot([user_embed, neg_item_embed], axes=-1)
    ruij = tf.keras.layers.Flatten()(tf.keras.layers.subtract([pos_score, neg_score]))
    loss = tf.keras.layers.Lambda(lambda x: compute_loss(*x))([ruij, neg_prob])

    num_clusters = 100
    reg = tf.keras.layers.ActivityRegularization(l2=coef2 / batch_size)
    dist = EuclideanDistillation(coef=5)
    def transform(x):
        return reg(dist(x))
    stop_grad = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x))
    num_codewords = [num_clusters]
    item_rce_layer = RCEncoding(num_codewords, att_mode='bilinear', rnn_mode='none', name='rcencoding')

    pos_item_embed_stop = stop_grad(pos_item_embed)
    neg_item_embed_stop = stop_grad(neg_item_embed)
    pos_item_embed_, pos_item_cluster_idx  = item_rce_layer(pos_item_embed_stop)
    neg_item_embed_, _ = item_rce_layer(neg_item_embed_stop)
    user_embed_ = tf.keras.layers.Dense(d, use_bias=False, name='user_dense',
                                        activity_regularizer=tf.keras.regularizers.l2(coef2 / batch_size))(stop_grad(user_embed))
    #user_embed_ = tf.keras.layers.Dense(d, use_bias=False)(stop_grad(user_embed))
    #user_embed_ = stop_grad(user_embed)
    pos_score_ = tf.keras.layers.dot([user_embed_, transform([pos_item_embed_stop, pos_item_embed_])], axes=-1)
    neg_score_ = tf.keras.layers.dot([user_embed_, transform([neg_item_embed_stop, neg_item_embed_])], axes=-1)
    ruij_ = tf.keras.layers.Flatten()(tf.keras.layers.subtract([pos_score_, neg_score_]))
    loss_ = tf.keras.layers.Lambda(lambda x: compute_loss(*x))([ruij_, neg_prob])
    loss2 = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([loss, loss_])

    model = tf.keras.Model(inputs=[user_id, pos_id, neg_id, neg_prob], outputs=loss2)
    model.compile(loss=identity_loss, optimizer=tf.keras.optimizers.Adam())

    #print(model.summary())
    train, test = IO.split_matrix(mat, 0.8)

    #sampler = IO.negative_sampler(train, neg_num, is_uniform=False)
    #dataset = IO.construct_dataset(sampler, neg_num).shuffle(50000).batch(batch_size).repeat(epochs)
    #model.fit(dataset, epochs=epochs, steps_per_epoch=int((train.nnz + batch_size - 1) / batch_size),
    #          callbacks=[EvaluateCallback('C:/Users/USTC/Desktop/', train, test)], verbose=0)
    #sampler = IO.negative_sampler_with_modelfile(train, 'C:/Users/USTC/Desktop/uv.npz', neg_num)
    #sampler = PopularSamplerModel(train, 'C:/Users/USTC/Desktop/uv.npz').negative_sampler(neg_num)
    #sampler = SamplerModel(train).negative_sampler(neg_num)
    #sampler = PopularSamplerModel(train).negative_sampler(neg_num)
    #dataset = IO.construct_dataset(sampler, neg_num).shuffle(50000).batch(batch_size).repeat(epochs)
    #model.fit(dataset, epochs=epochs, steps_per_epoch=int((train.nnz + batch_size - 1) / batch_size),
    #          callbacks=[EvaluateCallback(train, test)], verbose=0)
    #evaluate_model(model, train, test)
    #exit()
    #U, code, center = get_cluster(model, num_user, num_item)
    sampler = PopularSamplerModel(train, mode=1).negative_sampler(neg_num)
    #num_clusters = 10
    #sampler = ClusterPopularSamplerModel(train, {'U': U, 'code': code, 'center': center}).negative_sampler(neg_num)
    epochs_ = 1
    for i in range(int(epochs/epochs_)):
        dataset = IO.construct_dataset(sampler, neg_num).shuffle(50000).batch(batch_size).repeat(epochs_)
        model.fit(dataset, epochs=epochs_, steps_per_epoch=int((train.nnz + batch_size - 1) / batch_size), verbose=2)
              #callbacks=[EvaluateCallback(train, test, i*epochs_)], verbose=0)
        U, code, center = get_cluster(model, num_user, num_item)
        sampler = ClusterPopularSamplerModel(train, {'U': U, 'code': code, 'center': center}, mode=1).negative_sampler(neg_num)
    evaluate_model(model, train, test)


    exit()
    sampler = PopularSamplerModel(train).negative_sampler(neg_num)
    dataset = IO.construct_dataset(sampler, neg_num).shuffle(50000).batch(batch_size).repeat(epochs)
    model.fit(dataset, epochs=epochs, steps_per_epoch=int((train.nnz + batch_size - 1) / batch_size),verbose=1)
    evaluate_model(model, train, test)

    #model_pred_item_cluster = tf.keras.Model(inputs=pos_id, outputs=pos_item_cluster_idx)
    model_pred_item = tf.keras.Model(inputs=pos_id, outputs=pos_item_embed_)
    model_pred_user = tf.keras.Model(inputs=user_id, outputs=user_embed_)
    #item_code = model_pred_item.predict(np.arange(num_item))
    #item_center = item_rce_layer.get_weights()[0]
    U = np.squeeze(model_pred_user.predict(np.arange(num_user)))
    V = np.squeeze(model_pred_item.predict(np.arange(num_item)))
    m2 = Eval.evaluate_item(train, test, U, V)
    print(Eval.format(m2))
    U, V = get_uv(model)
    kmeans = KMeans(num_clusters, random_state=0).fit(V)
    code = kmeans.labels_
    center = kmeans.cluster_centers_
    V = center[code]
    m2 = Eval.evaluate_item(train, test, U, V)
    print(Eval.format(m2))
    U = np.ones([num_user, 1])
    V = train.sum(axis=0).A.T
    m2 = Eval.evaluate_item(train, test, U, V)
    print(Eval.format(m2))






if __name__ == "__main__":
    main()


