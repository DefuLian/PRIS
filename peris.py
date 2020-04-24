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
    train, test = IO.split_matrix(mat, 0.8)
    num_user, num_item = mat.shape
    d = 32
    batch_size = 128
    epochs = 60
    neg_num = 5
    coef = 2
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
    model = tf.keras.Model(inputs=[user_id, pos_id, neg_id, neg_prob], outputs=loss)
    model.compile(loss=identity_loss, optimizer=tf.keras.optimizers.Adam())
    #print(model.summary())
    #train, test = IO.split_matrix(mat, 0.8)

    #sampler = IO.negative_sampler(train, neg_num, is_uniform=False)
    #dataset = IO.construct_dataset(sampler, neg_num).shuffle(50000).batch(batch_size).repeat(epochs)
    #model.fit(dataset, epochs=epochs, steps_per_epoch=int((train.nnz + batch_size - 1) / batch_size),
    #          callbacks=[EvaluateCallback('C:/Users/USTC/Desktop/', train, test)], verbose=0)
    #sampler = IO.negative_sampler_with_modelfile(train, 'C:/Users/USTC/Desktop/uv.npz', neg_num)
    #sampler = PopularSamplerModel(train, 'C:/Users/USTC/Desktop/uv.npz').negative_sampler(neg_num)
    #sampler = SamplerModel(train).negative_sampler(neg_num)
    #sampler = PopularSamplerModel(train).negative_sampler(neg_num)
    #dataset = IO.construct_dataset(sampler, neg_num).shuffle(50000).batch(batch_size).repeat(epochs)
    #model.fit(dataset, epochs=epochs, steps_per_epoch=int((train.nnz + batch_size - 1) / batch_size), verbose=2)
    #          #callbacks=[EvaluateCallback(train, test, 0)], verbose=0)
    #evaluate_model(model, train, test)
    #U, V = get_uv(model)
    #sampler = TreeSamplerModel(train, {'U': U, 'V': V}, max_depth=10).negative_sampler(neg_num)
    sampler = SamplerModel(train).negative_sampler(neg_num)
    num_clusters = 100
    epochs_ = 1
    for i in range(int(epochs/epochs_)):
        #sampler = ExactSamplerModel(train, {'U':U, 'V':V}).negative_sampler(neg_num)
        #sampler = ClusterSamplerModel(train, {'U': U, 'V': V}, num_clusters=10).negative_sampler(neg_num)
        #sampler = PopularSamplerModel(train).negative_sampler(neg_num)
        dataset = IO.construct_dataset(sampler, neg_num).shuffle(50000).batch(batch_size).repeat(epochs_)
        model.fit(dataset, epochs=epochs_, steps_per_epoch=int((train.nnz + batch_size - 1) / batch_size), verbose=2)
              #callbacks=[EvaluateCallback(train, test, i*epochs_)], verbose=0)
        U, V = get_uv(model)
        #sampler = ClusterSamplerModel(train, {'U': U, 'V': V}, num_clusters=num_clusters).negative_sampler(neg_num)
        sampler = ExactSamplerModel(train, {'U': U, 'V': V}).negative_sampler(neg_num)
        #sampler = TreeSamplerModel(train, {'U': U, 'V': V}, max_depth=11).negative_sampler(neg_num)
    evaluate_model(model, train, test)




if __name__ == "__main__":
    main()


