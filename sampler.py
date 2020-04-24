from utils import IO, Eval, Misc
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.special
from sklearn.cluster import KMeans
import bisect
import random
import clustering

class SamplerModel:
    def __init__(self, mat):
        self.mat = mat.tocsr()
        self.num_users, self.num_items = mat.shape

    def preprocess(self, user_id):
        self.exist = set(self.mat.indices[j] for j in range(self.mat.indptr[user_id], self.mat.indptr[user_id + 1]))



    def __sampler__(self, user_id, pos_id):
        def sample():
            k = np.random.choice(self.num_items)
            return k, 1.0/self.num_items
        return sample, self.exist

    def negative_sampler(self, neg):
        def sample_negative(user_id, pos_id):
            sample, exist_ = self.__sampler__(user_id, pos_id)
            k, p = sample()
            while k in exist_:
                k, p = sample()
            return k, p

        def generate_tuples():
            for i in np.random.permutation(self.num_users):
                self.preprocess(i)
                for j in self.exist:
                    neg_item = [0] * neg
                    prob = [0.] * neg
                    for o in range(neg):
                        neg_item[o], prob[o] = sample_negative(i, j)
                    yield ([i], [j], neg_item, prob), 1

        return generate_tuples

class PopularSamplerModel(SamplerModel):
    def __init__(self, mat, mode=0):
        super(PopularSamplerModel, self).__init__(mat)
        pop_count = np.squeeze(mat.sum(axis=0).A)
        if mode == 0:
            pop_count = np.log(pop_count + 1)
        elif mode == 1:
            pop_count = np.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        self.pop_prob = pop_count / np.sum(pop_count)
        self.pop_cum_prob = self.pop_prob.cumsum()
    def __sampler__(self, user_id, pos_id):
        def sample():
            k = bisect.bisect(self.pop_cum_prob, random.random())
            p = self.pop_prob[k]
            return k, p
        return sample, self.exist



class ExactSamplerModel(SamplerModel):
    def __init__(self, mat, model):
        super(ExactSamplerModel, self).__init__(mat)
        if isinstance(model, str):
            model = np.load(model)
        self.user = model['U']
        self.item = model['V']
    def preprocess(self, user_id):
        super(ExactSamplerModel, self).preprocess(user_id)
        pred = self.user[user_id] @ self.item.T
        idx = np.argpartition(pred, -10)[-10:]
        pred[idx] = -np.inf
        self.score = sp.special.softmax(pred)
        self.score_cum = self.score.cumsum()
        self.score_cum[-1] = 1.0
    def __sampler__(self, user_id, pos_id):
        def sample():
            k = bisect.bisect(self.score_cum, random.random())
            p = self.score[k]
            return k, p
        return sample, self.exist

class ClusterSamplerModel(SamplerModel):
    def __init__(self, mat, model, num_clusters=100):
        super(ClusterSamplerModel, self).__init__(mat)
        if isinstance(model, str):
            model = np.load(model)
        user = model['U']
        item = model['V']
        clustering = KMeans(num_clusters, random_state=0).fit(item)
        self.code = clustering.labels_
        center = clustering.cluster_centers_
        code_mat = sp.sparse.csr_matrix((np.ones_like(self.code), (np.arange(self.num_items), self.code)),
                                        shape=(self.num_items, num_clusters))
        cluster_num = np.squeeze(code_mat.sum(axis=0).A)
        idx = cluster_num > 0
        code_mat = code_mat[:, idx].tocsc()
        self.code1 = code_mat.nonzero()[1]
        self.num_clusters = code_mat.shape[1]
        self.items_in_cluster = [[code_mat.indices[j] for j in range(code_mat.indptr[c], code_mat.indptr[c + 1])]
                                 for c in range(self.num_clusters)]
        self.num_items_in_cluster = np.array([len(self.items_in_cluster[c]) for c in range(self.num_clusters)])
        self.center_score = np.exp(np.matmul(user, center[idx].T))

    def preprocess(self, user_id):
        super(ClusterSamplerModel, self).preprocess(user_id)
        self.exist_items_in_cluster = [[] for _ in range(self.num_clusters)]
        for e in self.exist:
            self.exist_items_in_cluster[self.code[e]].append(e)
        self.num_exist_items_in_cluster = np.array(
            [len(self.exist_items_in_cluster[c]) for c in range(self.num_clusters)])

        cs_user = self.center_score[user_id] * (self.num_items_in_cluster - self.num_exist_items_in_cluster)
        self.cs_user = cs_user / cs_user.sum()
        self.cs_user_cum = self.cs_user.cumsum()

    def __sampler__(self, user_id, pos_id):
        c = bisect.bisect(self.cs_user_cum, random.random())
        prob = self.cs_user[c]/(self.num_items_in_cluster[c] - self.num_exist_items_in_cluster[c])
        item = self.items_in_cluster[c]

        def sample():
            k = np.random.choice(self.num_items_in_cluster[c])
            return item[k], prob

        return sample, self.exist_items_in_cluster[c]

class ClusterPopularSamplerModel(PopularSamplerModel):
    def __init__(self, mat, model, num_clusters=10, **kwargs):
        super(ClusterPopularSamplerModel, self).__init__(mat, **kwargs)
        if isinstance(model, str):
            model = np.load(model)
        user = model['U']
        if 'code' in model and 'center' in model:
            center = model['center']
            self.code = model['code']
            num_clusters = center.shape[0]
        else:
            item = model['V']
            clustering = KMeans(num_clusters, random_state=0).fit(item)
            self.code = clustering.labels_
            center = clustering.cluster_centers_

        code_mat = sp.sparse.csr_matrix((np.ones_like(self.code), (np.arange(self.num_items), self.code)),
                                        shape=(self.num_items, num_clusters))
        cluster_num = np.squeeze(code_mat.sum(axis=0).A)
        idx = cluster_num > 0
        code_mat = code_mat[:, idx].tocsc()
        self.code = code_mat.nonzero()[1]
        self.num_clusters = code_mat.shape[1]
        self.items_in_cluster = [np.array([code_mat.indices[j] for j in range(code_mat.indptr[c], code_mat.indptr[c + 1])])
                                 for c in range(self.num_clusters)]
        self.prob_items_in_cluster = [self.pop_prob[self.items_in_cluster[c]] for c in range(self.num_clusters)]
        self.prob_cluster = np.array([np.sum(self.prob_items_in_cluster[c]) for c in range(self.num_clusters)])
        self.prob_cum_items_in_cluster = [np.cumsum(self.prob_items_in_cluster[c])/self.prob_cluster[c] for c in range(self.num_clusters)]
        self.center_score = np.exp(np.matmul(user, center[idx].T))

    def preprocess(self, user_id):
        super(ClusterPopularSamplerModel, self).preprocess(user_id)
        self.exist_items_in_cluster = [[] for _ in range(self.num_clusters)]
        for e in self.exist:
            self.exist_items_in_cluster[self.code[e]].append(e)

        self.prob_exist_items_in_cluster = [self.pop_prob[self.exist_items_in_cluster[c]] for c in range(self.num_clusters)]
        self.prob_exist_cluster = np.array(
            [np.sum(self.prob_exist_items_in_cluster[c]) for c in range(self.num_clusters)])

        cs_user = self.center_score[user_id] * (self.prob_cluster - self.prob_exist_cluster)
        self.cs_user = cs_user / cs_user.sum()
        self.cs_user_cum = self.cs_user.cumsum()

    def __sampler__(self, user_id, pos_id):
        c = bisect.bisect(self.cs_user_cum, random.random())
        prob = self.cs_user[c]/(self.prob_cluster[c] - self.prob_exist_cluster[c])
        item = self.items_in_cluster[c]

        def sample():
            k = bisect.bisect(self.prob_cum_items_in_cluster[c], random.random())
            return item[k], prob * self.pop_prob[item[k]]

        return sample, self.exist_items_in_cluster[c]


class TreeSamplerModel(PopularSamplerModel):
    def __init__(self, mat, model, max_depth=10):
        super(TreeSamplerModel, self).__init__(mat)
        if isinstance(model, str):
            model = np.load(model)
        user = model['U']
        item = model['V']
        self.cluster_center, self.weight_sum_in_clusters, self.label = clustering.hierarchical_clustering(item, max_depth, weight=self.pop_prob)
        max_depth = int(np.log2(self.cluster_center.shape[0] - 1))
        self.leaf_start = 2**max_depth
        self.num_leaves = self.cluster_center.shape[0] - self.leaf_start
        self.items_in_leaf = clustering.distribute_into_leaf(np.arange(self.num_items), self.label, self.leaf_start, self.num_leaves)
        self.prob_items_in_leaf = [self.pop_prob[self.items_in_leaf[c]] for c in range(self.num_leaves)]
        self.prob_cum_items_in_leaf = [np.cumsum(self.prob_items_in_leaf[c])/self.weight_sum_in_clusters[c+self.leaf_start] for c in range(self.num_leaves)]
        self.center_score = np.exp(np.matmul(user, self.cluster_center.T))

    def preprocess(self, user_id):
        super(TreeSamplerModel, self).preprocess(user_id)
        self.label2weight = clustering.distribute_into_tree(self.exist, self.label, len(self.weight_sum_in_clusters), self.pop_prob)
        self.exist_items_in_leaf = clustering.distribute_into_leaf(self.exist, self.label, self.leaf_start, self.num_leaves)
        self.cs_user = self.center_score[user_id] * (self.weight_sum_in_clusters - self.label2weight)
        self.cs_user[2::2] = self.cs_user[2::2] / (self.cs_user[2::2] + self.cs_user[3::2])
        #for i in range(2, self.cs_user.shape[0], 2):
        #    self.cs_user[i] = self.cs_user[i] / (self.cs_user[i+1] + self.cs_user[i])
        #self.cs_user[3::2] = 1 - self.cs_user[2::2]

    def __sampler__(self, user_id, pos_id):
        c, p = clustering.leaf_sampling(self.cs_user)
        #c, p = clustering.leaf_sampling(self.center_score[user_id], self.weight_sum_in_clusters, self.label2weight)
        cum_prob = self.prob_cum_items_in_leaf[c - self.leaf_start]
        prob = self.prob_items_in_leaf[c - self.leaf_start]
        item_ = self.items_in_leaf[c - self.leaf_start]
        exist_ = self.exist_items_in_leaf[c - self.leaf_start]

        def sample():
            k = bisect.bisect(cum_prob, random.random())
            return item_[k], p * prob[k]

        return sample, exist_








#Misc.set_seed(10)
#mat = IO.load_matrix_from_file('C:/Users/USTC/Desktop/citeulikedata.mat')
#train, test = IO.split_matrix(mat, 0.8)
#sampler = IO.negative_sampler_with_modelfile(train, 'C:/Users/USTC/Desktop/uv.npz', 5)
#sampler = ClusterSamplerModel(train, 'C:/Users/USTC/Desktop/uv.npz', 50).negative_sampler(5)
#sampler = SamplerModel(train).negative_sampler(5)
#sampler = IO.negative_sampler(train, 5)
#sampler = IO.negative_sampler(train, 5, is_uniform=False)
#sampler = PopularSamplerModel(train).negative_sampler(5)
#sampler = ExactSamplerModel(train, 'C:/Users/USTC/Desktop/uv.npz').negative_sampler(5)
# with np.load('C:/Users/USTC/Desktop/uv.npz') as data:
#     U = data['U']
#     V = data['V']
# sampler = TreeSamplerModel(train, {'U': U, 'V': V}, max_depth=8).negative_sampler(5)
# for i,e in enumerate(sampler()):
#     #print(e)
#     if i>100000:
#         break
#     else:
#         pass