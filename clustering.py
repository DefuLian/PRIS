import numpy as np
import math
from collections import Counter
from sklearn.cluster import KMeans
#from matplotlib import pyplot as plt
from sklearn import manifold
import scipy as sp
import scipy.special
import random


def hierarchical_clustering(vector:np.ndarray, max_depth=3, weight=None):
    n, d = vector.shape
    if weight is None:
        weight = np.ones(n)
    depth_ = math.floor(np.log(n)) - 1
    if max_depth > depth_:
        max_depth = depth_

    num_clusters = 2**(max_depth + 1)
    cluster_center = np.zeros((num_clusters, d))
    weight_sum_in_clusters = np.zeros(num_clusters)
    label = np.zeros(n, dtype=np.int) - 1

    def clustering_(items, root_id):
        depth = int(np.log2(root_id))
        if depth < max_depth:
            left_root = 2 * root_id
            right_root = 2 * root_id + 1
            code, cluster_center[[left_root, right_root]] = balance(KMeans(2).fit(vector[items]), vector[items])
            left = items[(code == 0).nonzero()[0]]
            right = items[(code == 1).nonzero()[0]]
            left_weight = clustering_(left, left_root)
            right_weight = clustering_(right, right_root)
            weight_sum_in_clusters[[left_root, right_root]] = left_weight, right_weight
            return left_weight + right_weight
        else:
            weight_sum = 0.
            for i in items:
                label[i] = root_id
                weight_sum += weight[i]
            return weight_sum

    clustering_(np.arange(n), 1)

    return cluster_center, weight_sum_in_clusters, label


def balance(kmeans, vector):
    code = kmeans.labels_
    vector_transform = kmeans.transform(vector)
    n = code.shape[0]
    priority = np.abs(vector_transform[:, 1] - vector_transform[:, 0])
    cnt = Counter(code)
    left_num = int(np.ceil(n/2))
    if cnt[0] > cnt[1]:
        count_larger, count_smaller = cnt[0], cnt[1]
        label_larger, label_smaller = 0, 1
    else:
        count_larger, count_smaller = cnt[1], cnt[0]
        label_larger, label_smaller = 1, 0
    idx_larger = (code == label_larger).nonzero()[0]
    idx_smaller = (code == label_smaller).nonzero()[0]
    priority_larger = priority[idx_larger]
    priority_smaller = priority[idx_smaller]
    if left_num < count_larger:  # move larger to smaller
        count = count_larger - left_num
        idx_move_local = np.argpartition(priority_larger, count)[:count]
        idx_move_global = idx_larger[idx_move_local]
        code[idx_move_global] = label_smaller
    elif left_num > count_larger:  # move smaller to larger
        count = left_num - count_larger
        idx_move_local = np.argpartition(priority_smaller, count)[:count]
        idx_move_global = idx_smaller[idx_move_local]
        code[idx_move_global] = label_larger

    code_mat = np.eye(2)[code]
    cluster_sum = 1 / code_mat.sum(axis=0)
    code_mat = np.diag(cluster_sum)[code]
    center = code_mat.T @ vector
    return code, center


def distribute_into_tree(exist, label, length, weight=None):
    n = label.shape[0]
    if weight is None:
        weight = np.ones(n)
    label2weight = np.zeros(length)
    #label2weight = dict()
    for e in exist:
        l = label[e]
        while l > 1:
            label2weight[l] += weight[e]
            # if l not in label2weight:
            #    label2weight[l] = weight[e]
            # else:
            #    label2weight[l] += weight[e]
            l = int(l/2)
    return label2weight

def distribute_into_leaf(items, label, leaf_start, num_leaves):
    items_in_leaves = [[] for _ in range(num_leaves)]
    for i in items:
        items_in_leaves[label[i] - leaf_start].append(i)
    return items_in_leaves

def leaf_sampling(cluster_score):
    max_depth = int(np.log2(cluster_score.shape[0] - 1))
    curr_node = 1
    prob = 1
    while int(np.log2(curr_node)) < max_depth:
        left = 2 * curr_node
        #prob_l = cluster_score[left] / (cluster_score[left + 1] + cluster_score[left])
        prob_l = cluster_score[left]
        if random.random() < prob_l:
            curr_node = left
            prob = prob * prob_l
        else:
            curr_node = left + 1
            prob = prob * (1 - prob_l)
    return curr_node, prob

def leaf_sampling1(cluster_score, weight_sum_cluster, label2weight):
    max_depth = int(np.log2(cluster_score.shape[0] - 1))

    def sampling(root):
        if int(np.log2(root)) == max_depth:
            return root, 1
        left = 2 * root
        right = 2 * root + 1
        score, weight = cluster_score[left], weight_sum_cluster[left]
        score_l = score * (weight - label2weight[left])
        score, weight = cluster_score[right], weight_sum_cluster[right]
        score_r = score * (weight - label2weight[right])
        prob_l = score_l / (score_l + score_r)
        if random.random() < prob_l:
            id, prob = sampling(left)
            return id, prob * prob_l
        else:
            id, prob = sampling(right)
            return id, prob * (1 - prob_l)

    return sampling(1)



# pretrain_uv = 'uv.npz'
# data = np.load(pretrain_uv)
# U = data['U']
# V = data['V']
#
# cluster_center_, weight_sum_in_clusters_, label_ = hierarchical_clustering(V, 6)
#
# label2weight = distribute_into_tree(np.random.randint(0, V.shape[0], 100), label_)
# cnt = Counter()
# prob = 0
# cluster2prob = {}
# for _ in range(100000):
#     c = leaf_sampling(U[0], cluster_center_, weight_sum_in_clusters_, label2weight)
#     if c[0] not in cnt:
#         prob += c[1]
#         cluster2prob[c[0]] = c[1]
#     cnt[c[0]] += 1
# print(prob)
# sum_ = sum(cnt.values())
# for e in sorted(cnt.items(), key=lambda x:-x[1]):
#     print(e[0], e[1]/sum_, cluster2prob[e[0]])
# for node_id in range(2,2**4):
#     v = np.zeros(32)
#     node_depth = int(np.log2(node_id))
#     count = 0
#     for i in range(len(label_)):
#         l = label_[i]
#         depth = int(np.log2(l))
#         while depth > node_depth:
#             l = int(l/2)
#             depth = int(np.log2(l))
#         if l == node_id:
#             v += V[i]
#             count += 1
#
#     print(np.mean(np.abs(v / num_items_in_clusters_[node_id] - cluster_center_[node_id])))





# x = np.zeros((5000,2))
# x[:3000,:] =  np.random.randn(3000,2) + [0.5,1]
# x[3000:,:] =  np.random.randn(2000,2) + [2,0.3]
# idx1 = np.logical_and(code1 == code, code == 0)
# idx2 = np.logical_and(code1 == code, code == 1)
# idx3 = code1 != code
#
# #vector = manifold.TSNE().fit_transform(vector)
# plt.plot(vector[idx1, 0], vector[idx1, 1], '.')
# plt.plot(vector[idx2, 0], vector[idx2, 1], '.')
# plt.plot(vector[idx3, 0], vector[idx3, 1], '.')
# plt.show()
#def hierarchical_clustering_balanced_binary(vector:np.ndarray):

