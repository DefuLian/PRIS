from peris_model import PerisJointModel
import argparse
from utils import IO, Misc
import scipy as sp
import scipy.io
def main(config):
    Misc.set_seed()
    print(config)
    data = IO.load_matrix_from_file(config.data)
    train, test = IO.split_matrix(data, config.ratio)
    config.num_user, config.num_item = data.shape
    model = PerisJointModel(config)
    model.fit(train)
    m = model.evaluate(train, test)
    return m
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data', help='input data for training and testing')
    parser.add_argument('output', help='output evaluation results into the output file with matlab format')
    parser.add_argument('-r', '--ratio', type=float, default=0.8, help='ratio of split data')
    parser.add_argument('-d', '--dim', dest='d', type=int, default=32, help='embedding size')
    parser.add_argument('-e', '--epoch', dest='epochs', type=int, default=60, help='epoches for iterating data')
    parser.add_argument('-b', '--batch', dest='batch_size', type=int, default=128, help='batch size' )
    parser.add_argument('-n', '--neg', dest='neg_num', type=int, default=5, help='number of negative items')
    parser.add_argument('-a', '--coef', type=float, default=2, help='coefficient for activity regularization')
    parser.add_argument('-c', '--clusters', dest='num_clusters', type=int, default=100, help='number of clusters in group based sampler')
    parser.add_argument('-i', '--iter', dest='epochs_', type=int, default=1, help='epochs of sampler substitution')
    parser.add_argument('-s', '--sampler', type=int, choices=[2,3], default=3, help='choice of negative sampler')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate for adam')
    parser.add_argument('-a1', '--coef2', type=float, default=0.1, help='coefficient for activity regularization in clustering')
    parser.add_argument('-a2', dest='coef_kd', default=5, help='coefficient for knowledge distillation')
    parser.add_argument('-m', '--mode', type=int, default=1, choices=[1, 2], help='choice of popularity')
    parser.add_argument('-w', '--not-weight', dest='weighted', action='store_false', default=True, help='weighted loss function or not')
    config = parser.parse_args()
    m = main(config)
    sp.io.savemat(config.output, m)
    #print(Eval.format(m))