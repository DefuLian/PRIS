import scipy as sp
import scipy.io as sio
import sys
from os import listdir
from os.path import isfile, join

for fileName in sorted(listdir(sys.argv[1])):
  mat = sio.loadmat(join(sys.argv[1], fileName))
  k = int(sys.argv[2])
  print('ndcg@{}={:4f}\trecall@{}={:4f}\t{}'.format(k, mat['item_ndcg'][0, k-1], k, mat['item_recall'][0, k-1], fileName))