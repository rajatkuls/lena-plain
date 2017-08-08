import os
import glob
import numpy as np
import cPickle as pickle
from collections import Counter


f = glob.glob('../data/features/st/VanDam/*/*yclass')
t1 = pickle.load(open(f[0],'r'))

t2 = Counter(t1)

for e in f[1:]:
	t1 = pickle.load(open(e,'r'))
	t2 += Counter(t1)

print t2


f = glob.glob('../data/VanDam/*stm')

e = f[0]

t1 = open(e,'r').read().split('\n')[:-1]
t2 = Counter([x.split(' ')[1] for x in t1])

for e in f[1:]:
	t1 = open(e,'r').read().split('\n')[:-1]
	t2 += Counter([x.split(' ')[1] for x in t1])

