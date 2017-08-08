import glob
import numpy as np
import cPickle as pickle
from sklearn.cluster import KMeans
import os

np.random.seed(42)

VD_PATH = '../VanDam/'
DATA_PATH = '../dataOld/'
ratio = 0.5
k = 100
feature_type = 'mfcc'
OUTPUT_FILE = DATA_PATH + 'mfccPooled' + str(ratio)


fwrite = open(OUTPUT_FILE,"w")

mfccFiles = glob.glob(DATA_PATH+'*/*mfcc')


for e in mfccFiles:
	array = np.loadtxt(e,delimiter=';')
	np.random.shuffle(array)
	select_size = int(array.shape[0] * ratio)
	feat_dim = array.shape[1]
	for i in xrange(select_size):
		s = ';'.join(str(x) for x in array[i])
		fwrite.write(s + '\n')

fwrite.close()


pooled = np.loadtxt(OUTPUT_FILE,delimiter=';')

kmeans = KMeans(n_clusters=k, init='k-means++', max_iter = 20, n_jobs = -1, random_state = 0).fit(pooled)

os.system('mkdir -p kmeansDump')
pickle.dump(kmeans, open('kmeansDump/kmeans_' + str(k) + '_' + feature_type + '.p','wb'))



