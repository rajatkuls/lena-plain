import numpy as np
import os
import sys
from sklearn.cluster import KMeans
import cPickle as pickle
import glob
from collections import Counter

split = int(sys.argv[1])

k = 100
feature_type = 'mfcc'

VD_PATH = '../VanDam/'
DATA_PATH = '../data/'




kmeans = pickle.load(open('kmeansDump/kmeans_' + str(num_clusters) + '_' + feature_type + '.p','rb'))

#mfccFiles = glob.glob(DATA_PATH+'*/*mfcc')
outFolder = '../data/splits/mfcc/'
mfccFiles = pickle.load(open(outFolder+str(i)+'.p','r'))

for e in mfccFiles:
	t1 = np.loadtxt(e,delimiter=';')
	labels = kmeans.predict(input_data)
	c = Counter(labels)
	s = = ";".join(str(c[i]/float(input_data.shape[0])) for i in xrange(num_clusters))
	outputFile = open(e+'.hist','w')
	output_file.write(s)
	output_file.close()








