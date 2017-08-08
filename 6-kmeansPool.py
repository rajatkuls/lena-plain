import glob
import numpy as np
import cPickle as pickle
from sklearn.cluster import KMeans
import os

np.random.seed(42)

DATA_PATH = '../data/'
ratio1 = 0.5
ratio2 = 0.2
k = 100
feature_type = 'mfcc'
OUTPUT_FILE1 = DATA_PATH + 'mfccPooled' + str(ratio1)
OUTPUT_FILE2 = DATA_PATH + 'mfccPooled' + str(ratio2)


fwrite1 = open(OUTPUT_FILE1,"w")
fwrite2 = open(OUTPUT_FILE2,"w")

mfccFiles = glob.glob(DATA_PATH+'wav/*/*mfcc')

print len(mfccFiles)

pos = 0


for e in mfccFiles:
        if pos%500==0:
                print pos
        pos+=1
        try:
                array = np.loadtxt(e,delimiter=';')
                np.random.shuffle(array)
                select_size = int(array.shape[0] * ratio1)
                feat_dim = array.shape[1]
                for i in xrange(select_size):
                        s = ';'.join(str(x) for x in array[i])
                        fwrite1.write(s + '\n')
                select_size = int(array.shape[0] * ratio2)
                feat_dim = array.shape[1]
                for i in xrange(select_size):
                        s = ';'.join(str(x) for x in array[i])
                        fwrite2.write(s + '\n')
        except:
                pass



fwrite1.close()
fwrite2.close()

pooled1 = np.loadtxt(OUTPUT_FILE1,delimiter=';')
pooled2 = np.loadtxt(OUTPUT_FILE2,delimiter=';')

kmeans1 = KMeans(n_clusters=k, init='k-means++', max_iter = 20, n_jobs = -1, random_state = 0).fit(pooled1)
kmeans2 = KMeans(n_clusters=k, init='k-means++', max_iter = 20, n_jobs = -1, random_state = 0).fit(pooled2)
#kmeans3 = KMeans(n_clusters=k, init='k-means++', max_iter = 20, n_jobs = -1, random_state = 0).fit(t1)

os.system('mkdir -p kmeansDump')
pickle.dump(kmeans1, open('kmeansDump/kmeans_' + str(k) + '_' + feature_type + '_' + str(ratio1) + '.p','wb'))
pickle.dump(kmeans2, open('kmeansDump/kmeans_' + str(k) + '_' + feature_type + '_' + str(ratio1) + '.p','wb'))








