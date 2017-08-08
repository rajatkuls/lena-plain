import os
import glob
import sys
import cPickle as pickle
import ast


WAV_PATH = '../data/wav/'
SPLIT_PATH = '../data/splits/20folder/'
'''
wavFolders = glob.glob(WAV_PATH+'*')
CORES = 20
for i in xrange(CORES-1):
	pickle.dump(wavFolders[i*8:(i+1)*8],open(SPLIT_PATH+str(i)+'.p','w'))

i+=1
pickle.dump(wavFolders[i*8:],open(SPLIT_PATH+str(i)+'.p','w'))
'''


for i in xrange(20):
	os.system('nohup python 8-audioFeatures.py '+str(i)+' 1>stdout'+str(i)+'.log 2>stderr'+str(i)+'.log &')






















