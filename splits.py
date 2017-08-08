import os
import glob
import cPickle as pickle

VD_PATH = '../VanDam/'
DATA_PATH = '../data/'

WAV = VD_PATH + 'wav/'

#wavFiles = glob.glob(WAV+'*')
mfccFiles = glob.glob(DATA_PATH + '*/*mfcc')

CORES = 30

def chunkify(seq, num):
	avg = len(seq) / float(num)
	out = []
	last = 0.0
	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg
	return out

l = chunkify(mfccFiles,CORES)

outFolder = '../data/splits/mfcc/'

for i in xrange(len(l)):
	pickle.dump(l[i],open(outFolder+str(i)+'.p','w'))





