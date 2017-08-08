import os
import glob
import numpy as np
import cPickle as pickle

DATA_PATH = '../data/'
IGNORE = ['mfcc','hist','wav']
feat_dim = 100

def labelsMap(labelFile):
	if labelFile[labelFile.rfind('.')+1:] == 'SIL':
		return 0
	return 1

def getLabel(histFile):
	cand = glob.glob(t1[:t1.rfind('.')]+'.*')
	labelFile = [x for x in cand if not any([y in x for y in IGNORE])]
	#to find the label of the histFile
	assert len(labelFile) == 1 , "Confirm the ignored files when finding label"
	return labelsMap(labelFile[0])

histFiles = glob.glob(DATA_PATH + '*/*.hist') 

X = np.array([], dtype=np.int64).reshape(0,feat_dim)
y = np.array([], dtype=np.int64).reshape(0)

for e in histFiles:
xvec = np.genfromtxt(open(e,'r'),delimiter = ';').reshape(feat_dim,1)
X = np.concatenate((X,xvec.T))
cand = glob.glob(e[:e.rfind('.')]+'.*')
labelFile = [x for x in cand if not any([y in x for y in IGNORE])]
y = np.concatenate((y,np.asarray([labelsMap(labelFile)])))
pickle.dump(X,'../data/pickles/X_hist_'+str(feat_dim),'w')
pickle.dump(y,'../data/pickles/y_hist_'+str(feat_dim),'w')













