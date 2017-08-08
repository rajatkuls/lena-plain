from __future__ import print_function
import numpy as np
import os
import glob
import cPickle as pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC




ST_FEATUREPATH = '../data/features/st/VanDam/'
ST_FEATUREPATH1 = '../data/features/st/VanDam1/'

FOLDS_PATH = '../data/folds/VanDam/portion*'
FOLDS_PATH1 = '../data/folds/VanDam1/portion*'


WAV_PATH = '../data/VanDam/'
WAV1_PATH = '../data/VanDam1/'  # I can't write wavs to original location
DATA_PATH = '../data/'

FEAT_DIM = 34

labelType = 'sil'



def getFilesFromPortion(portion):
        t1 = open(portion,'r').read().split('\n')[:-1]
        return [x[:x.find('.')] for x in t1]

foldFileList  = []
foldFileList1 = []

for portion in glob.glob(FOLDS_PATH):
        foldFileList.append([x for x in getFilesFromPortion(portion)])

for portion in glob.glob(FOLDS_PATH1):
        foldFileList1.append([x for x in getFilesFromPortion(portion)])


folders = glob.glob(ST_FEATUREPATH+'*')

def getStFeatsArray(listOfFiles,labelType):
        X = np.array([]).reshape(0,FEAT_DIM)
        y = np.array([])
	for i in listOfFiles:
		y1 = pickle.load(open(i+'_y'+labelType,'r'))
		x1 = pickle.load(open(i+'_X','r'))[:,:len(y1)]
		y1 = np.asarray(y1)
		X = np.concatenate(X,x1)
		y = np.concatenate((y,y1))
	return X,y



#get completevec and mark rows:

X = np.array([]).reshape(FEAT_DIM,0)
y = np.array([])
lengths = []
for i in xrange(len(foldFileList1)):
	length=0
	for f in foldFileList1[i]:
		filename = ST_FEATUREPATH1+str(i)+'/'+f
		y1 = pickle.load(open(filename+'_y'+labelType,'r'))
                x1 = pickle.load(open(filename+'_X','r'))
                y1 = np.asarray(y1)
		if y1.shape[0] < x1.shape[1]:
			x1 = x1[:,:y1.shape[0]]
		else:
			y1 = y1[:x1.shape[1]]
		length+=y1.shape[0]
		X = np.hstack((X,x1))
                y = np.concatenate((y,y1))
	lengths.append(length)

boundaries = [sum(lengths[:x]) for x in xrange(len(lengths)+1)]
idxs = [(boundaries[i],boundaries[i+1]) for i in xrange(len(boundaries)-1)]
X = X.T

def yieldIdxs():
	for i in xrange(len(foldFileList1)):
		trainidxs = []
		for j in xrange(len(foldFileList1)):
			if i!=j:
				trainidxs += range(idxs[j][0],idxs[j][1])
		testidxs = range(idxs[i][0],idxs[i][1])
		yield trainidxs,testidxs

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]




svr = SVC()
clf = GridSearchCV(svr, tuned_parameters,cv=yieldIdxs())


scores = ['f1']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=yieldIdxs(),
                       scoring='%s_macro' % score)
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()



pickle.dump(clf,open('silence-daylongsvm.p','w'))





