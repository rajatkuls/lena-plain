from __future__ import print_function

import numpy as np
import os
import glob
import cPickle as pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score





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

'''
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
'''

X = np.array([]).reshape(FEAT_DIM,0)
y = np.array([])
lengths = []
for i in xrange(len(foldFileList)):
        length=0
        for f in foldFileList[i]:
                filename = ST_FEATUREPATH+str(i)+'/'+f
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


# 1 near
# 2 all = near+far

X1 = np.array([]).reshape(FEAT_DIM,0)
y1 = np.array([])
y2 = np.array([])
lengths1 = []
for i in xrange(len(foldFileList1)):
	print 'i '+str(i)
        length=0
        for f in foldFileList1[i]:
		print 'fold '+str(f)
                filename = ST_FEATUREPATH1+str(i)+'/'+f
                y1_temp = pickle.load(open(filename+'_y'+'sil_near','r'))
                y2_temp = pickle.load(open(filename+'_y'+'sil_all','r'))
                x1 = pickle.load(open(filename+'_X','r'))
                y1_temp = np.asarray(y1_temp)
                y2_temp = np.asarray(y2_temp)
		assert y1_temp.shape[0]==y2_temp.shape[0]
                if y1_temp.shape[0] < x1.shape[1]:
                        x1 = x1[:,:y1_temp.shape[0]]
                else:
                        y1_temp = y1_temp[:x1.shape[1]]
                        y2_temp = y2_temp[:x1.shape[1]]
                length+=y1_temp.shape[0]
		assert y1_temp.shape[0]==x1.shape[1]
                X1 = np.hstack((X1,x1))
                y1 = np.concatenate((y1,y1_temp))
                y2 = np.concatenate((y2,y2_temp))
        lengths1.append(length)

boundaries1 = [sum(lengths1[:x]) for x in xrange(len(lengths1)+1)]
idxs1 = [(boundaries1[i],boundaries1[i+1]) for i in xrange(len(boundaries1)-1)]
X1 = X1.T

'''
# all (including Far)
X2 = np.array([]).reshape(FEAT_DIM,0)
y2 = np.array([])
lengths2 = []
for i in xrange(len(foldFileList1)):
        print 'i '+str(i)
        length=0
        for f in foldFileList1[i]:
                print 'fold '+str(f)
                filename = ST_FEATUREPATH1+str(i)+'/'+f
                y1_temp = pickle.load(open(filename+'_y'+labelType,'r'))
                x1 = pickle.load(open(filename+'_X','r'))
                y1_temp = np.asarray(y1_temp)
                if y1_temp.shape[0] < x1.shape[1]:
                        x1 = x1[:,:y1_temp.shape[0]]
                else:
                        y1_temp = y1_temp[:x1.shape[1]]
                length+=y1_temp.shape[0]
                assert y1_temp.shape[0]==x1.shape[1]
                X2 = np.hstack((X2,x1))
                y2 = np.concatenate((y2,y1_temp))
        lengths2.append(length)

boundaries2 = [sum(lengths2[:x]) for x in xrange(len(lengths2)+1)]
idxs2 = [(boundaries2[i],boundaries2[i+1]) for i in xrange(len(boundaries2)-1)]
X2 = X2.T
'''


def yieldIdxs(foldFileList):
	for i in xrange(len(foldFileList)):
		trainidxs = []
		for j in xrange(len(foldFileList)):
			if i!=j:
				trainidxs += range(idxs[j][0],idxs[j][1])
		testidxs = range(idxs[i][0],idxs[i][1])
		yield trainidxs,testidxs


def yieldIdxs1(foldFileList):
        for i in xrange(len(foldFileList)):
                trainidxs = []
                for j in xrange(len(foldFileList)):
                        if i!=j:
                                trainidxs += range(idxs1[j][0],idxs1[j][1])
                testidxs = range(idxs1[i][0],idxs1[i][1])
                yield trainidxs,testidxs


def featureListToVectors(featureList):
        X = np.array([])
        Y = np.array([])
        for i, f in enumerate(featureList):
                if i == 0:
                        X = f
                        Y = i * np.ones((len(f), 1))
                else:
                        X = np.vstack((X, f))
                        Y = np.append(Y, i * np.ones((len(f), 1)))
        return (X, Y)


def getTotalEnergyVector(X_in):     # given a single list of wav paths, return their aggregate 10% vector
        EnergySt = X_in.T[1, :]
        E = np.sort(EnergySt)
        L1 = int(len(E) / 10)
        T1 = np.mean(E[0:L1]) + 0.000000000000001
        T2 = np.mean(E[-L1:-1]) + 0.000000000000001                # compute "higher" 10% energy threshold
        Class1 = X_in.T[:, np.where(EnergySt <= T1)[0]]         # get all features that correspond to low energy
        # Class1 = ShortTermFeatures[1,:][np.where(EnergySt <= T1)[0]]         # purely energy
        Class2 = X_in.T[:, np.where(EnergySt >= T2)[0]]         # get all features that correspond to high energy
        # Class2 = ShortTermFeatures[1,:][np.where(EnergySt >= T2)[0]]         # purely energy
        featuresSS = [Class1.T, Class2.T]                                  # form the binary classification task
        # [featuresNormSS, MEANSS, STDSS] = aT.normalizeFeatures(featuresSS) # normalize to 0-mean 1-std
        [X,y] = featureListToVectors(featuresSS)
        return X,y



scores = ['f1','precision','recall']
param_grid = {'n_estimators': [1,10,100], 'max_features': ['auto', 'sqrt', 'log2']}


param_score_dict = {}
for i in scores:
	param_score_dict[i] = {}


param_list = list(ParameterGrid(param_grid))

best_f1 = -1
best_f1_i = -1

for p,params in enumerate(param_list):
	params['n_jobs'] = 28
	splitGenerator = yieldIdxs(foldFileList)
	f1_list = []
	precision_list = []
	recall_list = []
	for i in xrange(len(foldFileList)):
		rfc = RandomForestClassifier(**params)
		rfc_test = RandomForestClassifier(**params)
		trainIdx,testIdx = splitGenerator.next()
		X_train = X[trainIdx]
		X_test  = X[testIdx]
		X_train_energy,y_train_energy = getTotalEnergyVector(X_train)
		X_test_energy,y_test_energy = getTotalEnergyVector(X_test)
		fit1 = rfc.fit(X_train_energy,y_train_energy)
		fit2 = rfc_test.fit(X_test_energy,y_test_energy)
		y_true = rfc_test.predict(X_test)
		y_pred = rfc.predict(X_test)
		f1_list.append(f1_score(y_true,y_pred,average='weighted'))
		if f1_list[-1] > best_f1:
			best_f1   = f1_list[-1]
			best_f1_i = i
		precision_list.append(precision_score(y_true,y_pred,average='weighted'))
		recall_list.append(recall_score(y_true,y_pred,average='weighted'))
	param_score_dict['f1'][p] 	=	np.mean(f1_list)
	param_score_dict['precision'][p]= 	np.mean(precision_list)
	param_score_dict['recall'][p] 	= 	np.mean(recall_list)

tempMax = -1
best_param_idx = None
for k,v in param_score_dict['f1'].items():
	if v>tempMax:
		tempMax=v
		best_param_idx = k

print 'Best params over F1 are: ' + str(param_list[best_param_idx])
print 'Corresponding F1       : ' + str(param_score_dict['f1'][best_param_idx])
print 'Corresponding precision: ' + str(param_score_dict['precision'][best_param_idx])
print 'Corresponding recall   : ' + str(param_score_dict['recall'][best_param_idx])

print 'Given these params, training a model over all short.'

short_rfc = RandomForestClassifier(**param_list[best_param_idx])
short_rfc.fit(X,y)
y_daylong_pred = short_rfc.predict(X1)


print 'Testing this over all daylong (near+far)'
print 'F1       : ' + str(f1_score(y1,y_daylong_pred))
print 'precision: ' + str(precision_score(y1,y_daylong_pred))
print 'recall   : ' + str(recall_score(y1,y_daylong_pred))

print 'Testing this over all daylong (only near)'
print 'F1       : ' + str(f1_score(y2,y_daylong_pred))
print 'precision: ' + str(precision_score(y2,y_daylong_pred))
print 'recall   : ' + str(recall_score(y2,y_daylong_pred))


print 'Also training the best performing subset over short.'
short_subset_rfc = RandomForestClassifier(**param_list[best_param_idx])
splitGenerator = yieldIdxs(foldFileList)
for i in xrange(best_f1_i):
	trainIdx,testIdx = splitGenerator.next()

X_train = X[trainIdx]
X_test  = X[testIdx]
X_train_energy,y_train_energy = getTotalEnergyVector(X_train)
X_test_energy,y_test_energy = getTotalEnergyVector(X_test)
best_subset_short_rfc = rfc.fit(X_train_energy,y_train_energy)


y_daylong_pred = best_subset_short_rfc.predict(X1)
print 'Testing this over all daylong (near+far)'
print 'F1       : ' + str(f1_score(y1,y_daylong_pred))
print 'precision: ' + str(precision_score(y1,y_daylong_pred))
print 'recall   : ' + str(recall_score(y1,y_daylong_pred))

print 'Testing this over all daylong (only near)'
print 'F1       : ' + str(f1_score(y2,y_daylong_pred))
print 'precision: ' + str(precision_score(y2,y_daylong_pred))
print 'recall   : ' + str(recall_score(y2,y_daylong_pred))













print "Working on daylong recordings now"

param_score_dict1 = {}
for i in scores:
        param_score_dict1[i] = {}


param_list = list(ParameterGrid(param_grid))

best_f1_daylong = -1
best_f1_i_daylong = -1


foldFileList2 = foldFileList1[:]
foldFileList1 = [[x[0] for x in foldFileList2]]


for p,params in enumerate(param_list):
	params['n_jobs'] = 28
	splitGenerator = yieldIdxs1(foldFileList1)
	f1_list = []
	precision_list = []
	recall_list = []
	for i in xrange(len(foldFileList1)):
		rfc = RandomForestClassifier(n_jobs=28)
		rfc_test = RandomForestClassifier(n_jobs=28)
		trainIdx,testIdx = splitGenerator.next()
		X_train = X1[trainIdx]
		X_test  = X1[testIdx]
		X_train_energy,y_train_energy = getTotalEnergyVector(X_train)
		X_test_energy,y_test_energy = getTotalEnergyVector(X_test)
		fit1 = rfc.fit(X_train_energy,y_train_energy)
		fit2 = rfc_test.fit(X_test_energy,y_test_energy)
		y_true = rfc_test.predict(X_test)
		y_pred = rfc.predict(X_test)
		f1_list.append(f1_score(y_true,y_pred,average='weighted'))
		precision_list.append(precision_score(y_true,y_pred,average='weighted'))
		recall_list.append(recall_score(y_true,y_pred,average='weighted'))
	param_score_dict1['f1'][p]       =       np.mean(f1_list)
	param_score_dict1['precision'][p]=       np.mean(precision_list)
	param_score_dict1['recall'][p]   =       np.mean(recall_list)

tempMax = -1
best_param_idx1 = None
for k,v in param_score_dict1['f1'].items():
        if v>tempMax:
                tempMax=v
                best_param_idx1 = k

print 'Best params over F1 are: ' + str(param_list[best_param_idx1])
print 'Corresponding F1       : ' + str(param_score_dict1['f1'][best_param_idx1])
print 'Corresponding precision: ' + str(param_score_dict1['precision'][best_param_idx1])
print 'Corresponding recall   : ' + str(param_score_dict1['recall'][best_param_idx1])


print 'Training global model with these parameters. First for near'
daylong_rfc_near = RandomForestClassifier(**param_list[best_param_idx])
daylong_rfc_near.fit(X1,y1)
y_daylong_pred = daylong_rfc_near.predict(X1)


print 'Testing this over all daylong (near)'
print 'F1       : ' + str(f1_score(y1,y_daylong_pred))
print 'precision: ' + str(precision_score(y1,y_daylong_pred))
print 'recall   : ' + str(recall_score(y1,y_daylong_pred))

print 'Testing this over all daylong (near+far)'
print 'F1       : ' + str(f1_score(y2,y_daylong_pred))
print 'precision: ' + str(precision_score(y2,y_daylong_pred))
print 'recall   : ' + str(recall_score(y2,y_daylong_pred))

y_short_pred_near = daylong_rfc_near.predict(X)

print 'Testing this over all short' 
print 'F1       : ' + str(f1_score(y,y_short_pred_near))
print 'precision: ' + str(precision_score(y,y_short_pred_near))
print 'recall   : ' + str(recall_score(y,y_short_pred_near))


print 'Training global model with these parameters. This time near+far'
daylong_rfc_all = RandomForestClassifier(**param_list[best_param_idx])
daylong_rfc_all.fit(X1,y2)
y_daylong_pred = daylong_rfc_near.predict(X1)


print 'Testing this over all daylong (near)'
print 'F1       : ' + str(f1_score(y1,y_daylong_pred))
print 'precision: ' + str(precision_score(y1,y_daylong_pred))
print 'recall   : ' + str(recall_score(y1,y_daylong_pred))

print 'Testing this over all daylong (near+far)'
print 'F1       : ' + str(f1_score(y2,y_daylong_pred))
print 'precision: ' + str(precision_score(y2,y_daylong_pred))
print 'recall   : ' + str(recall_score(y2,y_daylong_pred))

y_short_pred_all = daylong_rfc_all.predict(X)

print 'Testing this over all short'
print 'F1       : ' + str(f1_score(y,y_short_pred_all))
print 'precision: ' + str(precision_score(y,y_short_pred_all))
print 'recall   : ' + str(recall_score(y,y_short_pred_all))




# heldout
print "Working on heldout BS80 now"

param_score_dicth = {}
for i in scores:
        param_score_dicth[i] = {}


best_f1_heldout = -1
best_f1_i_heldout = -1


foldFileList2 = foldFileList1[:]
foldFileList1 = [[x[0] for x in foldFileList2[1:]]]

X1h = X1[854120:]
y1h = y1[854120:]
y2h = y2[854120:]


X1_BS80 = X1[:854120]
y1_BS80 = y1[:854120]
y2_BS80 = y2[:854120]





for p,params in enumerate(param_list):
        params['n_jobs'] = 28
	rfc = RandomForestClassifier(n_jobs=28)
	rfc_test = RandomForestClassifier(n_jobs=28)
	X_train = X1h
	X_test  = X1_BS80
	X_train_energy,y_train_energy = getTotalEnergyVector(X_train)
	X_test_energy,y_test_energy = getTotalEnergyVector(X_test)
	fit1 = rfc.fit(X_train_energy,y_train_energy)
	fit2 = rfc_test.fit(X_test_energy,y_test_energy)
	y_true = rfc_test.predict(X_test)
	y_pred = rfc.predict(X_test)
        param_score_dicth['f1'][p]       =       np.mean(f1_score(y_true,y_pred,average='weighted'))
        param_score_dicth['precision'][p]=       np.mean(precision_score(y_true,y_pred,average='weighted'))
        param_score_dicth['recall'][p]   =       np.mean(recall_score(y_true,y_pred,average='weighted'))




tempMax = -1
best_param_idxh = None
for k,v in param_score_dicth['f1'].items():
        if v>tempMax:
                tempMax=v
                best_param_idxh = k

print 'Best params over F1 are: ' + str(param_list[best_param_idxh])
print 'Corresponding F1       : ' + str(param_score_dicth['f1'][best_param_idxh])
print 'Corresponding precision: ' + str(param_score_dicth['precision'][best_param_idxh])
print 'Corresponding recall   : ' + str(param_score_dicth['recall'][best_param_idxh])


print 'Training global model with these parameters. Using near'
daylong_rfc_near_h = RandomForestClassifier(**param_list[best_param_idx])
daylong_rfc_near_h.fit(X1h,y2h)
y_daylong_near_h = daylong_rfc_near_h.predict(X1h)


print 'Testing this over all daylong (near)'
print 'F1       : ' + str(f1_score(y1,y_daylong_))
print 'precision: ' + str(precision_score(y1,y_daylong_pred))
print 'recall   : ' + str(recall_score(y1,y_daylong_pred))

print 'Testing this over all daylong (near+far)'
print 'F1       : ' + str(f1_score(y2,y_daylong_pred))
print 'precision: ' + str(precision_score(y2,y_daylong_pred))
print 'recall   : ' + str(recall_score(y2,y_daylong_pred))

y_short_pred_near_h = daylong_rfc_near_h.predict(X)

print 'Testing this over all short'
print 'F1       : ' + str(f1_score(y,y_short_pred_near_h))
print 'precision: ' + str(precision_score(y,y_short_pred_near_h))
print 'recall   : ' + str(recall_score(y,y_short_pred_near_h))








# SVM section
scores = ['f1','precision','recall']

param_grid_svm = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100]},
                    {'kernel': ['linear'], 'C': [1, 10, 100]}]



param_score_dict1_svm = {}
for i in scores:
        param_score_dict_svm[i] = {}


param_list_svm = list(ParameterGrid(param_grid_svm))

best_f1 = -1
best_f1_i = -1


for p,params in enumerate(param_list_svm):
        splitGenerator = yieldIdxs1(foldFileList1)
        f1_list = []
        precision_list = []
        recall_list = []
        for i in xrange(len(foldFileList1)):
                rfc = SVC(**params)
                rfc_test = SVC(**params)
                trainIdx,testIdx = splitGenerator.next()
                X_train = X1[trainIdx]
                X_test  = X1[testIdx]
                X_train_energy,y_train_energy = getTotalEnergyVector(X_train)
                X_test_energy,y_test_energy = getTotalEnergyVector(X_test)
                fit1 = rfc.fit(X_train_energy,y_train_energy)
                fit2 = rfc_test.fit(X_test_energy,y_test_energy)
                y_true = rfc_test.predict(X_test)
                y_pred = rfc.predict(X_test)
                f1_list.append(f1_score(y_true,y_pred,average='weighted'))
                precision_list.append(precision_score(y_true,y_pred,average='weighted'))
                recall_list.append(recall_score(y_true,y_pred,average='weighted'))
        param_score_dict1_svm['f1'][p]       =       np.mean(f1_list)
        param_score_dict1_svm['precision'][p]=       np.mean(precision_list)
        param_score_dict1_svm['recall'][p]   =       np.mean(recall_list)

tempMax = -1
best_param_idx1 = None
for k,v in param_score_dict1['f1'].items():
        if v>tempMax:
                tempMax=v
                best_param_idx1 = k

print 'Best params over F1 are: ' + str(param_list[best_param_idx1])
print 'Corresponding F1       : ' + str(param_score_dict1['f1'][best_param_idx1])
print 'Corresponding precision: ' + str(param_score_dict1['precision'][best_param_idx1])
print 'Corresponding recall   : ' + str(param_score_dict1['recall'][best_param_idx1])










exit(0)
#get completevec and mark rows:

X = np.array([]).reshape(FEAT_DIM,0)
y = np.array([])
lengths = []
for i in xrange(len(foldFileList)):
	length=0
	for f in foldFileList[i]:
		filename = ST_FEATUREPATH+str(i)+'/'+f
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
	for i in xrange(len(foldFileList)):
		trainidxs = []
		for j in xrange(len(foldFileList)):
			if i!=j:
				trainidxs += range(idxs[j][0],idxs[j][1])
		testidxs = range(idxs[i][0],idxs[i][1])
		yield trainidxs,testidxs











exit(0)


tuned_parameters = [{'n_estimators': [1,10,100],
			'max_features': ['auto', 'sqrt', 'log2']}]




rfc = RandomForestClassifier(n_jobs=-2)


scores = ['f1']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(rfc, tuned_parameters, cv=yieldIdxs(),
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



pickle.dump(clf,open('speech159rfc.p','w'))





