import os
import sys
import glob
import cPickle as pickle
import numpy as np


TRAIN_DATA = '../data/trainData/'
PREDICTIONS = '../data/predictions/'

val_fold = int(sys.argv[1])

# val_fold = 3

all = [0,1,2,3,4]
train_folds = [x for x in all if x!=val_fold]



X_train = pickle.load(open(TRAIN_DATA+str(train_folds[0])+'X.p','r'))
y_train = pickle.load(open(TRAIN_DATA+str(train_folds[0])+'y.p','r'))

for e in train_folds[1:]:
	X_train = np.vstack((X_train,pickle.load(open(TRAIN_DATA+str(e)+'X.p','r'))))
	y_train = np.hstack((y_train,pickle.load(open(TRAIN_DATA+str(e)+'y.p','r'))))

X_val = pickle.load(open(TRAIN_DATA+str(val_fold)+'X.p','r'))
y_val = pickle.load(open(TRAIN_DATA+str(val_fold)+'y.p','r'))



from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train, y_train)
#clf.score(X_val,y_val)
t1 = clf.predict(X_val)
pickle.dump(t1,open(PREDICTIONS+str(val_fold)+'_y.p','r'))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100,n_jobs=-2,class_weight='balanced')
clf.fit(X_train, y_train)
t1 = clf.predict(X_val)
pickle.dump(t1,open(PREDICTIONS+str(val_fold)+'_y.p','r'))

#clf.score(X_val,y_val)











