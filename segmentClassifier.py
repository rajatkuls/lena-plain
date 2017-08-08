import os
import glob
from pyAudioAnalysis import audioTrainTest as aT

labels = glob.glob('../data/classes/*')

aT.featureAndTrain(labels, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svm1000", False)
aT.featureAndTrain(labels, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svm1000_scr", False)
aT.featureAndTrain(labels, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "rf", "rf1000", False)
aT.featureAndTrain(labels, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knn1000", False)
aT.featureAndTrain(labels, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm_rbf", "svmrbf1000", False)





