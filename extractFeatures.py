import os
import glob
import sys
import ast
import cPickle as pickle
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioTrainTest as aT
import numpy as np
import labelTools
import helper
import pdb

# Feature extraction. Assume for both 159 and daylong
# smooth 1.0 weight 0.2
# TODO figure out if these are just features, to be further converted to per-fold arrays.
# This will yield a generator to sklearn

os.system('mkdir -p ../data/features/')


WAV_PATH = '../data/VanDam/'
WAV1_PATH = '../data/VanDam1/'	# I can't write wavs to original location
DATA_PATH = '../data/'

FOLDS_PATH = '../data/folds/VanDam/portion*'
FOLDS_PATH1 = '../data/folds/VanDam1/portion*'


FRAME_WIDTH = 0.001
INV_FRAME_WIDTH = 1000
MIN_DUR = 0.0001

shortSpeechDict = labelTools.label_map
daylongNearSpeechDict = labelTools.label_map1_near
daylongFarSpeechDict = labelTools.label_map1_far
daylongAllSpeechDict = dict(daylongNearSpeechDict,**daylongFarSpeechDict)


def silMapFn(label,speechDict):
	if label in speechDict.keys():
		return 1
	return 0

label_map = labelTools.label_map
# contains CHI, MOT, FAT, OCH, OAD, OTH

classDict 	 = {'CHI':1,'MOT':2,'FAT':3,'OCH':4,'OAD':5,'OTH':0}
classDictBinary  = {'CHI':1,'MOT':2,'FAT':2,'OCH':1,'OAD':2,'OTH':0} # child adult
classDictTernary = {'CHI':1,'MOT':2,'FAT':3,'OCH':1,'OAD':0,'OTH':0} # child male female

def classMapFn(label,typeSpeechDict):
	if label in typeSpeechDict.keys():
		label = typeSpeechDict[label]
	if label in classDict.keys():
		return classDict[label]
	return 0

def classMapBinaryFn(label,typeSpeechDict):
        if label in typeSpeechDict.keys():
                label = typeSpeechDict[label]
        if label in classDictBinary.keys():
                return classDictBinary[label]
        return 0

def classMapTernaryFn(label,typeSpeechDict):
        if label in typeSpeechDict.keys():
                label = typeSpeechDict[label]
        if label in classDictTernary.keys():
                return classDictTernary[label]
        return 0


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

def basename(filename):
	return filename[filename.rfind('/')+1:filename.rfind('.')]

def getFilesFromPortion(portion):
	t1 = open(portion,'r').read().split('\n')[:-1]
	return [x[:x.find('.')] for x in t1]

foldFileList  = []
foldFileList1 = []

for portion in glob.glob(FOLDS_PATH):
	foldFileList.append([WAV_PATH+x+'.wav' for x in getFilesFromPortion(portion)])

for portion in glob.glob(FOLDS_PATH1):
	foldFileList1.append([WAV1_PATH+x[:4]+'/'+x+'.wav' for x in getFilesFromPortion(portion)])


# wavs  = glob.glob(WAV_PATH)
# wavs1 = glob.glob(WAV1_PATH)

#Energy block

stWin = 0.05
stStep = 0.05



def getTotalAudio(folder_to_wavs):
        total = np.asarray([])
	print folder_to_wavs
        FirstFs = audioBasicIO.readAudioFile(folder_to_wavs[0])[0]
        for wav in folder_to_wavs:
                [Fs,x] = audioBasicIO.readAudioFile(wav)
                if Fs != FirstFs:
                        print >> sys.stderr, "Inconsistent bitrates in files, found " + str(FirstFs)+" and "+str(Fs)
                total = np.concatenate((total,x))
        return (Fs,total)

def getTotalEnergyVector(folder_to_wavs):     # given a single list of wav paths, return their aggregate 10% vector
	[Fs,x] = getTotalAudio(folder_to_wavs)
	ShortTermFeatures = aF.stFeatureExtraction(x, Fs, stWin * Fs, stStep * Fs)
	EnergySt = ShortTermFeatures[1, :]
	E = np.sort(EnergySt)
	L1 = int(len(E) / 10) 
	T1 = np.mean(E[0:L1]) + 0.000000000000001 
	T2 = np.mean(E[-L1:-1]) + 0.000000000000001                # compute "higher" 10% energy threshold
	Class1 = ShortTermFeatures[:, np.where(EnergySt <= T1)[0]]         # get all features that correspond to low energy
	# Class1 = ShortTermFeatures[1,:][np.where(EnergySt <= T1)[0]]         # purely energy
	Class2 = ShortTermFeatures[:, np.where(EnergySt >= T2)[0]]         # get all features that correspond to high energy
	# Class2 = ShortTermFeatures[1,:][np.where(EnergySt >= T2)[0]]         # purely energy
	featuresSS = [Class1.T, Class2.T]                                  # form the binary classification task
	[featuresNormSS, MEANSS, STDSS] = aT.normalizeFeatures(featuresSS) # normalize to 0-mean 1-std
	[X,y] = featureListToVectors(featuresNormSS)
	return X,y,Fs

def getStVectorPerWav(wavFile,stWin,stStep):     # given a wav, get entire sT features 
        [Fs,x] = getTotalAudio([wavFile])
        ShortTermFeatures = aF.stFeatureExtraction(x, Fs, stWin * Fs, stStep * Fs)
        [featuresNormSS, MEANSS, STDSS] = aT.normalizeFeatures([ShortTermFeatures]) # normalize to 0-mean 1-std
        [X,y] = featureListToVectors([featuresNormSS])
        return X,y,Fs

def getRawStVectorPerWav(wavFile,stWin,stStep):
	[Fs,x] = audioBasicIO.readAudioFile(wavFile)
	return aF.stFeatureExtraction(x, Fs, stWin * Fs, stStep * Fs)
	

def getLabelPerWav(stmFile,stStep,labelsMapFn,labelsDict):
	seg = helper.convertToFrames(stmFile)
	labels = [labelsMapFn(x[2],labelsDict) for x in seg]
	return labels[::int(stStep*INV_FRAME_WIDTH)]

def getRawStDataPerWav(wavFile,stWin,stStep,labelsMapFn):
	X = getRawStVectorPerWav(wavFile,stWin,stStep)
	y = getLabelPerWav(getStmForWav(wavFile),stStep,labelsMapFn)
	X = X[:,:len(y)]
	return X,np.asarray(y)
	

def yieldEnergyFeats(foldFileList):
	for i in xrange(len(foldFileList)):
		trainlist = []
		for j in xrange(len(foldFileList)):
			if i!=j:
				trainlist.extend(foldFileList[j])
		testlist = [WAV_PATH+x+'.wav' for x in foldFileList[i]]
		trainlist = [WAV_PATH+x+'.wav' for x in trainlist]
		yield getTotalEnergyVector(trainlist),getTotalEnergyVector(testlist)

def getStmForWav(wavFile):
	return wavFile[:wavFile.rfind('.')]+'.stm'

def getRootName(wavFile):
	wavFile = wavFile[wavFile.rfind('/')+1:]
	return wavFile[:wavFile.find('.')]
	


'''
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1,n_jobs=-2)

g = yieldEnergyFeats(foldFileList)
t1 = g.next()
X_train = t1[0][0]
y_train = t1[0][1]

t2 = ['../data/VanDam1/BS80/BS80_030213.wav']
h = getTotalEnergyVector(t2)
#t3 = h.next()
X_test = h[0]
y_test = h[1]

clf.fit(X_train,y_train)
clf.score(X_test,y_test)
'''
if __name__ == "__main__":
	os.system('mkdir -p ../data/features/energy/VanDam')
	os.system('mkdir -p ../data/features/energy/VanDam1')
	print foldFileList1
	for i in xrange(len(foldFileList1)):
		print i
		print "IIIIIIIII"
		os.system('mkdir -p ../data/features/st/VanDam1/'+str(i))
		for wavFile in foldFileList1[i]:
			print wavFile
			# X = getRawStVectorPerWav(wavFile,stWin,stStep)
			# y_sil_near = getLabelPerWav(getStmForWav(wavFile),stStep,silMapFn,daylongNearSpeechDict)
			# y_sil_all = getLabelPerWav(getStmForWav(wavFile),stStep,silMapFn,daylongAllSpeechDict)
			# y_class = getLabelPerWav(getStmForWav(wavFile),stStep,classMapFn,daylongNearSpeechDict)
			y_class2 = getLabelPerWav(getStmForWav(wavFile),stStep,classMapBinaryFn,daylongNearSpeechDict)
			y_class3 = getLabelPerWav(getStmForWav(wavFile),stStep,classMapTernaryFn,daylongNearSpeechDict)
			# pickle.dump(X,open('../data/features/st/VanDam1/'+str(i)+'/'+getRootName(wavFile)+'_X','w'))
			# pickle.dump(y_sil_near,open('../data/features/st/VanDam1/'+str(i)+'/'+getRootName(wavFile)+'_ysil_near','w'))
			# pickle.dump(y_sil_all,open('../data/features/st/VanDam1/'+str(i)+'/'+getRootName(wavFile)+'_ysil_all','w'))
			# pickle.dump(y_class,open('../data/features/st/VanDam1/'+str(i)+'/'+getRootName(wavFile)+'_yclass','w'))
			pickle.dump(y_class2,open('../data/features/st/VanDam1/'+str(i)+'/'+getRootName(wavFile)+'_yclass2','w'))
			pickle.dump(y_class3,open('../data/features/st/VanDam1/'+str(i)+'/'+getRootName(wavFile)+'_yclass3','w'))



	print foldFileList
	for i in xrange(len(foldFileList)):
		print i
		print "normal"
		os.system('mkdir -p ../data/features/st/VanDam/'+str(i))
		for wavFile in foldFileList[i]:
			print wavFile
			# X = getRawStVectorPerWav(wavFile,stWin,stStep)
			# y_sil = getLabelPerWav(getStmForWav(wavFile),stStep,silMapFn,shortSpeechDict)
			# y_class = getLabelPerWav(getStmForWav(wavFile),stStep,classMapFn,shortSpeechDict)
			y_class2 = getLabelPerWav(getStmForWav(wavFile),stStep,classMapBinaryFn,shortSpeechDict)
			y_class3 = getLabelPerWav(getStmForWav(wavFile),stStep,classMapTernaryFn,shortSpeechDict)
			# pickle.dump(X,open('../data/features/st/VanDam/'+str(i)+'/'+getRootName(wavFile)+'_X','w'))
			# pickle.dump(y_sil,open('../data/features/st/VanDam/'+str(i)+'/'+getRootName(wavFile)+'_ysil','w'))
			# pickle.dump(y_class,open('../data/features/st/VanDam/'+str(i)+'/'+getRootName(wavFile)+'_yclass','w'))
			pickle.dump(y_class2,open('../data/features/st/VanDam/'+str(i)+'/'+getRootName(wavFile)+'_yclass2','w'))
			pickle.dump(y_class3,open('../data/features/st/VanDam/'+str(i)+'/'+getRootName(wavFile)+'_yclass3','w'))




