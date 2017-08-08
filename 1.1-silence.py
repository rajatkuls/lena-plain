import os
import glob
import sys
import ast
import cPickle as pickle
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioTrainTest as aT
import numpy as np

# Given a collection of wavs, generate an SVM classifier on energy

#split = 0
'''
WAV_PATH = '../VanDam/wav/'
TEST_PATH = '../VanDam1/'
DATA_PATH = '../data/'

stWin = 0.05
stStep = 0.05

def getTotalAudio(folder_to_wavs):
	wavFiles = glob.glob(folder_to_wavs+'*')
	total = np.asarray([])
	FirstFs = audioBasicIO.readAudioFile(wavFiles[0])[0]
	for wav in wavFiles:
		[Fs,x] = audioBasicIO.readAudioFile(wav)
		if Fs != FirstFs:
			print >> sys.stderr, "Inconsistent bitrates in files, found " + str(FirstFs)+" and "+str(Fs)
		total = np.concatenate((total,x))
	return (Fs,total)



[Fs,x] = getTotalAudio(WAV_PATH)
ShortTermFeatures = aF.stFeatureExtraction(x, Fs, stWin * Fs, stStep * Fs)
# ShortTermFeatures = pickle.load(open('../data/ShortTermFeatures.p','r'))

EnergySt = ShortTermFeatures[1, :]
E = np.sort(EnergySt)
L1 = int(len(E) / 10) 
T1 = np.mean(E[0:L1]) + 0.000000000000001 
T2 = np.mean(E[-L1:-1]) + 0.000000000000001                # compute "higher" 10% energy threshold
Class1 = ShortTermFeatures[:, np.where(EnergySt <= T1)[0]]         # get all features that correspond to low energy
Class2 = ShortTermFeatures[:, np.where(EnergySt >= T2)[0]]         # get all features that correspond to high energy
featuresSS = [Class1.T, Class2.T]                                    # form the binary classification task and ...
[featuresNormSS, MEANSS, STDSS] = aT.normalizeFeatures(featuresSS)
SVM = aT.trainSVM(featuresNormSS, 1.0)
'''


testFiles = glob.glob('../data/VanDam1/*/*wav')


# Make the output stm under the below parameters :


SIL = 'SIL'
smooth = str(sys.argv[1])
weight = str(sys.argv[2])

#t1 = pickle.load(open(DATA_PATH+'/splits/wav/'+str(split)+'.p'))

testFiles = ['../'+x for x in testFiles]
#wavFiles = [wavFiles[0]]
os.chdir('pyAudioAnalysis')

def getStmPath(e):
	return e.replace('wav','stm')

def getSilentRow(filename,start,end):
	return filename+' SIL '+filename+'_SIL '+str(start)+' '+str(end)

def getSpeechRow(filename,start,end):
	return filename+' SPE '+filename+'_SPE '+str(start)+' '+str(end)

for e in testFiles:
	t1 = os.popen('python audioAnalysis.py silenceRemoval -i ' + e + ' --smoothing '+smooth+' --weight '+weight).read()
	segments = ast.literal_eval(t1)
	stmFile = getStmPath(e) + '_'+smooth+'_'+weight+'_1.1'
	filename = e[e.rfind('/')+1:]
	filename = filename[:filename.find('.')]
	#makeNewSeg
	newSeg = []
	if segments[0][0]>0:
		newSeg.append((True,0,segments[0][0]))
	newSeg.append((False,segments[0][0],segments[0][1]))
	for i in xrange(1,len(segments)):
		newSeg.append((True,segments[i-1][1],segments[i][0]))
		newSeg.append((False,segments[i][0],segments[i][1]))
	f = os.popen('ffprobe -i '+e+' -show_entries format=duration -v quiet -of csv="p=0"').read()[:-1]
	endTime = float(f)
	if endTime>newSeg[-1][2]:
		newSeg.append((True,newSeg[-1][2],endTime))
	curTime = 0.0
	curRow = 0
	#oldStm = open(stmFile,'r').read().split('\n')[:-1]
	newStm = ''
	for i in xrange(len(newSeg)):
		if newSeg[i][0]:
			newStm+=getSilentRow(filename,newSeg[i][1],newSeg[i][2])+'\n'
			#makeSilent
		else:
			newStm+=getSpeechRow(filename,newSeg[i][1],newSeg[i][2])+'\n'
	outFile = open(stmFile,'w')
	outFile.write(newStm)
	outFile.close()





