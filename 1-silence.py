import os
import glob
import sys
import cPickle as pickle
import ast

# Given smooth, weight as sys inputs, generate STM file for silence/speech in Vandam1 setting

#split = 0
VD1_PATH = '../VanDam1/'
DATA_PATH = '../data/'
SIL = 'SIL'
smooth = str(sys.argv[1])
weight = str(sys.argv[2])

#t1 = pickle.load(open(DATA_PATH+'/splits/wav/'+str(split)+'.p'))

wavFiles = glob.glob('../data/VanDam1/*/*wav')
wavFiles = ['../'+x for x in wavFiles]
#wavFiles = [wavFiles[0]]
os.chdir('pyAudioAnalysis')

def getStmPath(e):
	return e.replace('mp3','stm')

def getSilentRow(filename,start,end):
	return filename+' SIL '+filename+'_SIL '+str(start)+' '+str(end)

def getSpeechRow(filename,start,end):
	return filename+' SPE '+filename+'_SPE '+str(start)+' '+str(end)

for e in mp3Files:
	t1 = os.popen('python audioAnalysis.py silenceRemoval -i ' + e + ' --smoothing '+smooth+' --weight '+weight).read()
	segments = ast.literal_eval(t1)
	stmFile = getStmPath(e) + '_'+smooth+'_'+weight
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





