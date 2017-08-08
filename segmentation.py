import os
import glob
import sys
import cPickle as pickle
import ast

# BEST PARAMS

smooth = '1.0'
weight = '0.6'

split = int(sys.argv[1])
#split = 0
VD_PATH = '../VanDam/'
DATA_PATH = '../data/'
SIL = 'SIL'

t1 = pickle.load(open(DATA_PATH+'/splits/wav/'+str(split)+'.p'))

wavFiles = ['../'+x for x in t1]

os.chdir('pyAudioAnalysis')

def makeRowSilent(row,start,end):
        t1 = row[:]
        t1[1] = SIL
        t1[2] = t1[0]+'_'+t1[1]
        t1[3] = str(start)
        t1[4] = str(end)
        return t1[:6]

def makeRowSpeech(row,start,end):
        t1 = row[:]
        t1[3] = str(start)
        t1[4] = str(end)
        return t1

for e in wavFiles:
	t1 = os.popen('python audioAnalysis.py silenceRemoval -i ' + e + ' --smoothing '+smooth+' --weight '+weight).read()
	segments = ast.literal_eval(t1)
	stmFile = e.replace('wav','stm')
	#makeNewSeg
	newSeg = []
	if segments[0][0]>0:
		newSeg.append((True,0,segments[0][0]))
	newSeg.append((False,segments[0][0],segments[0][1]))
	for i in xrange(1,len(segments)):
		newSeg.append((True,segments[i-1][1],segments[i][0]))
		newSeg.append((False,segments[i][0],segments[i][1]))
	f = open(stmFile,'r').read().split('\n')[-2]
	endTime = float(f.split()[4])
	if endTime>newSeg[-1][2]:
		newSeg.append((True,newSeg[-1][2],endTime))
	curTime = 0.0
	curRow = 0
	oldStm = open(stmFile,'r').read().split('\n')[:-1]
	newStm = ''
	for i in xrange(len(newSeg)):
		row = oldStm[curRow].split()
		rowEnd = float(row[4])
		segStart = newSeg[i][1]
		segEnd = newSeg[i][2]
		while segStart>rowEnd:
			curRow+=1
			row = oldStm[curRow].split()
			rowEnd = float(row[4])
		if newSeg[i][0]:
			t1 = makeRowSilent(row,segStart,segEnd)
			newStm += ' '.join(x for x in t1) + '\n'
			#makeSilent
		else:
			#makeSpeech
			if segEnd<rowEnd:
				t1 = makeRowSpeech(row,segStart,segEnd)
				newStm += ' '.join(x for x in t1) + '\n'
				#makeSpeech segStart to segEnd
			elif segEnd==rowEnd:
				#makeSpeech segStart to segEnd
				t1 = makeRowSpeech(row,segStart,segEnd)
				newStm += ' '.join(x for x in t1) + '\n'
				curRow+=1
			else:
				#makeSpeech segStart to rowEnd
				t1 = makeRowSpeech(row,segStart,rowEnd)
				newStm += ' '.join(x for x in t1) + '\n'
				#makeSpeech rowEnd to segEnd
				t1 = makeRowSpeech(row,rowEnd,segEnd)
				newStm += ' '.join(x for x in t1) + '\n'
				curRow+=1
	outFile = open(stmFile.replace('VanDam','data'),'w')
	outFile.write(newStm)
	outFile.close()



