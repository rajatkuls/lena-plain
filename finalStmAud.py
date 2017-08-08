# module to create stm from an input vec

import numpy as np
import cPickle as pickle

stStep = 0.05 # each frame assumed this wide

def stmNewLine(medianame,label,start,end):
	return '\t'.join([medianame,label,medianame+'_'+label,str(start),str(end)]) + '\n'

def writeToStm(y,labelsDict,medianame,outfilename):
	revDict = {v:k for k,v in labelsDict.items()}
	y = np.asarray(y)
	boundaries = list((y[:-1] != y[1:]).nonzero()[0] + 1) + [y.shape[0]-1]
	labels = [revDict[y[x-1]] for x in boundaries]
	curFrames=0
	assert len(boundaries)==len(labels)
	stm = ''
	for i in xrange(len(labels)):
		stm+=stmNewLine(medianame,labels[i],curFrames*0.05,boundaries[i]*0.05)
		curFrames = boundaries[i]
	f = open(outfilename,'w')
	f.write(stm)
	f.close()

def audacityNewLine(start,end,label):
	return '\t'.join([str(start),str(end),label]) + '\n'

def writeToAudacity(y,labelsDict,outfilename):
        revDict = {v:k for k,v in labelsDict.items()}
        y = np.asarray(y)
        boundaries = list((y[:-1] != y[1:]).nonzero()[0] + 1) + [y.shape[0]-1]
        labels = [revDict[y[x-1]] for x in boundaries]
        curFrames=0
        assert len(boundaries)==len(labels)
        txt = ''
        for i in xrange(len(labels)):
                txt+=audacityNewLine(curFrames*0.05,boundaries[i]*0.05,labels[i])
                curFrames = boundaries[i]
        f = open(outfilename,'w')
        f.write(txt)
        f.close()
