import os
import glob
from sklearn.metrics import f1_score,jaccard_similarity_score
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
import cPickle as pickle
import sys

# Calculate scores from stm file, given inputs as smooth and weight go compare with the original and print scores
# Per frame

FRAME_WIDTH = 0.001
INV_FRAME_WIDTH = 1000
MIN_DUR = 0.0001

def getTrueStm(pred_stm):
	searchString = '.stm'
	return pred_stm[:pred_stm.find(searchString)+len(searchString)]

def getFixedStm(pred_stm):
	return getTrueStm(pred_stm).replace('.stm','.fixed.stm')

stmFile = '../VanDam/stm/AR31_021109a.stm'

def convertToFrames(stmFile):
	stm = open(stmFile,'r').read().split('\n')[:-1]
	seg = []
	for line in stm:        
		start = float(line.split(' ')[3])       
		end = float(line.split(' ')[4])
		t1 = [(start+i*FRAME_WIDTH,start+(i+1)*FRAME_WIDTH,line.split(' ')[1]) for i in xrange(int((end-start+MIN_DUR)*INV_FRAME_WIDTH))]
		if str(start)!=str(t1[0][0]) or str(end)!=str(t1[-1][1]):
			print start,end
			print t1[0][0],t1[-1][1]
			print
		seg.extend(t1)
	return seg

# This seg now contains it in frame_width granularity. I added MIN_DUR to cover rounding errors from below this frame_width

def labelToInt(label):
	# LabelsMap
	if label == 'SIL':
		return 0
	return 1

def convertStmToVec(stm):
	getcontext().prec = 10
	y = []
	for row in stm:
		reps = Decimal(row.split()[4]) - Decimal(row.split()[3])
		reps = int(reps/Decimal(frame_width))
		lab = labelToInt(row.split()[1])
		for i in xrange(reps):
			y.append(lab)
	return y
