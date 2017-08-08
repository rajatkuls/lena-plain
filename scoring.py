import os
import glob
from sklearn.metrics import f1_score,jaccard_similarity_score
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
import cPickle as pickle


# Calculate scores from stm file

y_true = []
y_pred = []

# Per frame

frame_width = 0.001

#true_stm = '../VanDam/stm/AR31_021109a.stm'
#pred_stm = '../data/stm/AR31_021109a.stm'

true_stm = '../data/sil/BS80_030213.stm'
pred_stm = '../data/sil/BS80_030213.mp3newStm'

true_stm = open(true_stm,'r').read().split('\n')[:-1]
pred_stm = open(pred_stm,'r').read().split('\n')[:-1]

assert float(true_stm[-1].split()[4]) == float(pred_stm[-1].split()[4]) , "STM file durations do not match"

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

y = convertStmToVec(true_stm)
y1 = convertStmToVec(pred_stm)
#assert len(y)==len(y1), "STM frames computed do not match"
y1 = y1[:len(y)]


print f1_score(y,y1,average = 'weighted')
print jaccard_similarity_score(y,y1)







