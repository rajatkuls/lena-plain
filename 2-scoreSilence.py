import os
import glob
from sklearn.metrics import f1_score,jaccard_similarity_score
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
import cPickle as pickle
import sys


# Calculate scores from stm file, given inputs as smooth and weight go compare with the original and print scores

smooth = str(sys.argv[1])
weight = str(sys.argv[2])

y_true = []
y_pred = []

# Per frame

frame_width = 0.001

def getTrueStm(pred_stm):
	searchString = '.stm'
	return pred_stm[:pred_stm.find(searchString)+len(searchString)]

pred_stm = '../data/VanDam1/BS80/BS80_030213.stm_'+smooth+'_'+weight
true_stm = getTrueStm(pred_stm)

true_stm = open(true_stm,'r').read().split('\n')[:-1]
pred_stm = open(pred_stm,'r').read().split('\n')[:-1]

if float(true_stm[-1].split()[4]) != float(pred_stm[-1].split()[4]):
	t1 = pred_stm[-1].split()
        t1[4] = true_stm[-1].split()[4]
        pred_stm[-1] = ' '.join(t1)


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

os.system('mkdir -p hyp/2')
f = open('hyp/2/'+smooth+'_'+weight+'.scored','w')
f.write('F1\n')
f.write(str(f1_score(y,y1,average = 'weighted')))
f.write('\nJaccard\n')
f.write(str(jaccard_similarity_score(y,y1)))
f.close()







