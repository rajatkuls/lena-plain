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


'''
# This segment proves that the true stm has some gaps in the segments. I now assume that the segment beginning lines are the markers.
t1 = open(true_stm).read().split('\n')[:-1]
end1 = t1[0].split(' ')[4]

for i in xrange(len(t1)-1):
	st = t1[i+1].split(' ')[3]
	if st!=end1:
		print t1[i]
		print i
	end1=t1[i+1].split(' ')[4]
'''










pred_stm = '../data/VanDam1/BS80/BS80_030213.stm_'+smooth+'_'+weight
true_stm = getTrueStm(pred_stm)
fixed_stm = getFixedStm(pred_stm)

true_stm = convertToFrames(true_stm)
# pred_stm = convertToFrames(pred_stm)
fixed_stm = convertToFrames(fixed_stm)

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

os.system('mkdir -p hyp/2.1')
f = open('hyp/2.1/'+smooth+'_'+weight+'.scored','w')
f.write('F1\n')
f.write(str(f1_score(y,y1,average = 'weighted')))
f.write('\nJaccard\n')
f.write(str(jaccard_similarity_score(y,y1)))
f.write('\n')
f.close()







