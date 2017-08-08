import sys
import os
import glob

NUM = 5.0

best_i = 0
best_j = 0
best_score = 0.0
to_check = 1 # for F1. 3 for jaccard
for i in [x/NUM for x in xrange(int(NUM)+1)]:
        for j in [x/NUM for x in xrange(int(NUM)+1)]:
		try:
			f = open('hyp/2/'+str(i)+'_'+str(j)+'.scored','r').read().split('\n')
			score = float(f[to_check])
			if score>best_score:
				best_i = i
				best_j = j
				best_score = score
		except:
			pass

print best_i, best_j, best_score







