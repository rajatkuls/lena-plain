import sys
import os
import glob

if len(glob.glob('stopjobs'))==1:
        print 'Exit called'
        exit(0)

NUM = 30

for i in xrange(NUM):
	os.system('nohup python 4-segmentation.py '+str(i)+' 1>stdout'+str(i)+'.log 2>stderr'+str(i)+'.log &')


