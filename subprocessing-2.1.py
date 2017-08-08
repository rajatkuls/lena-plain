import sys
import os
import glob

if len(glob.glob('stopjobs'))==1:
        print 'Exit called'
        exit(0)

NUM = 4.0

for i in [x/NUM for x in xrange(int(NUM)+1)]:
	for j in [x/NUM for x in xrange(int(NUM)+1)]:
		os.system('nohup python 1.1-silence.py '+str(i)+' '+str(j)+' 1>stdout'+str(i)+'.log 2>stderr'+str(i)+'.log &')
		os.system('nohup python 2.1-scoreSilence.py '+str(i)+' '+str(j)+' 1>stdout'+str(i)+'.log 2>stderr'+str(i)+'.log &')


