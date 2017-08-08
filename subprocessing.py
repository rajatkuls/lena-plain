import sys
import os
import glob

if len(glob.glob('stopjobs'))==1:
        print 'Exit called'
        exit(0)
CORES = 30
num = 159/CORES

for i in xrange(CORES):
	a = i*num
	b = (i+1)*num
	os.system('nohup python tools.py '+str(a)+' '+str(b)+' 1>stdout'+str(i)+'.log 2>stderr'+str(i)+'.log &')

a = b
b = 159

os.system('nohup python tools.py '+str(a)+' '+str(b)+' 1>stdout'+str(i)+'.log 2>stderr'+str(i)+'.log &')




