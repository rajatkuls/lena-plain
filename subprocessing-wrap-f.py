import os
import sys

for i in xrange(8):
	os.system('nohup python wrap-f-subpr.py '+str(i*20)+' '+str((i+1)*20)+' 1>stdout'+str(i)+'.log 2>stderr'+str(i)+'.log &')



