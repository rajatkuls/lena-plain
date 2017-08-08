import os

for i in xrange(30):
	os.system('nohup python batch.py ' + str(i) + ' &')


