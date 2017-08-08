import glob

# Quick tool to check for gaps in stm

stm = glob.glob('../data/stm/*') + glob.glob('../VanDam1/*/*fixed*')

for s in xrange(len(stm)):
	print str(s) + ' ',
	f = open(stm[s],'r').read().split('\n')[:-1]
	for i in xrange(1,len(f)):
		curr = f[i]
		prev = f[i-1]
		if curr.split(' ')[3]!=prev.split(' ')[4]:
			print stm[s]+" is not correct"
















