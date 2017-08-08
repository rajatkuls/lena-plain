import sys

file = sys.argv[1]
print file

f = open(file,'r').read().split('\n')
f = f[:-1]

for line in f:
    start = float(line.split(' ')[3])
    end = float(line.split(' ')[4])
    if start>end:
            print line





