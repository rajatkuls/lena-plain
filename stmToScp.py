import glob
import extractFeatures
import labelTools

nearDict = labelTools.label_map1_near
ext = '*.vad.scp'

stmFiles = glob.glob('../data/VanDam1/*.stm ')

stmFile = '../data/VanDam1/BS80_030213.stm'

f = open(stmFile,'r').read().split('\n')[:-1]

def getStartLine(f,x):
	pos = 0
	for i in f:
		if x<=float(i.split(' ')[4]):
			return pos
		pos+=1


def getEndLine(f,x):
	pos = 0
        for i in f:
                if x<=float(i.split(' ')[4]):
                        return pos
                pos+=1


start = getStartLine(f,0)
if start=-1:
	start=0

end = getEndLine(f,30*60)

assert start<end

medianame = extractFeatures.basename(stmFile)

scp = ''

for i in xrange(start,end):
	t1 = f[i].split(' ')
	if t1[1] in nearDict.keys():
		scp += medianame+'.fea['+str(int(float(t1[3])*100))+','+str(int(float(t1[4])*100))+']\n'



outFile = open(ext.replace('*',medianame),'w')
outFile.write(scp)
outFile.close()







