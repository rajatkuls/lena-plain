import glob
import os
import sys
import cPickle as pickle

#a = int(sys.argv[1])
#b = int(sys.argv[2])

split = int(sys.argv[1])

VD_PATH = '../VanDam/'
DATA_PATH = '../data/'
OPENSMILE_PATH = '/data/ASR5/rkulshre/tools/opensmile-2.3.0/'


STM = DATA_PATH + 'stm/'
WAV_PATH = DATA_PATH + 'wav/'

#stmFiles = glob.glob(STM+'*')
#stmFiles.sort()

def getFilename(stmFile):
	return stmFile[stmFile.rfind('/')+1:stmFile.rfind('.')]

def makeDataPerRow(stmFile,wavFile,label,onset,offset,pos):
	filename = getFilename(stmFile)
	folder = WAV_PATH+filename
	outFileRoot = folder + '/' + str(pos)
	outFileWav = outFileRoot + '.wav'
	os.system('mkdir -p '+folder)
	os.system('ffmpeg -y -i ' + wavFile + ' -ss ' + onset + ' -to ' + offset + ' ' + outFileWav)
	os.system(OPENSMILE_PATH+'SMILExtract -C MFCC12_0_D_A.conf -I ' + \
					outFileWav + ' -O ' + outFileRoot+'.mfcc')
	os.system('touch ' + outFileRoot + '.label.' + label)

def makeDataPerFile(stmFile,wavFile):
	f = open(stmFile,'r').read().split('\n')[:-1]
	for pos,row in enumerate(f):
		t1 = row.split()
		label = t1[1]
		onset = t1[3]
		offset = t1[4]
		#create wav, mfcc, label
		makeDataPerRow(stmFile,wavFile,label,onset,offset,pos)
		
def makeDataPerBatch(stmFiles,a,b):
	for i in stmFiles[a:b]:
		makeDataPerFile(i)

def makeDataPerAll(stmFiles):
	makeDataPerBatch(stmFiles,0,len(stmFiles))

def makeDataPerSplit(split):
	t1 = pickle.load(open(DATA_PATH+'/splits/wav/'+str(split)+'.p'))
	for wavFile in t1:
		stmFile = wavFile.replace('wav','stm')
	        stmFile = stmFile.replace('VanDam','data')
		makeDataPerFile(stmFile,wavFile)


makeDataPerSplit(split)




