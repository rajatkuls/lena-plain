import cPickle as pickle
import glob
import numpy as np



ST_FEATUREPATH = '../data/features/st/VanDam/'
ST_FEATUREPATH1 = '../data/features/st/VanDam1/'

FOLDS_PATH = '../data/folds/VanDam/portion*'
FOLDS_PATH1 = '../data/folds/VanDam1/portion*'


WAV_PATH = '../data/VanDam/'
WAV1_PATH = '../data/VanDam1/'  # I can't write wavs to original location
DATA_PATH = '../data/'


def getFilesFromPortion(portion):
        t1 = open(portion,'r').read().split('\n')[:-1]
        return [x[:x.find('.')] for x in t1]

foldFileList  = []
foldFileList1 = []

for portion in glob.glob(FOLDS_PATH):
        foldFileList.append([x for x in getFilesFromPortion(portion)])

for portion in glob.glob(FOLDS_PATH1):
        foldFileList1.append([x for x in getFilesFromPortion(portion)])


folders = glob.glob(ST_FEATUREPATH+'*')


for i in xrange(len(foldFileList)):
	for f in foldFileList[i]:
		filename = ST_FEATUREPATH+str(i)+'/'+f
plpname  = filename.replace('/st/','/plp/')
plpname = plpname[:plpname.rfind('/')-2]+plpname[plpname.rfind('/'):]
plpname = plpname[:plpname.rfind('/')-7]+plpname[plpname.rfind('/'):] + '.csv'
x_temp = np.genfromtxt(plpname)









