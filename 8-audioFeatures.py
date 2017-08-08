import os
import glob
import sys
import cPickle as pickle


split = int(sys.argv[1])


WAV_PATH = '../data/wav/'
SPLIT_PATH = '../data/splits/20folder/'


folders = pickle.load(open(SPLIT_PATH+str(split)+'.p','r'))
folders = ['../'+x for x in folders]

os.chdir('pyAudioAnalysis')

for f in folders:
        os.system('python audioAnalysis.py featureExtractionDir -i '+f+' -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050')




