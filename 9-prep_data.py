# prepare data for each fold and dump to a pickle. 
# for each fold, get X , y where y has a consistent notation.
# define notation of y here

import glob
import os
import numpy as np
import cPickle as pickle
import sys



label_map = {
    "ADU": "OAD", # adult
    "ALL": "OCH", # all_together child
    "ATT": "OAD", # toll booth attendant
    "AU2": "OAD", # audience2 adult
    "AU3": "OAD", # audience3 adult
    "AUD": "OAD", # audience1 adult
    "AUN": "OAD", # aunt
    "BAB": "OCH", # sibling
    "BEN": "OCH", # Ben child
    "BRO": "OCH", # brother
    "CHI": "CHI", # target child
    "ELE": "OAD", # unidentified adult
    "EVA": "MOT", # Eva mother
    "FAT": "FAT", # father
    "GMA": "OAD", # grandmother
    "GPA": "OAD", # grandfather
    "GRA": "OAD", # grandmother
    "GRF": "OAD", # grandfather
    "GRM": "OAD", # grandmother
    "JAN": "OAD", # Janos visitor
    "JIM": "OAD", # Jim grandfather
    "JOE": "OCH", # Joey child
    "KID": "OCH", # child
    "KUR": "OAD", # Kurt adult
    "LOI": "OAD", # Louise aunt
    "MAD": "OCH", # Madeleine child
    "MAG": "CHI", # Maggie target child
    "MAN": "OAD", # man
    "MAR": "OCH", # Mark brother
    "MOT": "MOT", # mother
    "NEI": "OAD", # neighbor adult
    "OTH": "OAD", # other adult
    "PAR": "OAD", # participant adult
    "PAR1": "OAD", # participant adult
    "PAR2": "OAD", # participant adult/child *
    "PAR3": "OAD", # participant adult
    "PAR4": "OAD", # participant adult
    "PER": "OAD", # person
    "ROS": "CHI", # Ross target child
    "SI1": "OCH", # sibling
    "SI2": "OCH", # sibling
    "SI3": "OCH", # sibling
    "SIB": "OCH", # sibling
    "SP01": "OAD", # adult
    "SPO1": "OAD", # female adult
    "TEA": "OAD", # adult
    "TEL": "OTH", # media
    "TOY": "OTH", # toy
    "UN1": "OCH", # adult/child *
    "UN2": "OCH", # child
    "UN3": "OAD", # adult/child *
    "UN4": "OCH", # child
    "UNK": "OAD", # uncle
    "VAC": "OAD", # unidentified person
    "VIS": "OAD", # visitor
    "WOM": "OAD", # woman
    "SIL": "SIL"  # SIL for labelsDict
}

labelsDict = {'SIL':0,'CHI':1,'MOT':2,'FAT':3,'OCH':4,'OAD':5,'OTH':6}

# sys.argv[1] is fold no.
# t1 = np.load('../data/wav/AR31_021109a/1.wav_st.npy')

def getStatsFromArray(x):
        #f = np.hstack((np.mean(x,axis=0),np.std(x,axis=0),np.max(x,axis=0),np.min(x,axis=0))).reshape(4*x.shape[1],1).T
        f = np.hstack((np.mean(x,axis=1),np.std(x,axis=1))).reshape(2*x.shape[0],1).T
        f[np.isnan(f)] = 0
        return f

def getStatsFromWav(wav):
	try:
		x = np.load(wav+'_st.npy')
	except:
		return None
	return getStatsFromArray(x)

def getLabelFromWav(wav):
	try:
		label = labelsDict[label_map[glob.glob(wav[:-4]+'.label*')[0][-3:]]]
	except:
		label = labelsDict['OTH']
	return label
	


FOLDS_PATH = '../data/folds/'
DATA_PATH  = '../data/wav/'
FEAT_DIM = 34

portions = glob.glob(FOLDS_PATH+'portion*')

portion = portions[int(sys.argv[1])]
# for portion in portions:
# portion = portions[0]

t1 = open(portion,'r').read().split('\n')[:-1]
portionFolders = [DATA_PATH+x[:x.find('.')] for x in t1]
wavs = []
for folder in portionFolders:
	wavs.extend(glob.glob(folder+'/*.wav'))



X = np.array([], dtype=np.int64).reshape(0,FEAT_DIM*2)
y = np.array([],dtype=np.int64)

for wav in wavs:
	t1 = getStatsFromWav(wav)
	if t1 is not None:
		X = np.vstack((X,getStatsFromWav(wav)))
		y = np.hstack((y,getLabelFromWav(wav)))

pickle.dump(X,open('../data/trainData/'+str(sys.argv[1])+'X.p','wb'))
pickle.dump(y,open('../data/trainData/'+str(sys.argv[1])+'y.p','wb'))
	












