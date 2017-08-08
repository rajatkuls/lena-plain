import glob
import labelTools
import os

OPENSMILE_PATH = '/data/ASR5/rkulshre/tools/opensmile-2.3.0/'

WAV_PATH = '../data/VanDam/'
WAV1_PATH = '../data/VanDam1/'  # I can't write wavs to original location
DATA_PATH = '../data/features/plp/'



wavFiles = glob.glob(WAV_PATH+'*wav')
wavFiles1 = glob.glob(WAV1_PATH+'*/*wav')

os.system('mkdir -p '+DATA_PATH)


def basename(filename):
	return filename[filename.rfind('/')+1:filename.rfind('.')]


for wav in wavFiles+wavFiles1:
	print wav
	csvFile = DATA_PATH+basename(wav)+'.csv'
	os.system(OPENSMILE_PATH+'SMILExtract -C PLP_0_D_A_Z.conf -I ' + wav + ' -O ' + csvFile)






















