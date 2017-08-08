import os
import sys

split = int(sys.argv[1])

os.system('python segmentation.py ' + str(split))
os.system('python tools.py ' + str(split))



