import glob
import os
import numpy as np
import labelTools

files = glob.glob('VanDam1/*/*stm')

human = labelTools.label_map


total_turns = []
total_durations = []
total_tv = []

for file in files:
	turn = 0
	duration = 0
	tv = 0
	f = open(file,'r').read().split('\n')[:-1]
	for line in f:
		if line.split(' ')[1] in human.keys():
			dur = float(line.split(' ')[4]) - float(line.split(' ')[3])
			duration += dur
			turn+=1
		if line.split(' ')[1] == 'TVN':
			dur = float(line.split(' ')[4]) - float(line.split(' ')[3])
			tv+=dur
	total_tv.append(tv)
	total_turns.append(turn)
	total_durations.append(duration)











