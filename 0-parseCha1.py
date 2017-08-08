import glob
# This contains reading from .cha file and writing to stm

STM_PATH = '../data/VanDam1/'

def getName(chaFile):
	x = chaFile[chaFile.rfind('/')+1:chaFile.rfind('.')]
	return STM_PATH + x[:x.find('_')]  +'/'+ x+ '.stm'

def stmNewLine(medianame,code,time):
        s = ' '.join([medianame,code,medianame+'_'+code]+[x[:-3]+'.'+x[-3:] for x in time.split('_')])
        # TODO Add more info to s, take more inputs to this function
        return s + '\n'

def stmNewMultiline(medianame,code,line):
	stm = ''
	t1 = line.split('\x15')
	for i in xrange(len(t1)):
		if i%2==1:
			stm += stmNewLine(medianame,code,t1[i])
	return stm

def makeFixedStm(chaFile):
	f = open(chaFile,'r').read().split('@Bg')
	header = f[0].split('@')[1:]
	# Consume header
	if 'UTF8' not in header[0]:
		raise Exception("Format error")
	header = header[1:]
	if 'Begin' not in header[0]:
		raise Exception("Format error")
	header = header[1:]
	if 'Languages' not in header[0]:
		raise Exception("Format error")
	header = header[1:]
	if 'Participants' not in header[0]:
		raise Exception("Format error")
	participants = header[0]
	header = header[1:]
	if 'Options' not in header[0]:
		raise Exception("Format error")
	header = header[1:]
	# ID Format: @ID: language|corpus|code|age|sex|group|SES|role|education|custom|
	# 			0   1	  2	3   4	5    6	  7	8	9
	IDs = {}
	while True:
		if 'ID' not in header[0]:
			break
		info = header[0].split('|')
		IDs[info[2]] = info[9]	# here the mapping is from say MAN code to Male_Adult_Near custom
		header = header[1:]
	for k in IDs.keys():
		if k not in IDs:
			raise Exception("IDs and participants do not match")
	medianame = chaFile[:chaFile.find('.')] 
	for i in header:
		if 'Media' in i:
			medianame = i.split()[1]
			if medianame[-1]==',':
				medianame = medianame[:-1]
	stm = ''
	for i in f[1:]:
		t1 = i.split('*')
		if '\x15' in t1[0]:
			print i
			raise Exception("There's a time marker in the opening of this segment, confirm splitting")
		for line in t1[1:]:
			if line[3]!=':' or not line[:3].isupper():
				raise Exception("Line does not start with a 3 digit code like TVN:")
			code = line[:3]
			if line.count('\x15')==2:
				time = line[line.find('\x15')+1:line.find('\x15',line.find('\x15')+1)]
				stm+= stmNewLine(medianame,code,time)
			elif line.count('\x15') > 2:
				stm+= stmNewMultiline(medianame,code,line)
			else:
				print line
				raise Exception("Is there no audio in this segment?")
	stm = stm.replace(' .0 ',' 0.000 ')
	outfile = open(getName(chaFile),'w')
	print getName(chaFile)
	outfile.write(stm)
	outfile.close()

for chaFile in glob.glob('../VanDam1/*/*.cha'):
	makeFixedStm(chaFile)



