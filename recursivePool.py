import glob

ratio = 0.4

def poolFiles(l):
	if len(list)==1:
		array = np.loadtxt(l[0],delimiter=';')
		np.random.shuffle(array)
		select_size = int(array.shape[0] * ratio)
		feat_dim = array.shape[1]






