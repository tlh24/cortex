

import os
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import pdb

mypath = '/home/tlh24/Dropbox/cortex/images/'
os.chdir(mypath)

files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# need to sort by number.

def getnum(fname):
	s = re.findall(r'\d+', fname)
	if len(s) == 1:
		return int(s[0])
	else:
		return -1

indx = list(map(getnum, files))
indx = np.argsort(indx)
files_sort = [files[i] for i in indx]

mypath = '/home/tlh24/Dropbox/cortex/'
os.chdir(mypath)
print(os.getcwd())

e = 0
for fil in files_sort:
	s = f'cp %simages/%s %simages_sort/gar%05d.png' % (mypath, fil , mypath,e)
	print(s)
	os.system(s)
	e = e+1

# ffmpeg -framerate 60 -i  gar%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
