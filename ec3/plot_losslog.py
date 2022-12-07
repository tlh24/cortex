import numpy as np
import csv
import pdb
import matplotlib.pyplot as plt
 
 
with open("losslog.txt", 'r') as x:
    data = list(csv.reader(x, delimiter="\t"))
 
data = np.array(data)
data = data.astype(float)
plt.plot(data[:,0], data[:, 1], 'b')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()
