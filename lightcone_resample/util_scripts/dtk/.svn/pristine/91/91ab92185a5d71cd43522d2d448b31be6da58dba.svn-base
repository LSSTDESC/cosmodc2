

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("array_data.txt")
ps   = np.loadtxt("ps_data.txt")
data2 = np.loadtxt("array_data2.txt")
corr   = np.loadtxt("corr_data.txt")

plt.figure();
plt.title("data slice");
plt.plot(data[:,0],data[:,1]);



plt.figure();
plt.title("PS result");
plt.plot(ps[:,1],ps[:,2]);
plt.plot(ps[:,1],ps[:,2],'bx')

plt.figure();
plt.title("Corr Result");
plt.plot(corr[:,1],corr[:,2])
plt.plot(corr[:,1],corr[:,2],'bx')
plt.yscale('log')
plt.show();

