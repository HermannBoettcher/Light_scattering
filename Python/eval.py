from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

data76 = np.loadtxt('76.dat')

#to_delete = []
#for k in range(len(data76)):
#    #print(data76[k][1])
#    if data76[k][1] == 0:
#        to_delete.append(k)
#    else:
#        continue

data76 = np.transpose(data76)

intensity_time = plt.figure(1, figsize=(5,5))
plt.plot(data76[0], data76[1])
plt.xlabel('Time in seconds')
plt.ylabel('Intensity in ??? ')
plt.grid(True)
plt.tight_layout()
intensity_time.savefig('intensity_time.pdf')
#plt.plot(data76[0], data76[1])

#print(data76)
data76[1] = -1 / 218 * np.log(data76[1]) + 1 / 218 * np.log(0.5)

distance_time = plt.figure(2, figsize=(5,5))
plt.plot(data76[0], data76[1])
plt.xlabel('Time in seconds')
plt.ylabel('Distance in ???')
plt.grid(True)
plt.tight_layout()
distance_time.savefig('distance_time.pdf')

