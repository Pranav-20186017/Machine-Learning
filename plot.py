import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt("data.txt",dtype=np.float64,delimiter=",")
X = data[::,0:2]
Y = data[::,-1:]
# Plotting example dataset
plt.figure(figsize = (15,4),dpi=100)
plt.subplot(121)
plt.scatter(X[::,0:1],Y)
plt.xlabel("Size of house (X1)")
plt.ylabel("Price (Y)")
plt.subplot(122)
plt.scatter(X[::,-1:],Y)
plt.xlabel("Number of Bedrooms (X2)")
plt.ylabel("Price (Y)")
plt.show()