import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
f = open("data.txt","r")
x = []
y = []
lines = f.readlines()
for line in lines:
	temp = (line.strip("\n")).split(",")
	k = []
	k.append(temp[0])
	k.append(temp[1])
	k = list(map(int, k))
	x.append(k)
	y.append(temp[2])
y = list(map(int, y))
X = np.asarray(x)
Y = np.asarray(y)
