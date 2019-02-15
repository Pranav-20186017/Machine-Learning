import re
import numpy as np
from sklearn.naive_bayes import GaussianNB
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

test_features = [[2567,4],[1200,3],[852,2],[1852,4],[1203,3]]
test_labels = [314000,299000,179900,299900,239500]
clf = GaussianNB()
clf.fit(X, Y)
pred = clf.predict(test_features)
print(pred)
print(accuracy_score(pred,test_labels))
clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
print(clf_pf.predict([[852,2]]))
