from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
import numpy as np
iris_dataset = load_iris()
X_train , X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state= 0)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)
KNeighborsClassifier(algorithm='auto',leaf_size = 30, metric = 'minkowski',
    metric_params = None, n_jobs = 1, n_neighbors = 1, p =2, weights = 'uniform')

X_new = np.array([[5, 2.9, 1, 0.2]])
print(X_new.shape)

prediction = knn.predict(X_new)
print(prediction)
print("predicted target name : {}".format(iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

print("Test set score : {:.2f}".format(knn.score(X_test,y_test)))