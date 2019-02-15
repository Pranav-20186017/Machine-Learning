from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

iris_dataset = load_iris()
X_train , X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state= 0)


iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)

grr = pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize = (15, 15), marker = 'o', hist_kwds = {'bins':20}, s= 60, alpha=0.8,cmap = mglearn.cm3)

plt.savefig('graph.png')

#Version from stack overflow
# import pandas as pd
# import matplotlib.pyplot as plt
# # %matplotlib inline

# from sklearn import datasets

# iris_dataset = datasets.load_iris()
# X = iris_dataset.data
# Y = iris_dataset.target

# iris_dataframe = pd.DataFrame(X, columns=iris_dataset.feature_names)

# # Create a scatter matrix from the dataframe, color by y_train
# grr = pd.plotting.scatter_matrix(iris_dataframe, c=Y, figsize=(15, 15), marker='o',
#                                  hist_kwds={'bins': 20}, s=60, alpha=.8)

# plt.savefig('foo.png')