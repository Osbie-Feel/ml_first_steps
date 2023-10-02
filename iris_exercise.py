import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris

# iris_dataset = load_iris()
# print("typ: ", type(iris_dataset))
# iris_df = pd.DataFrame(data= np.c_[iris_dataset['data'], iris_dataset['target']],
#                      columns= iris_dataset['feature_names'] + ['target'])

# import some data to play with
iris = datasets.load_iris()
num_features = len(iris.data[0])
print("Number of features: {}".format(num_features))

for i in range(num_features):
    for j in range(num_features):
        if i < j:
            # take the two features.
            X = iris.data[:, [i,j]]
            Y = iris.target
            feature_i = iris['feature_names'][i]
            feature_j = iris['feature_names'][j]

            # Create an instance of Logistic Regression Classifier and fit the data.
            logreg = LogisticRegression(C=1e5)
            logreg.fit(X, Y)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            h = 0.02  # step size in the mesh
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure(1, figsize=(4, 3))
            plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, shading='auto')

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors="k", cmap=plt.cm.Paired)
            plt.xlabel(feature_i)
            plt.ylabel(feature_j)

            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())

            plt.show()
            print("Report für die Iris-Merkmale: {0} und {1}".format(feature_i, feature_j))
            print(classification_report(Y, logreg.predict(X)))
