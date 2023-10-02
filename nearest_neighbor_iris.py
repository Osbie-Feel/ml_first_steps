import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()
num_features = len(iris.data[0])
print("Number of features: {}".format(num_features))

for i in range(num_features):
    for j in range(num_features):
        if i < j:
            X = iris.data[:, [i,j]]
            y = iris.target

            h = 0.02  # step size in the mesh

            # Create color maps
            cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
            cmap_bold = ["darkorange", "c", "darkblue"]

            # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

            for weights in ["uniform", "distance"]:
                # we create an instance of Neighbours Classifier and fit the data.
                clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
                clf.fit(X, y)
                y_pred = clf.predict(X)
                print(classification_report(y, y_pred))

                cm = confusion_matrix(y, y_pred, labels=clf.classes_)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
                disp.plot()

                # Plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, x_max]x[y_min, y_max].
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                plt.figure(figsize=(8, 6))
                plt.contourf(xx, yy, Z, cmap=cmap_light)

                # Plot also the training points
                sns.scatterplot(
                    x=X[:, 0],
                    y=X[:, 1],
                    hue=iris.target_names[y],
                    palette=cmap_bold,
                    alpha=1.0,
                    edgecolor="black",
                )
                plt.xlim(xx.min(), xx.max())
                plt.ylim(yy.min(), yy.max())
                plt.title(
                    "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
                )
                plt.xlabel(iris.feature_names[i])
                plt.ylabel(iris.feature_names[j])

            plt.show()