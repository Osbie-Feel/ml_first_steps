# #############################################################################
# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor

# k-Nearest Neighbor Regression

# load the datasets
diabetes = load_diabetes()
# fit a knreg to the data
knreg = KNeighborsRegressor()
knreg.fit(diabetes.data, diabetes.target)
print(knreg)
# make predictions
expected = diabetes.target
predicted = knreg.predict(diabetes.data)
# summarize the fit of the knreg
mse = np.mean((predicted-expected)**2)
print("Mean Squared Error: ", mse)
print("R2 Score: ", knreg.score(diabetes.data, diabetes.target))

# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(diabetes.data, diabetes.target))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(diabetes.data, diabetes.target))

X,y = load_diabetes(return_X_y=True)

n_neighbors = 5

for i, weights in enumerate(["uniform", "distance"]):
    knn = knreg(n_neighbors, weights=weights)
    y_pred = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color="darkorange", label="data")
    plt.plot(T, y_pred, color="navy", label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.tight_layout()
plt.show()