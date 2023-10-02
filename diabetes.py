import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset as df
diabetes_df = datasets.load_diabetes()
feature_names = diabetes_df.feature_names
print("Diabetes featurenames:\n ", feature_names)
num_features = len(feature_names)
print(num_features)
pdf = PdfPages("diabetes.pdf")
fig = plt.figure()

# iterate through all features
for i in range(num_features):
    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, i]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

    # Plot outputs
    # sub = fig.add_subplot(2, 2, (i % 4) + 1)
    sub = fig.add_subplot(111)

    sub.scatter(diabetes_X_test, diabetes_y_test, color="black")
    sub.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
    sub.title.set_text(diabetes_df.feature_names[i])

    pdf.savefig(fig)

pdf.close()

diabetes_Xf, diabetes_yf = datasets.load_diabetes(return_X_y=True,
                                                    as_frame=True)
print('diabetes_X\n', diabetes_Xf)
print('\ndiabetes_y\n', diabetes_yf)
