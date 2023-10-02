import numpy as np
from numpy.random import uniform, normal,poisson, binomial
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

np.random.seed (5)

n_sample = 100
a = 0.6
b = 0.4
sd = 0.3

x = uniform(1, 5, size= n_sample)
mu = a * x + b

y = normal(mu, sd)
slope, intercept, r_value , p_value , std_err = stats.linregress (x,y)

xvals = np.array ([0,5.5])

yvals = slope * xvals + intercept

plt.scatter (x, y, s=20, alpha=0.8)

plt.plot(xvals , yvals , color='magenta')

plt.show()
