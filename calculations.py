# I want to calculate the cdf of multivariate normal with t variables, where m = [0,0,....,0],
# Sigma = [[1,...,1], [1,2,2,...,2],...,[1,2,3,...,t]], and the evaluations are: [0.5, 1, 1.5, ...., t/2] and
# [-0.5, -1, -1.5, ...., -t/2]

import numpy as np
from scipy import stats
x_1 = [1/2, 1, 1.5, 2, 2.5]
x_2 = [1/2, 1, 1.5, 1000, 2.5]
x_3 = [1/2, 1, 1.5, 2, 1000]
mean = [0, 0, 0, 0, 0]
cov = [[1, 1, 1, 1, 1], [1, 2, 2, 2, 2], [1, 2, 3, 3, 3], [1, 2, 3, 4, 4], [1, 2, 3, 4, 5]]


# x_1 = [1/2, 1, 1.5, 2]
# x_2 = [1/2, 1, 1000, 2]
# x_3 = [1/2, 1, 1.5, 1000]
# mean = [0, 0, 0, 0]
# cov = [[1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 3, 3], [1, 2, 3, 4]]

# x_1 = [1/2, 1, 1.5]
# x_2 = [1/2, 1000, 1.5]
# x_3 = [1/2, 1, 1000]
# mean = [0, 0, 0]
# cov = [[1, 1, 1], [1, 2, 2], [1, 2, 3]]

# x_1 = [1/2, 1]
# x_2 = [1000, 1]
# x_3 = [1/2, 1000]
# mean = [0, 0]
# cov = [[1, 1], [1, 2]]


y_cdf_1 = stats.multivariate_normal.cdf(x_1, mean, cov)
y_cdf_2 = stats.multivariate_normal.cdf(x_2, mean, cov)
y_cdf_3 = stats.multivariate_normal.cdf(x_3, mean, cov)

print(y_cdf_2)
print(y_cdf_3)
print(y_cdf_2-y_cdf_1)
print(y_cdf_3-y_cdf_1)
# print(cov)
