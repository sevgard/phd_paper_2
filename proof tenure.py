# I want to calculate the cdf of multivariate normal with t variables, where m = [0,0,....,0],
# Sigma = [[1,...,1], [1,2,2,...,2],...,[1,2,3,...,t]], and the evaluations are: [0.5, 1, 1.5, ...., t/2] and
# [-0.5, -1, -1.5, ...., -t/2]

import numpy as np
from scipy import stats

y_positive = []
# y_negative = []
for n in range(20):
    t = n+1
    x_positive = np.linspace(1/2, t/2, t)
    # x_negative = np.linspace(-1/2, -t/2, t)

    mean = []
    for i in range(0, t):
        mean.append(0)

    cov = []
    for i in range(0, t):
        line_i = []
        for j in range(0, t):
            line_i.append(min(i+1, j+1))
        # print(line_i)
        cov.append(line_i)


    y_cdf_positive = stats.multivariate_normal.cdf(x_positive, mean, cov)
    # y_cdf_negative = stats.multivariate_normal.cdf(x_negative, mean, cov)
    print(y_cdf_positive)
    y_positive.append(y_cdf_positive)
    # y_negative.append(y_cdf_negative)
# print(x_positive)
# print(x_negative)
# print(mean)
# print(cov)

print('y_positive: ', y_positive)
# print('y_negative: ', y_negative)