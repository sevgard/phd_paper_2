from scipy.stats import norm
import pandas as pd
from scipy.stats import rv_discrete
# from scipy.integrate import quad, dblquad
import math
# from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt

'''Parameters of the model'''
# Number of periods
T = 20

# General ability
theta_l = 2
theta_h = 4
prob_theta = 0.5

# Firm-specific match
mu_l = -1
mu_h = 1
prob_mu = 0.5

# Noise for general ability
mean_e = 0
sigma_e = 1

# Noise for firm-specific match
mean_n = 0
sigma_n = 1

# Correlation between social contacts

# General ability
cor_theta_ij = 0.5
cov_theta_ij = cor_theta_ij * prob_theta * (1 - prob_theta)

# Firm-specific match
cor_mu_ij = 0.5
cov_mu_ij = cor_mu_ij * prob_mu * (1 - prob_mu)


# PDF of normal distribution
def pdf(x, m, sd,):
    prob_dens_func = (1 / (sd * (2 * math.pi) ** 0.5)) * math.exp(- (x-m) ** 2/ (sd ** 2 * 2))
    return prob_dens_func


# output function:
def y(theta, mu, epsilon, eta):
    return theta + mu + epsilon + eta


# Probability function for theta = theta_h
def p_i_t(p_i_t1, x_i_t):
    return p_i_t1 / (p_i_t1 + (1 - p_i_t1) * math.exp((theta_h - theta_l) / sigma_e ** 2 * (-x_i_t + (theta_h + theta_l) / 2)))


# Probability function for mu = 1
def q_i_t(q_i_t1, z_i_t):
    return q_i_t1 / (q_i_t1 + (1 - q_i_t1) * math.exp((mu_h - mu_l) / sigma_n ** 2 * (-z_i_t + (mu_h + mu_l) / 2)))


'Wages'


# Ey_i_o = theta_h * prob_theta + (1 - prob_theta) * theta_l + mu_h * prob_mu + (1-prob_mu) * mu_l
# expected output in period t
def Ey_i_t(p_i_t, q_i_t):
    return theta_h * p_i_t + (1 - p_i_t) * theta_l + mu_h * q_i_t + (1-q_i_t) * mu_l


# wage paid in period t+1:
# w_i_0 = theta_h * prob_theta + (1 - prob_theta) * theta_l + mu_h * prob_mu + (1-prob_mu) * mu_l
def w_i_t(p_i_t):
    return theta_h * p_i_t + (1 - p_i_t) * theta_l + mu_h * prob_mu + (1-prob_mu) * mu_l


# expected profit in period t+1
def pi_i_t(q_i_t):
    return (mu_h - mu_l) * (q_i_t - prob_mu)



data = {
    'id': [],
    'time': [],
    'theta_i': [],
    'mu_i': [],
    'epsilon': [],
    'eta': [],
    'x_i_t': [],
    'z_i_f_t': [],
    'y_i_t': [],
    'E[y_i_t]': [],
    'wage_i_t': [],
    'p_i_t': [],
    'q_i_t': [],
    'Profit_i_t': [],
    'E[Profit_i_t]': [],
    'empl_status': []
}
df = pd.DataFrame(data)

for n in range(500):
    # 0. Define the list for all varialbes we need:
    X_i_T = [0]         # general ability signal
    Z_i_f_T = [0]       # firm-specific match signal
    P_i_T = [prob_theta]          # probability that theta_i = theta_h
    Q_i_T = [prob_mu]          # probability that mu_i_f = 1
    Y_i_T = [0]         # realized output
    EY_i_T = [0]         # expected output in the period
    W_i_T = [0]          # wage in period
    PI_i_T = [0]         # realized profit in the period
    EPI_i_T = [0]        # expected profit in the period

    # 1. Set the values for the realized general ability theta_i and firm-specific match mu_i_f
    # 1.1. Random variable for the general ability, theta_i
    xk_theta = (theta_l, theta_h)
    pk_theta = (1 - prob_theta, prob_theta)
    theta = rv_discrete(name='theta', values=(xk_theta, pk_theta))
    theta_i = theta.rvs(size=1)[0]

    # 1.2. Random variable for the firm-specific match, mu_i_f
    xk_mu = (mu_l, mu_h)
    pk_mu = (1 - prob_mu, prob_mu)
    mu = rv_discrete(name='mu', values=(xk_mu, pk_mu))
    mu_i = mu.rvs(size=1)[0]
    # print("theta_i: ", theta_i)
    # print("mu_i: ", mu_i)

    # 2. Define the sequence of noise variables epsilon_i_t and eta_i_f_t
    E = np.random.normal(loc=mean_e, scale=sigma_e, size=T)
    N = np.random.normal(loc=mean_n, scale=sigma_n, size=T)

    # Iterate over every period i for each worker n:
    prob_i_t = prob_theta
    qrob_i_t = prob_mu
    empl_status = "stayed"
    for i in range(T):
        prob_i_t1 = prob_i_t
        qrob_i_t1 = qrob_i_t
        empl_status1 = empl_status
        x_i_t = theta_i + E[i]      # general ability signal
        X_i_T.append(x_i_t)
        z_i_f_t = mu_i + N[i]       # firm-specific match signal
        Z_i_f_T.append(z_i_f_t)
        y_i_f_t = y(theta_i, mu_i, E[i], N[i])      # realized output
        Y_i_T.append(y_i_f_t)
        prob_i_t = p_i_t(prob_i_t, x_i_t)       # probability of theta_i = theta_h
        P_i_T.append(prob_i_t)
        qrob_i_t = q_i_t(qrob_i_t, z_i_f_t)     # probability of mu_i = 1
        Q_i_T.append(qrob_i_t)
        wage_i_t = w_i_t(prob_i_t1)              # wage in period t
        W_i_T.append(wage_i_t)
        exp_y_i_t = Ey_i_t(prob_i_t1, qrob_i_t1)  # expected output in period t
        EY_i_T.append(exp_y_i_t)
        profit_i_f_t = y_i_f_t - wage_i_t       # profit in period t
        PI_i_T.append(profit_i_f_t)
        eprofit_i_f_t = exp_y_i_t - wage_i_t    # expected profit in period t
        EPI_i_T.append(eprofit_i_f_t)
        if empl_status1 == 'stayed':
            if eprofit_i_f_t >= 0:
                empl_status = 'stayed'
            else:
                empl_status = 'fired'
        else:
            empl_status = 'left'
        line = [n+1, i+1, theta_i, mu_i, E[i], N[i], x_i_t, z_i_f_t, y_i_f_t, exp_y_i_t,
                wage_i_t, prob_i_t, qrob_i_t, profit_i_f_t, eprofit_i_f_t, empl_status]
        df.loc[len(df)] = line

print(df)
df.to_excel("output_no_ref_two_signals.xlsx")
# print('theta_i ', theta_i, 'mu_i ', mu_i)
# print('X_i_T: ', X_i_T)
# print('Z_i_f_T: ', Z_i_f_T)
# print('Y_i_t: ', Y_i_T)
# print('P_i_T: ', P_i_T)
# print('Q_i_T: ', Q_i_T)
# print('W_i_T: ', W_i_T)
# print('EY_i_T: ', EY_i_T)
# print('PI_i_T: ', PI_i_T)
# print('EPI_i_T: ', EPI_i_T)








# '''Technical functions'''
# # sequence of noise variables
# E = np.random.normal(loc=mean_e, scale=sigma_e, size=T)
# N = np.random.normal(loc=mean_n, scale=sigma_n, size=T)
#
#
# # # visualization
# # print(E)
# # plt.figure(figsize=(7, 5))
# # x = np.linspace(1, T, T)
# # plt.plot(x, E+x*0)
# # plt.plot(x, N+x*0)
# # plt.plot(x, x*0, '--r')
# # plt.show()
#
#
#
# '''Main functions of the model'''
# 'Workers output'
#
#
# # random variable for the general ability
# xk_theta = (theta_l, theta_h)
# pk_theta = (1 - prob_theta, prob_theta)
# theta = rv_discrete(name='theta', values=(xk_theta, pk_theta))
# theta_i = theta.rvs(size=1)[0]
#
# # random variable for the firm-specific match
# xk_mu = (mu_l, mu_h)
# pk_mu = (1 - prob_mu, prob_mu)
# mu = rv_discrete(name='mu', values=(xk_mu, pk_mu))
# mu_i = mu.rvs(size=1)[0]
# # print("theta_i: ", theta_i)
# # print("mu_i: ", mu_i)

# # sequence of outputs
# Y_i_T = [0]
# for i in range(T):
#     Y_i_T.append(y(theta_i, mu_i, E[i], N[i]))
#
# print("Y_i_T: ", Y_i_T)
#
# 'Probabilities'
# # Sequence of signals
# X_i_T = []
# Z_i_f_T = []
# for i in range(T):
#     X_i_T.append(theta_i + E[i])
#     Z_i_f_T.append(mu_i + N[i])
#
# print("X_i_T: ", X_i_T)
# print("Z_i_f_T: ", Z_i_f_T)
#
#
# P_i_T = [prob_theta]
# prob_i_t = prob_theta
# for i in range(T):
#     prob_i_t = p_i_t(prob_i_t, X_i_T[i])
#     P_i_T.append(prob_i_t)
#
# # # print("probability_theta_i = theta_h: ", P_i_T)
#
#
# Q_i_T = [prob_mu]
# qrob_i_t = prob_mu
# for i in range(T):
#     qrob_i_t = q_i_t(qrob_i_t, Z_i_f_T[i])
#     Q_i_T.append(qrob_i_t)
#
# # print("probability_mu_i = 1: ", Q_i_T)
#
#
# W_i_T = []
# EY_i_T = []
# PI_i_T = []
# for i in range(T):
#     wage_i_t = w_i_t(P_i_T[i])
#     exp_y_i_t = Ey_i_t(P_i_T[i], Q_i_T[i])
#     profit_i_t = exp_y_i_t - wage_i_t
#     W_i_T.append(wage_i_t)
#     EY_i_T.append(exp_y_i_t)
#     PI_i_T.append(profit_i_t)
#
# # print("expected_output_i_t: ", EY_i_T)
# # print("wage_i_t: ", W_i_T)
# # print("expected_profit_i_t: ", PI_i_T)
#
#
# # # output sequence visualization
# # plt.figure(figsize=(7, 5))
# # x = np.linspace(0, T, T)
# # # plt.plot(Y_i_T, '-g')
# # plt.plot(P_i_T, '-r', label='P(theta = H)')
# # plt.plot(Q_i_T, '-k', label='P(mu = 1)')
# # plt.plot(x, x*0, '--b')
# # leg1 = plt.legend()
# # plt.show()
#
#
# # # wage and profit sequence visualization
# # plt.figure(figsize=(7, 5))
# # x = np.linspace(0, T, T)
# # # plt.plot(Y_i_T, '-g', label='Y_i_t')
# # plt.plot(EY_i_T, '--g', label='E[Y_i_t]')
# # plt.plot(W_i_T, '-r', label='W_i_t')
# # plt.plot(PI_i_T, '-k', label='Profit_i_t')
# # plt.plot(x, x*0, '--b')
# # leg2 = plt.legend()
# # plt.show()