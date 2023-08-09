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
theta_h = 5
prob_theta = 0.5

# Firm-specific match
mu_l = -1
mu_h = 1
prob_mu = 0.5

# Noise for output
mean_e = 0
sigma_e = 1


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
def y(theta, mu, epsilon):
    return theta + mu + epsilon


# Probability function for theta = theta_h
def p_i_t(p_i_t1, q_i_t1, z_i_f_t):
    return p_i_t1 / (p_i_t1 + (1 - p_i_t1) *
                     (pdf(z_i_f_t - theta_l - 1, mean_e, sigma_e) * q_i_t1
                      + pdf(z_i_f_t - theta_l + 1, mean_e, sigma_e) * (1 - q_i_t1)) /
                     (pdf(z_i_f_t - theta_h - 1, mean_e, sigma_e) * q_i_t1
                      + pdf(z_i_f_t - theta_h + 1, mean_e, sigma_e) * (1 - q_i_t1))
                     )


# Probability function for mu = 1
def q_i_t(p_i_t1, q_i_t1, z_i_f_t):
    return q_i_t1 / (q_i_t1 + (1 - q_i_t1) *
                     (pdf(z_i_f_t - theta_h + 1, mean_e, sigma_e) * p_i_t1
                      + pdf(z_i_f_t - theta_l + 1, mean_e, sigma_e) * (1 - p_i_t1)) /
                     (pdf(z_i_f_t - theta_h - 1, mean_e, sigma_e) * p_i_t1
                      + pdf(z_i_f_t - theta_l - 1, mean_e, sigma_e) * (1 - p_i_t1))
                     )

'Wages'


# Ey_i_o = theta_h * prob_theta + (1 - prob_theta) * theta_l + mu_h * prob_mu + (1-prob_mu) * mu_l
# expected output in period t
def ey_i_t(p_i_t1, q_i_t1):
    return theta_h * p_i_t1 + (1 - p_i_t1) * theta_l + mu_h * q_i_t1 + (1-q_i_t1) * mu_l


# wage paid in period t+1:
# w_i_0 = theta_h * prob_theta + (1 - prob_theta) * theta_l + mu_h * prob_mu + (1-prob_mu) * mu_l
def w_i_t(p_i_t1):
    return theta_h * p_i_t1 + (1 - p_i_t1) * theta_l + mu_h * prob_mu + (1-prob_mu) * mu_l


# expected profit in period t+1
def epi_i_t(p_i_t1, q_i_t1):
    return ey_i_t(p_i_t1, q_i_t1) - w_i_t(p_i_t1)



data = {
    'id': [],
    'time': [],
    'theta_i': [],
    'mu_i': [],
    'epsilon': [],
    'z_i_f_t': [],
    'y_i_t': [],
    'E[y_i_t]': [],
    'wage_i_t': [],
    'p_i_t': [],
    'p_i_t1': [],
    'q_i_t': [],
    'q_i_t1': [],
    'Profit_i_t': [],
    'E[Profit_i_t]': [],
    'empl_status': []
}
df = pd.DataFrame(data)

for n in range(500):
    # 0. Define the list for all varialbes we need:
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

    # Iterate over every period i for each worker n:
    prob_i_t = prob_theta
    qrob_i_t = prob_mu
    empl_status = "hired"
    for i in range(T):
        prob_i_t1 = prob_i_t
        qrob_i_t1 = qrob_i_t
        empl_status1 = empl_status
        z_i_f_t = theta_i + mu_i + E[i]  # output signal
        Z_i_f_T.append(z_i_f_t)
        y_i_f_t = theta_i + mu_i + E[i]  # realized output
        Y_i_T.append(y_i_f_t)
        prob_i_t = p_i_t(prob_i_t1, qrob_i_t1, z_i_f_t)  # probability of theta_i = theta_h
        P_i_T.append(prob_i_t)
        qrob_i_t = q_i_t(prob_i_t1, qrob_i_t1, z_i_f_t)  # probability of mu_i = 1
        Q_i_T.append(qrob_i_t)
        wage_i_t = w_i_t(prob_i_t1)  # wage in period t
        W_i_T.append(wage_i_t)
        exp_y_i_t = ey_i_t(prob_i_t1, qrob_i_t1)  # expected output in period t
        EY_i_T.append(exp_y_i_t)
        profit_i_f_t = y_i_f_t - wage_i_t  # profit in period t
        PI_i_T.append(profit_i_f_t)
        eprofit_i_f_t = exp_y_i_t - wage_i_t  # expected profit in period t
        EPI_i_T.append(eprofit_i_f_t)
        if i == 0:
            empl_status = 'hired'
        else:
            if empl_status1 == 'fired':
                empl_status = 'left'
            elif empl_status1 == 'left':
                empl_status = 'left'
            elif empl_status1 == 'stayed':
                if eprofit_i_f_t >= 0:
                    empl_status = 'stayed'
                else:
                    empl_status = 'fired'
            else:
                if eprofit_i_f_t >= 0:
                    empl_status = 'stayed'
                else:
                    empl_status = 'fired'
        line = [n + 1, i + 1, theta_i, mu_i, E[i], z_i_f_t, y_i_f_t, exp_y_i_t,
                wage_i_t, prob_i_t, prob_i_t1, qrob_i_t, qrob_i_t1, profit_i_f_t, eprofit_i_f_t, empl_status]
        df.loc[len(df)] = line

print(df)
df.to_excel("output_no_ref_one_signal.xlsx")
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