# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.special import loggamma
from scipy.stats import beta
import matplotlib.pyplot as plt

# 定义对数 Beta 分布的概率密度函数
# def log_beta_pdf(theta, a, b):
#     """Compute the log of the Beta PDF using loggamma for stability."""
    
#     # Compute the log of the Beta PDF
#     log_pdf = (a - 1) * np.log(theta) + (b - 1) * np.log(1 - theta)
#     log_pdf -= (loggamma(a) + loggamma(b) - loggamma(a + b))
#     return log_pdf

# # 样本数据
# samples = np.array([1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1])
# a = 3
# b = 5
# T = np.sum(samples)
# H = len(samples) - T

# # 更新后的 Beta 分布参数
# a_post = a + T
# b_post = b + H

# # 定义 theta 网格
# theta = np.linspace(1e-5, 1 - 1e-5, 1000)

# # 计算对数后验分布
# log_posterior = log_beta_pdf(theta, a_post, b_post)
# posterior = np.exp(log_posterior)  # 转换为概率密度

# # 使用 beta.pdf 计算参考值
# beta_pdf_values = beta.pdf(theta, a_post, b_post)

# # 绘制后验分布和 beta.pdf 的对比
# plt.plot(theta, posterior, label='Computed Posterior', color='orange')
# #plt.plot(theta, beta_pdf_values, label='Beta PDF (Reference)', linestyle='--', color='blue')
# plt.xlabel('θ')
# plt.ylabel('Density')
# plt.legend()
# plt.show()
import numpy as np
from scipy.special import loggamma
from scipy.stats import beta
import matplotlib.pyplot as plt

def compute_log_prior(theta, a, b):
    """Compute log p(theta | a, b) using the Beta distribution."""
    epsilon = 1e-10  # Small value to prevent log(0)
    theta = np.clip(theta, epsilon, 1 - epsilon)  # Clip to avoid log(0)
    beta_constant = loggamma(a) + loggamma(b) - loggamma(a + b)
    log_prior = (a - 1) * np.log(theta) + (b - 1) * np.log(1 - theta) - beta_constant
    return log_prior

def compute_log_likelihood(theta, samples):
    """Compute log p(D | theta) for the given values of theta."""
    num_tails = np.sum(samples)
    num_heads = len(samples) - num_tails
    epsilon = 1e-10  # Small value to prevent log(0)
    theta = np.clip(theta, epsilon, 1 - epsilon)  # Clip to avoid log(0)
    log_likelihood = num_tails * np.log(theta) + num_heads * np.log(1 - theta)
    return log_likelihood

def compute_log_posterior(theta, samples, a, b):
    """Compute the unnormalized log-posterior distribution."""
    T = np.sum(samples)
    H = len(samples) - T
    epsilon = 1e-10
    theta = np.clip(theta, epsilon, 1 - epsilon)  # Clip to avoid log(0)
    
    # Compute the unnormalized log-posterior
    log_posterior_unnorm = compute_log_likelihood(theta, samples) + compute_log_prior(theta, a, b)
    return log_posterior_unnorm  # Return the unnormalized log-posterior

# Sample data
samples = np.array([1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1])
a = 3
b = 5
T = np.sum(samples)
H = len(samples) - T
theta = np.linspace(1e-5, 1 - 1e-5, 1000)

# Compute the unnormalized log-posterior
log_posterior = compute_log_posterior(theta, samples, a, b)
posterior = np.exp(log_posterior)  # Convert to probability density

# Normalize the posterior so that it sums to 1 over the range
posterior /= np.trapz(posterior, theta)  # Use numerical integration to normalize

# Compute the reference Beta PDF
beta_pdf_values = beta.pdf(theta, a + T, b + H)

# Plot the posterior distribution and the reference Beta PDF
plt.plot(theta, posterior, label='Computed Posterior', color='orange')
#plt.plot(theta, beta_pdf_values, label='Beta PDF (Reference)', linestyle='--', color='blue')
plt.xlabel('θ')
plt.ylabel('Density')
plt.legend()
plt.show()

