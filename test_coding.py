

import numpy as np
from numpy import linalg as LA
from numpy import *
import scipy.linalg




import os

def mmWaverate_calculation(random_state=7, n_bs=4):
    np.random.seed(random_state)
    # calculate the noise power
    bandwidth = 1e6 # Hz
    boltzmann = float(1.381e-23) #constant
    kelvin = 290
    noise_figures = power(10,float_(0.7)) # 7 dB
    noise_power = float(bandwidth*boltzmann*kelvin*noise_figures)  # 1.2665e-13
    noise_power = 1 # normalize

    rate_ = ones((n_bs, 1), dtype=float)
    sinr_0 = 100.0 * np.random.rand(n_bs, 1)
    inr_0 = 1.0 * np.random.rand(n_bs, 1)
    range_x = np.arange(n_bs)
    for bs in range_x:
        rate_[bs, 0] = float(log2 (1 + (sinr_0[bs, 0])//(inr_0[bs, 0] + noise_power)))
    return rate_


xxx = mmWaverate_calculation()
print(xxx)
# # Generate the small fading channel based on the channel model in equation (2)
# def channel_generation(n_tx=64, n_rx= 8, n_rf_chain = 2, n_bs = 4, n_ue = 4):
#     channel_ = zeros((n_bs, n_ue, n_tx, n_rx), dtype=complex)
#     theta_   = ones((n_tx,n_tx), dtype=complex)
#     z_ = 1/np.math.sqrt(2) * (( np.random.standard_normal((n_tx, n_tx)) + 1j* np.random.standard_normal((n_tx, n_tx)) ))
#     q_, r_ = scipy.linalg.qr(z_)
#     #q_, r_ = np.qr
#     uni_matrix = q_[:, 0:(n_rf_chain-1)] # Semi unitary matrix uni_matrix' * uni_matrix = I
#     n_streams = ones((1, n_rf_chain))  # number of streams/main paths
#     matrix_d = np.diag((n_streams))    #diagonal matrix D  with positive diagonal entries.
#     theta_ = uni_matrix * matrix_d * np.transpose(uni_matrix)
#     #print(theta_.shape)
#     for n in range(n_bs):
#         for m in range(n_bs):
#             channel_[n - 1, m - 1, :, :] = np.matmul(scipy.linalg.sqrtm(theta_) * 1 / (np.math.sqrt(n_tx)) , ((1/n_tx * np.random.standard_normal((n_tx, n_rx)) +1j* 1/n_tx * np.random.standard_normal((n_tx, n_rx)))))
#             #channel_[n - 1, m - 1, :, :] = scipy.linalg.sqrtm(theta_) * 1 / (np.math.sqrt(n_tx)) * ((1/n_tx * np.random.standard_normal((n_tx, n_rx)) +1j* 1/n_tx * np.random.standard_normal((n_tx, n_rx))))
#
#
#     return channel_
#
# debugchannel = channel_generation()
# print(debugchannel)
# print('pass')
