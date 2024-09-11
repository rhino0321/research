import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math

import numpy as np
from scipy.stats import entropy, zscore, skew, kurtosis
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error, r2_score
from functions import semi_soft_thresholding, wavelet_decomposition, wavelet_reconstruction, approx_entropy
from plot_functions import snr_entropy_rmse, original_data
import tensorflow as tf
from keras import layers, Model
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.backend import random_normal

file_path = 'C:\\Users\\USER\\Desktop\\1220.csv'
data = pd.read_csv(file_path)

param_columns = data.iloc[:, 1:4].astype(float)
third_column = data.iloc[:, 3]
first_nonzero_index = (third_column != 0).idxmax()
print(first_nonzero_index)


temp = param_columns.iloc[first_nonzero_index : data.shape[0] - 1, 0].values
temp = np.nan_to_num(temp)
ph = param_columns.iloc[first_nonzero_index : data.shape[0] - 1, 1].values
ph = np.nan_to_num(ph)
ec = param_columns.iloc[first_nonzero_index : data.shape[0] - 1, 2].values
ec = np.nan_to_num(ec)


N = data.shape[0] - 1 - first_nonzero_index
print(N)
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
noise = np.random.normal(0, 0.005, size=N)
temp_noise = temp + noise
zscore_temp = zscore(temp_noise)
zscore_ph = zscore(ph)
zscore_ec = zscore(ec)
max_decomposition_level = math.floor(math.log2(N))

temp_snr_list = []
temp_rmse_list = []
temp_appro_entropy_list = []
ph_snr_list = []
ph_rmse_list = []
ph_appro_entropy_list = []
ec_snr_list = []
ec_rmse_list = []
ec_appro_entropy_list = []
'''
for decomposition_level in range (1, max_decomposition_level + 1):

    #wavelet decomposition
    temp_coeffs , ph_coeffs, ec_coeffs = wavelet_decomposition(zscore_temp, zscore_ph, zscore_ec, decomposition_level)
    
    #semi soft thresholding
    temp_coeffs, ph_coeffs, ec_coeffs = semi_soft_thresholding(temp_coeffs, ph_coeffs, ec_coeffs)

    #wavelet reconstuction
    temp_rec, ph_rec, ec_rec= wavelet_reconstruction(temp_coeffs, temp, ph_coeffs, ph, ec_coeffs, ec)

    temp_snr_list.append(10 * np.log10(np.sum(np.abs(temp)) / np.sum(np.abs(temp_rec - temp))))
    temp_rmse_list.append(np.sqrt(mean_squared_error(temp, temp_rec)))
    ph_snr_list.append(10 * np.log10(np.sum(np.abs(ph)) / np.sum(np.abs(ph_rec - ph))))
    ph_rmse_list.append(np.sqrt(mean_squared_error(ph, ph_rec)))
    ec_snr_list.append(10 * np.log10(np.sum(np.abs(ec)) / np.sum(np.abs(ec_rec - ec))))
    ec_rmse_list.append(np.sqrt(mean_squared_error(ec, ec_rec)))
    temp_appro_entropy_list.append(approx_entropy(temp_coeffs))
    ph_appro_entropy_list.append(approx_entropy(ph_coeffs))
    ec_appro_entropy_list.append(approx_entropy(ec_coeffs))

#plot original data
#original_data(temp, ph, ec, first_nonzero_index)

#plot snr entropy rmse 
snr_entropy_rmse(max_decomposition_level, ec_snr_list, ec_rmse_list, ec_appro_entropy_list, temp_snr_list, temp_rmse_list, temp_appro_entropy_list, ph_snr_list, ph_rmse_list, ph_appro_entropy_list)

'''
index = range(4800, 5200)
select_ori = temp[4800:5200]
temp_coeffs , ph_coeffs, ec_coeffs = wavelet_decomposition(zscore_temp, zscore_ph, zscore_ec, 1)

temp_coeffs, ph_coeffs, ec_coeffs = semi_soft_thresholding(temp_coeffs, ph_coeffs, ec_coeffs)

temp_rec, ph_rec, ec_rec= wavelet_reconstruction(temp_coeffs, temp, ph_coeffs, ph, ec_coeffs, ec)
scalar = MinMaxScaler()
zscore_temp_rec = zscore(temp_rec)
select_rec = temp_rec[4800:5200]

plt.subplot(2,1,1)
plt.plot(index, select_ori)
plt.xlabel('original data')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(index, select_rec)
plt.xlabel('after wavelet transform')
plt.grid(True)
plt.subplots_adjust(hspace=0.5)
plt.show()

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
#VAE-RESLSTM MODEL
L = 15
m = 4
latent_dim = 15

recent_fluctuation_matrix = np.zeros((N, L, m))

for num in range(L, N):
    bottom = num - L
    for i in range(0, L):
        max = bottom + i
        for j in range(0, m):
            if (max - (m - 1 - j) >= bottom and max - (m - 1 - j) >= 0):
                recent_fluctuation_matrix[num, i, j] = zscore_temp_rec[max - (m - 1 - j)]
input_data = np.zeros((N, L))
target_data = np.zeros(N)
print(len(input_data))
for i in range(L, N):
    input_data[i] = zscore_temp_rec[i - L : i]
    target_data[i] = zscore_temp_rec[i]


test_refluctuation_matrix = recent_fluctuation_matrix[L + int((N - L) * 0.8):]
test_input_data = input_data[L + int((N - L) * 0.8):]
test_target_data = target_data[L + int((N - L) * 0.8):]



def custom_loss(omega1, omega2, B, m):
    def loss(y_true, y_pred):
        reconstuction_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true[0], y_pred[0])) / m / B
        prediction_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true[1], y_pred[1])) / B
        omega1_loss = reconstuction_loss / tf.exp(omega1)
        omega2_loss = (prediction_loss) / tf.exp(omega2)
        total_loss = 0.5 * (omega1_loss + omega2_loss + omega1 + omega2)
        return total_loss
    return loss


def wrapped_custom_loss(omega1, omega2, B, m):
    def wrapped_loss(y_true, y_pred):
        return tf.py_function(custom_loss(omega1, omega2, B, m), (y_true, y_pred), tf.float32)
    return wrapped_loss

custom_objects = {'sampling': sampling}
vae_lstm = tf.keras.models.load_model(
    './model/my_model.h5',
    custom_objects={'loss': wrapped_custom_loss}
)
decoded_data, test_predictions = vae_lstm.predict([test_refluctuation_matrix, test_input_data])
vae_lstm_r2 = r2_score(test_target_data, test_predictions)
vae_lstm_rmse = np.sqrt(mean_squared_error(test_target_data, test_predictions))
DS = []
for i in range (1, len(test_target_data)):
    if ((test_target_data[i] - test_target_data[i - 1]) * (test_predictions[i] - test_predictions[i - 1]) >= 0):
        DS.append(1)
    else:
        DS.append(0)
DS_value = (100 / len(DS)) * sum(DS)
print("DS val : ", DS_value)
index = range(0, 1000)
test_target_data = test_target_data * np.std(temp_rec) + np.mean(temp_rec)
test_predictions = test_predictions * np.std(temp_rec) + np.mean(temp_rec)
select_ori = test_target_data[0:1000]
select_rec = test_predictions[0:1000]

plt.subplot(2,1,1)
plt.plot(index, select_ori)
plt.xlabel('target data')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(index, select_rec)
plt.xlabel('model predict data')
plt.grid(True)
plt.subplots_adjust(hspace=0.5)
plt.show()

plt.plot(index, select_ori, color='blue', label='testing data')  # 蓝色线
plt.plot(index, select_rec, color='green', label='model predict data')  # 绿色线
plt.legend()  # 显示图例
plt.grid(True)
plt.show()
print("vae lstm rmse : ", vae_lstm_rmse)
print("vae lstm r2 : ", vae_lstm_r2)