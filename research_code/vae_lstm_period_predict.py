import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import pywt
import pickle 
import gzip
import os
import numpy as np
from scipy.stats import entropy, zscore, skew, kurtosis
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error, r2_score
from functions import semi_soft_thresholding, wavelet_decomposition, wavelet_reconstruction, approx_entropy
from plot_functions import snr_entropy_rmse, original_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import ConnectionPatch
from scipy.fft import fft, fftfreq
import tensorflow as tf
from keras import layers, Model
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, LSTM, Concatenate, LeakyReLU, TimeDistributed, Layer, GRU
from keras import backend as K
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf
file_path = 'C:\\Users\\USER\\Desktop\\nycu-data\\log.csv'
file_test_path = 'C:\\Users\\USER\\Desktop\\nycu-data\\log(0529).csv'
data = pd.read_csv(file_path)
data_test = pd.read_csv(file_test_path)

param_columns = data.iloc[:, 1:4].astype(float)
third_column = data.iloc[:, 3]
first_nonzero_index = (third_column != 0).idxmax()
subseq_after_first_nonzero = third_column.iloc[first_nonzero_index + 1:]
first_zero_after_nonzero_index = (subseq_after_first_nonzero < 100).idxmax()
print(first_nonzero_index)

test_param_columns = data_test.iloc[:, 1:4].astype(float)
test_third_column = data_test.iloc[:, 3]
test_first_nonzero_index = (test_third_column != 0).idxmax()
test_subseq_after_first_nonzero = test_third_column.iloc[test_first_nonzero_index + 1:]
test_first_zero_after_nonzero_index = data_test.shape[0]
#test_first_zero_after_nonzero_index = (test_subseq_after_first_nonzero < 100).idxmax()
print(test_first_nonzero_index, test_first_zero_after_nonzero_index)


temp = param_columns.iloc[first_nonzero_index : first_zero_after_nonzero_index - 1, 0].values
temp = np.nan_to_num(temp)
ph = param_columns.iloc[first_nonzero_index : first_zero_after_nonzero_index - 1, 1].values
ph = np.nan_to_num(ph)
ec = param_columns.iloc[first_nonzero_index : first_zero_after_nonzero_index - 1, 2].values
ec = np.nan_to_num(ec)

test_temp = test_param_columns.iloc[test_first_nonzero_index : test_first_zero_after_nonzero_index - 1, 0].values
test_temp = np.nan_to_num(test_temp)
test_ph = test_param_columns.iloc[test_first_nonzero_index : test_first_zero_after_nonzero_index - 1, 1].values
test_ph = np.nan_to_num(test_ph)
test_ec = test_param_columns.iloc[test_first_nonzero_index : test_first_zero_after_nonzero_index - 1, 2].values
test_ec = np.nan_to_num(test_ec)

min_length = min(len(temp), len(test_temp))
data_monday = temp[:min_length]
data_tuesday = test_temp[:min_length]

# 计算皮尔逊相关系数
pearson_corr = np.corrcoef(data_monday, data_tuesday)[0, 1]

print("皮尔逊相关系数:", pearson_corr)

leaky_relu = LeakyReLU(alpha=0.01)

N = first_zero_after_nonzero_index - 1 - first_nonzero_index
print(N)
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
noise = np.random.normal(0, 0.005, size=N)
temp_noise = temp + noise
ph_noise = ph + noise
ec_noise = ec + noise
zscore_temp = zscore(temp_noise)
zscore_ph = zscore(ph_noise)
zscore_ec = zscore(ec_noise)

test_N = test_first_zero_after_nonzero_index - 1 - test_first_nonzero_index
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
test_noise = np.random.normal(0, 0.005, size=test_N)
test_temp_noise = test_temp + test_noise
test_ph_noise = test_ph + test_noise
test_ec_noise = test_ec + test_noise
test_zscore_temp = zscore(test_temp_noise)
test_zscore_ph = zscore(test_ph_noise)
test_zscore_ec = zscore(test_ec_noise)

max_decomposition_level = math.floor(math.log2(N))

temp_coeffs , ph_coeffs, ec_coeffs = wavelet_decomposition(zscore_temp, zscore_ph, zscore_ec, 1)

temp_coeffs, ph_coeffs, ec_coeffs = semi_soft_thresholding(temp_coeffs, ph_coeffs, ec_coeffs)

temp_rec, ph_rec, ec_rec= wavelet_reconstruction(temp_coeffs, temp, ph_coeffs, ph, ec_coeffs, ec)

temp_coeffs , ph_coeffs, ec_coeffs = wavelet_decomposition(test_zscore_temp, test_zscore_ph, test_zscore_ec, 1)

temp_coeffs, ph_coeffs, ec_coeffs = semi_soft_thresholding(temp_coeffs, ph_coeffs, ec_coeffs)

test_temp_rec, test_ph_rec, test_ec_rec= wavelet_reconstruction(temp_coeffs, test_temp_noise, ph_coeffs, test_ph, ec_coeffs, test_ec)

zscore_temp_rec = zscore(temp_rec)
test_zscore_temp_rec = zscore(test_temp_rec)

L = 2
m = 2
batch_size = 512
epochs = 200

input_data = np.zeros((N, L))
target_data = np.zeros(N)
recent_fluctuation_matrix = np.zeros((N, L, m))

for num in range(L, N):
    bottom = num - L
    for i in range(0, L):
        Max = bottom + i
        for j in range(0, m):
            if (Max - (m - 1 - j) >= bottom and Max - (m - 1 - j) >= 0):
                recent_fluctuation_matrix[num, i, j] = zscore_temp_rec[Max - (m - 1 - j)]

for i in range(L, N):
    input_data[i] = zscore_temp_rec[i - L : i]
    target_data[i] = zscore_temp_rec[i]

test_input_data = np.zeros((test_N, L))
test_target_data = np.zeros(test_N)
test_recent_fluctuation_matrix = np.zeros((test_N, L, m))

for num in range(L, test_N):
    bottom = num - L
    for i in range(0, L):
        Max = bottom + i
        for j in range(0, m):
            if (Max - (m - 1 - j) >= bottom and Max - (m - 1 - j) >= 0):
                test_recent_fluctuation_matrix[num, i, j] = test_zscore_temp_rec[Max - (m - 1 - j)]


for i in range(L, test_N):
    test_input_data[i] = test_zscore_temp_rec[i - L : i]
    test_target_data[i] = test_zscore_temp_rec[i]

min_len = min(N, test_N)

train_size = int((min_len - L) * 0.6)
train_input_data = input_data[L : L + train_size]
train_refluctuation_matrix = recent_fluctuation_matrix[L : L + train_size]
train_target_data = target_data[L : L + train_size]

val_input_data = input_data[L + train_size : ]
val_target_data = target_data[L + train_size : ]
val_refluctuation_matrix = recent_fluctuation_matrix[L + train_size : ]

t_input_data = test_input_data[L :]
t_target_data = test_target_data[L :]
t_refluctuation_matrix = test_recent_fluctuation_matrix[L :]


train_ratio = 0.8
val_test_ratio = 0.2
latent_dim = 2
H1_cell = 2
H2_cell = 90
#VAE Encoder
input_refluctuation = Input(shape = (L, m), name = "input_refluctuation")
input_data = Input(shape = (L, ), name = "input_data")
leaky_relu = LeakyReLU(alpha=0.01)
flatten_x = Flatten()(input_refluctuation)
H1 = Dense(H1_cell, activation = leaky_relu, name = "H1")(flatten_x)
H2 = Dense(H1_cell * 2, activation = leaky_relu, name = "H2")(H1)
#z = Dense(latent_dim, name = "latent_space")(H2)
#H2 = Dense(H2_cell, activation = 'relu', name = "H2")(H1)

z_mean = Dense(latent_dim, name = "mean")(H2)
z_log_var = Dense(latent_dim, name = "log_variance")(H2)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape = (K.shape(z_mean)[0], latent_dim), mean = 0., stddev = 1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon
z = Lambda(sampling, output_shape = (latent_dim,), name = "encoder_output")([z_mean, z_log_var])

encoder = Model(input_refluctuation, [z_mean, z_log_var, z], name = "encoder")
encoder.summary()

combined_lstm_input = Concatenate(axis = 1)([z, input_data])
combined_lstm_input_reshaped = Reshape((L, 2))(combined_lstm_input)
#bottle neck
#lstm...
#LSTM_output = LSTM(units = latent_dim * 30, input_shape=(latent_dim, 2), return_sequences = False)(combined_lstm_input_reshaped)
#LSTM_H1 = Dense(latent_dim, activation = leaky_relu, name = "LSTM_H1")(LSTM_output)
#predict_val = Dense(1)(LSTM_H1)

lstm_inputs = Lambda(lambda x: [x[:, i:i+1, :] for i in range(latent_dim)])(combined_lstm_input_reshaped)
lstm_layers = [LSTM(units=1, return_sequences=True)(lstm_input) for lstm_input in lstm_inputs]
concatenated_output = Concatenate()(lstm_layers)
flattened_output = Flatten()(concatenated_output)
last_hidden_state = Lambda(lambda x: x[:, 1:])(flattened_output)
LSTM_H1 = Dense(H1_cell, activation = leaky_relu, name = "LSTM_H1")(last_hidden_state)
LSTM_H2 = Dense(H1_cell * 2, activation = leaky_relu, name = "LSTM_H2")(LSTM_H1)
predict_val = Dense(1)(LSTM_H2)
#lstm decoder....predict...output

#VAE Decoder
decoder_input = Input(shape = (latent_dim, ), name = "Decoder_input")
H6 = Dense(H1_cell * 2, activation = leaky_relu, name = "H6")(decoder_input)
H7 = Dense(H1_cell, activation = leaky_relu, name = "H7")(H6)
decoder_non_flatten = Dense(L * m, name = "decoder_output")(H7)
decoder_output = Reshape((L, m))(decoder_non_flatten)
decoder = Model(decoder_input, decoder_output, name = "decoder")
decoder.summary()

decoder_output = decoder(encoder(input_refluctuation)[2])


vae_lstm = Model(inputs = [input_refluctuation, input_data], outputs = [decoder_output, predict_val], name = "vae_lstm")

def custom_loss(omega1, omega2, B, m):
    def loss(y_true, y_pred):
    
        reconstuction_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true[0], y_pred[0])) / m / B
        prediction_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true[1], y_pred[1])) / B

        omega1_loss = reconstuction_loss / tf.exp(omega1)
        omega2_loss = (prediction_loss) / tf.exp(omega2)
        total_loss = 0.5 * (omega1_loss + omega2_loss + omega1 + omega2)

        return total_loss
    return loss

omega1 = tf.Variable(initial_value = 0.1, trainable = True, name = 'omega1')
omega2 = tf.Variable(initial_value = 0.1, trainable = True, name = 'omega2')
vae_loss = custom_loss(omega1, omega2, batch_size, m)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.007)
vae_lstm.compile(optimizer = optimizer, loss = vae_loss)
vae_lstm.summary()
start_time = time.time()
history = vae_lstm.fit(
    [train_refluctuation_matrix, train_input_data],
    [train_refluctuation_matrix, train_target_data], 
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([val_refluctuation_matrix, val_input_data], [val_refluctuation_matrix, val_target_data])
)
end_time = time.time()


decoded_data, vae_lstm_test_predictions = vae_lstm.predict([t_refluctuation_matrix, t_input_data])
lstm_rmse = np.sqrt(mean_squared_error(t_target_data, vae_lstm_test_predictions))
lstm_r2 = r2_score(t_target_data, vae_lstm_test_predictions)
DS = []
for i in range (1, len(t_target_data)):
    if ((t_target_data[i] - t_target_data[i - 1]) * (vae_lstm_test_predictions[i] - t_target_data[i - 1]) >= 0):
        DS.append(1)
    elif ((t_target_data[i] - t_target_data[i - 1]) * (vae_lstm_test_predictions[i] - t_target_data[i - 1]) < 0):
        DS.append(0)
DS_value = (100 / len(DS)) * sum(DS)
print("DS val : ", DS_value)
index = range(0, 2000)
t_target_data = t_target_data * np.std(test_temp_rec) + np.mean(test_temp_rec)
vae_lstm_test_predictions = vae_lstm_test_predictions * np.std(test_temp_rec) + np.mean(test_temp_rec)



'''
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(index, select_ori, color='blue', label='Testing data')
ax.plot(index, select_rec, color='green', label='Model predicted data')
ax.set_xlabel('Sample')
ax.set_ylabel(r'Electrical Conductivity $\mu$S/cm')
ax.legend()

# 创建插图

ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper center')
ax_inset.plot(index, select_ori, color='blue', label='Testing data')
ax_inset.plot(index, select_rec, color='green', label='Model predicted data')
ax_inset.set_xlim(600, 700)
ax_inset.set_ylim(min(select_ori[600:700].min(), select_rec[600:700].min()), max(select_ori[600:700].max(), select_rec[600:700].max()))
ax_inset.set_xticklabels([])  # 隐藏x轴标签
#ax_inset.set_yticklabels([])  # 隐藏y轴标签

# 在主图上框选区域并连接到插图
from matplotlib.patches import ConnectionPatch

# 框选区域
rect = plt.Rectangle((600, min(select_ori[600:700].min(), select_rec[600:700].min())),
                     100, max(select_ori[600:700].max(), select_rec[600:700].max()) - min(select_ori[600:700].min(), select_rec[600:700].min()),
                     linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect)

# 添加连接线并设置颜色和样式
start_point = (700, max(select_ori[600:700].max(), select_rec[600:700].max()))
end_point = (600, max(select_ori[600:700].min(), select_rec[600:700].min()))

# 创建连接线并设置颜色和样式
con = ConnectionPatch(xyA=start_point, xyB=(0, 0), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)

# 指定连接线的起始点和终止点
start_point = (700, min(select_ori[600:700].min(), select_rec[600:700].min()))
end_point = (600, min(select_ori[600:700].max(), select_rec[600:700].max()))

# 创建连接线并设置颜色和样式
con = ConnectionPatch(xyA=start_point, xyB=(1, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)

plt.show()
'''
'''
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(index, select_ori, color='blue', label='Testing data')
ax.plot(index, select_rec, color='green', label='Model predicted data')
ax.set_xlabel('Sample')
ax.set_ylabel('Temperature')
ax.legend()

# 创建插图并放置在图的底部中心
ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower center')
ax_inset.plot(index, select_ori, color='blue', label='Testing data')
ax_inset.plot(index, select_rec, color='green', label='Model predicted data')
ax_inset.set_xlim(1000, 1100)
ax_inset.set_ylim(min(select_ori[1000:1100].min(), select_rec[1000:1100].min()), max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))


ax_inset.grid(False)


ax_inset.set_xticklabels([])  
#ax_inset.set_yticklabels([])  


rect = plt.Rectangle((1000, min(select_ori[1000:1100].min(), select_rec[1000:1100].min())),
                     100, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()) - min(select_ori[1000:1100].min(), select_rec[1000:1100].min()),
                     linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect)


start_point = (1100, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))
end_point = (1000, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))


con = ConnectionPatch(xyA=start_point, xyB=(1, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)


start_point = (1100, min(select_ori[1000:1100].min(), select_rec[1000:1100].min()))
end_point = (1000, min(select_ori[1000:1100].min(), select_rec[1000:1100].min()))


con = ConnectionPatch(xyA=start_point, xyB=(0, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)

plt.show()
'''
print("vae lstm rmse : ", lstm_rmse)
print("vae lstm r2 : ", lstm_r2)
training_time = end_time - start_time
print(f"Model training time: {training_time:.2f} seconds")

##################################################################
param_columns = data.iloc[:, 1:4].astype(float)
third_column = data.iloc[:, 3]
first_nonzero_index = (third_column != 0).idxmax()
subseq_after_first_nonzero = third_column.iloc[first_nonzero_index + 1:]
first_zero_after_nonzero_index = (subseq_after_first_nonzero < 100).idxmax()
print(first_nonzero_index)

test_param_columns = data_test.iloc[:, 1:4].astype(float)
test_third_column = data_test.iloc[:, 3]
test_first_nonzero_index = (test_third_column != 0).idxmax()
test_subseq_after_first_nonzero = test_third_column.iloc[test_first_nonzero_index + 1:]
test_first_zero_after_nonzero_index = data_test.shape[0]
#test_first_zero_after_nonzero_index = (test_subseq_after_first_nonzero < 100).idxmax()
print(test_first_nonzero_index, test_first_zero_after_nonzero_index)


temp = param_columns.iloc[first_nonzero_index : first_zero_after_nonzero_index - 1, 0].values
temp = np.nan_to_num(temp)
ph = param_columns.iloc[first_nonzero_index : first_zero_after_nonzero_index - 1, 1].values
ph = np.nan_to_num(ph)
ec = param_columns.iloc[first_nonzero_index : first_zero_after_nonzero_index - 1, 2].values
ec = np.nan_to_num(ec)

test_temp = test_param_columns.iloc[test_first_nonzero_index : test_first_zero_after_nonzero_index - 1, 0].values
test_temp = np.nan_to_num(test_temp)
test_ph = test_param_columns.iloc[test_first_nonzero_index : test_first_zero_after_nonzero_index - 1, 1].values
test_ph = np.nan_to_num(test_ph)
test_ec = test_param_columns.iloc[test_first_nonzero_index : test_first_zero_after_nonzero_index - 1, 2].values
test_ec = np.nan_to_num(test_ec)

min_length = min(len(ec), len(test_ec))
data_monday = ec[:min_length]
data_tuesday = test_ec[:min_length]

# 计算皮尔逊相关系数
pearson_corr = np.corrcoef(data_monday, data_tuesday)[0, 1]
print("皮尔逊相关系数:", pearson_corr)

leaky_relu = LeakyReLU(alpha=0.01)

N = first_zero_after_nonzero_index - 1 - first_nonzero_index
print(N)
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
noise = np.random.normal(0, 0.005, size=N)
temp_noise = temp + noise
ph_noise = ph + noise
ec_noise = ec + noise
zscore_temp = zscore(temp_noise)
zscore_ph = zscore(ph_noise)
zscore_ec = zscore(ec_noise)

test_N = test_first_zero_after_nonzero_index - 1 - test_first_nonzero_index
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
test_noise = np.random.normal(0, 0.005, size=test_N)
test_temp_noise = test_temp + test_noise
test_ph_noise = test_ph + test_noise
test_ec_noise = test_ec + test_noise
test_zscore_temp = zscore(test_temp_noise)
test_zscore_ph = zscore(test_ph_noise)
test_zscore_ec = zscore(test_ec_noise)

max_decomposition_level = math.floor(math.log2(N))

temp_coeffs , ph_coeffs, ec_coeffs = wavelet_decomposition(zscore_temp, zscore_ph, zscore_ec, 1)

temp_coeffs, ph_coeffs, ec_coeffs = semi_soft_thresholding(temp_coeffs, ph_coeffs, ec_coeffs)

temp_rec, ph_rec, ec_rec= wavelet_reconstruction(temp_coeffs, temp, ph_coeffs, ph, ec_coeffs, ec)

temp_coeffs , ph_coeffs, ec_coeffs = wavelet_decomposition(test_zscore_temp, test_zscore_ph, test_zscore_ec, 1)

temp_coeffs, ph_coeffs, ec_coeffs = semi_soft_thresholding(temp_coeffs, ph_coeffs, ec_coeffs)

test_temp_rec, test_ph_rec, test_ec_rec= wavelet_reconstruction(temp_coeffs, test_temp_noise, ph_coeffs, test_ph, ec_coeffs, test_ec)

zscore_temp_rec = zscore(temp_rec)
test_zscore_temp_rec = zscore(test_temp_rec)
L = 2
m = 2
batch_size = 1024
epochs = 500

input_data = np.zeros((N, L))
target_data = np.zeros(N)

for i in range(L, N):
    input_data[i] = zscore_temp_rec[i - L : i]
    target_data[i] = zscore_temp_rec[i]

test_input_data = np.zeros((test_N, L))
test_target_data = np.zeros(test_N)
print(len(input_data))
for i in range(L, test_N):
    test_input_data[i] = test_zscore_temp_rec[i - L : i]
    test_target_data[i] = test_zscore_temp_rec[i]

train_size = int((N - L) * 0.8)
train_input_data = input_data[L : L + train_size]
train_target_data = target_data[L : L + train_size]

val_input_data = input_data[L + train_size : ]
val_target_data = target_data[L + train_size : ]

t_input_data = test_input_data[L :]
t_target_data = test_target_data[L :]

input_data = Input(shape = (L, ), name = "input_data")
lstm_input_reshaped = Reshape((L, 1))(input_data)
LSTM_L1 = LSTM(units = 1, activation = 'sigmoid', return_sequences = False, dropout = 0.2)(lstm_input_reshaped)
predict_val = Dense(1)(LSTM_L1)
lstm = Model(inputs = input_data, outputs = predict_val, name = "lstm")

def custom_loss(B):
    def loss(y_true, y_pred):
        prediction_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred)) / B

        return prediction_loss
    return loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
lstm_loss = custom_loss(batch_size)
lstm.compile(optimizer = optimizer, loss = lstm_loss)
start_time = time.time()
history = lstm.fit(
    train_input_data,
    train_target_data, 
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(val_input_data, val_target_data)
)
end_time = time.time()

lstm_test_predictions = lstm.predict(t_input_data)
lstm_rmse = np.sqrt(mean_squared_error(t_target_data, lstm_test_predictions))
lstm_r2 = r2_score(t_target_data, lstm_test_predictions)
DS = []
for i in range (1, len(t_target_data)):
    if ((t_target_data[i] - t_target_data[i - 1]) * (lstm_test_predictions[i] - t_target_data[i - 1]) >= 0):
        DS.append(1)
    elif ((t_target_data[i] - t_target_data[i - 1]) * (lstm_test_predictions[i] - t_target_data[i - 1]) < 0):
        DS.append(0)
DS_value = (100 / len(DS)) * sum(DS)
print("DS val : ", DS_value)
index = range(0, 2000)
t_target_data = t_target_data * np.std(test_temp_rec) + np.mean(test_temp_rec)
lstm_test_predictions = lstm_test_predictions * np.std(test_temp_rec) + np.mean(test_temp_rec)


'''
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(index, select_ori, color='blue', label='Testing data')
ax.plot(index, select_rec, color='green', label='Model predicted data')
ax.set_xlabel('Sample')
ax.set_ylabel(r'Electrical Conductivity $\mu$S/cm')
ax.legend()

# 创建插图

ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper center')
ax_inset.plot(index, select_ori, color='blue', label='Testing data')
ax_inset.plot(index, select_rec, color='green', label='Model predicted data')
ax_inset.set_xlim(600, 700)
ax_inset.set_ylim(min(select_ori[600:700].min(), select_rec[600:700].min()), max(select_ori[600:700].max(), select_rec[600:700].max()))
ax_inset.set_xticklabels([])  # 隐藏x轴标签
#ax_inset.set_yticklabels([])  # 隐藏y轴标签

# 在主图上框选区域并连接到插图
from matplotlib.patches import ConnectionPatch

# 框选区域
rect = plt.Rectangle((600, min(select_ori[600:700].min(), select_rec[600:700].min())),
                     100, max(select_ori[600:700].max(), select_rec[600:700].max()) - min(select_ori[600:700].min(), select_rec[600:700].min()),
                     linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect)

# 添加连接线并设置颜色和样式
start_point = (700, max(select_ori[600:700].max(), select_rec[600:700].max()))
end_point = (600, max(select_ori[600:700].min(), select_rec[600:700].min()))

# 创建连接线并设置颜色和样式
con = ConnectionPatch(xyA=start_point, xyB=(0, 0), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)

# 指定连接线的起始点和终止点
start_point = (700, min(select_ori[600:700].min(), select_rec[600:700].min()))
end_point = (600, min(select_ori[600:700].max(), select_rec[600:700].max()))

# 创建连接线并设置颜色和样式
con = ConnectionPatch(xyA=start_point, xyB=(1, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)

plt.show()
'''
'''
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(index, select_ori, color='blue', label='Testing data')
ax.plot(index, select_rec, color='green', label='Model predicted data')
ax.set_xlabel('Sample')
ax.set_ylabel('pH')
ax.legend()

# 创建插图并放置在图的底部中心
ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower center')
ax_inset.plot(index, select_ori, color='blue', label='Testing data')
ax_inset.plot(index, select_rec, color='green', label='Model predicted data')
ax_inset.set_xlim(1000, 1100)
ax_inset.set_ylim(min(select_ori[1000:1100].min(), select_rec[1000:1100].min()), max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))


ax_inset.grid(False)


ax_inset.set_xticklabels([])  
#ax_inset.set_yticklabels([])  


rect = plt.Rectangle((1000, min(select_ori[1000:1100].min(), select_rec[1000:1100].min())),
                     100, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()) - min(select_ori[1000:1100].min(), select_rec[1000:1100].min()),
                     linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect)


start_point = (1100, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))
end_point = (1000, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))


con = ConnectionPatch(xyA=start_point, xyB=(1, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)


start_point = (1100, min(select_ori[1000:1100].min(), select_rec[1000:1100].min()))
end_point = (1000, min(select_ori[1000:1100].min(), select_rec[1000:1100].min()))


con = ConnectionPatch(xyA=start_point, xyB=(0, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)

plt.show()

'''
'''
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(index, select_ori, color='blue', label='Testing data')
ax.plot(index, select_rec, color='green', label='Model predicted data')
ax.set_xlabel('Sample')
ax.set_ylabel('Temperature (Celsius)')
ax.legend()

# 创建插图并放置在图的底部中心
ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower right')
ax_inset.plot(index, select_ori, color='blue', label='Testing data')
ax_inset.plot(index, select_rec, color='green', label='Model predicted data')
ax_inset.set_xlim(1000, 1100)
ax_inset.set_ylim(min(select_ori[1000:1100].min(), select_rec[1000:1100].min()), max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))


ax_inset.grid(False)


ax_inset.set_xticklabels([])  
#ax_inset.set_yticklabels([])  


rect = plt.Rectangle((1000, min(select_ori[1000:1100].min(), select_rec[1000:1100].min())),
                     100, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()) - min(select_ori[1000:1100].min(), select_rec[1000:1100].min()),
                     linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect)


start_point = (1100, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))
end_point = (1000, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))


con = ConnectionPatch(xyA=start_point, xyB=(1, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)


start_point = (1100, min(select_ori[1000:1100].min(), select_rec[1000:1100].min()))
end_point = (1000, min(select_ori[1000:1100].min(), select_rec[1000:1100].min()))


con = ConnectionPatch(xyA=start_point, xyB=(0, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)

plt.show()
'''

print("lstm rmse : ", lstm_rmse)
print("lstm r2 : ", lstm_r2)
training_time = end_time - start_time
print(f"Model training time: {training_time:.2f} seconds")

#############################################################
param_columns = data.iloc[:, 1:4].astype(float)
third_column = data.iloc[:, 3]
first_nonzero_index = (third_column != 0).idxmax()
subseq_after_first_nonzero = third_column.iloc[first_nonzero_index + 1:]
first_zero_after_nonzero_index = (subseq_after_first_nonzero < 100).idxmax()
print(first_nonzero_index)

test_param_columns = data_test.iloc[:, 1:4].astype(float)
test_third_column = data_test.iloc[:, 3]
test_first_nonzero_index = (test_third_column != 0).idxmax()
test_subseq_after_first_nonzero = test_third_column.iloc[test_first_nonzero_index + 1:]
test_first_zero_after_nonzero_index = data_test.shape[0]
#test_first_zero_after_nonzero_index = (test_subseq_after_first_nonzero < 100).idxmax()
print(test_first_nonzero_index, test_first_zero_after_nonzero_index)


temp = param_columns.iloc[first_nonzero_index : first_zero_after_nonzero_index - 1, 0].values
temp = np.nan_to_num(temp)
ph = param_columns.iloc[first_nonzero_index : first_zero_after_nonzero_index - 1, 1].values
ph = np.nan_to_num(ph)
ec = param_columns.iloc[first_nonzero_index : first_zero_after_nonzero_index - 1, 2].values
ec = np.nan_to_num(ec)

test_temp = test_param_columns.iloc[test_first_nonzero_index : test_first_zero_after_nonzero_index - 1, 0].values
test_temp = np.nan_to_num(test_temp)
test_ph = test_param_columns.iloc[test_first_nonzero_index : test_first_zero_after_nonzero_index - 1, 1].values
test_ph = np.nan_to_num(test_ph)
test_ec = test_param_columns.iloc[test_first_nonzero_index : test_first_zero_after_nonzero_index - 1, 2].values
test_ec = np.nan_to_num(test_ec)

min_length = min(len(ec), len(test_ec))
data_monday = ec[:min_length]
data_tuesday = test_ec[:min_length]

# 计算皮尔逊相关系数
pearson_corr = np.corrcoef(data_monday, data_tuesday)[0, 1]
print("皮尔逊相关系数:", pearson_corr)

leaky_relu = LeakyReLU(alpha=0.01)

N = first_zero_after_nonzero_index - 1 - first_nonzero_index
print(N)
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
noise = np.random.normal(0, 0.005, size=N)
temp_noise = temp + noise
ph_noise = ph + noise
ec_noise = ec + noise
zscore_temp = zscore(temp_noise)
zscore_ph = zscore(ph_noise)
zscore_ec = zscore(ec_noise)

test_N = test_first_zero_after_nonzero_index - 1 - test_first_nonzero_index
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
test_noise = np.random.normal(0, 0.005, size=test_N)
test_temp_noise = test_temp + test_noise
test_ph_noise = test_ph + test_noise
test_ec_noise = test_ec + test_noise
test_zscore_temp = zscore(test_temp_noise)
test_zscore_ph = zscore(test_ph_noise)
test_zscore_ec = zscore(test_ec_noise)

max_decomposition_level = math.floor(math.log2(N))

temp_coeffs , ph_coeffs, ec_coeffs = wavelet_decomposition(zscore_temp, zscore_ph, zscore_ec, 1)

temp_coeffs, ph_coeffs, ec_coeffs = semi_soft_thresholding(temp_coeffs, ph_coeffs, ec_coeffs)

temp_rec, ph_rec, ec_rec= wavelet_reconstruction(temp_coeffs, temp, ph_coeffs, ph, ec_coeffs, ec)

temp_coeffs , ph_coeffs, ec_coeffs = wavelet_decomposition(test_zscore_temp, test_zscore_ph, test_zscore_ec, 1)

temp_coeffs, ph_coeffs, ec_coeffs = semi_soft_thresholding(temp_coeffs, ph_coeffs, ec_coeffs)

test_temp_rec, test_ph_rec, test_ec_rec= wavelet_reconstruction(temp_coeffs, test_temp_noise, ph_coeffs, test_ph, ec_coeffs, test_ec)

zscore_temp_rec = zscore(temp_rec)
test_zscore_temp_rec = zscore(test_temp_rec)
L = 2
m = 2
batch_size = 1024
epochs = 500

input_data = np.zeros((N, L))
target_data = np.zeros(N)

for i in range(L, N):
    input_data[i] = zscore_temp_rec[i - L : i]
    target_data[i] = zscore_temp_rec[i]

test_input_data = np.zeros((test_N, L))
test_target_data = np.zeros(test_N)
print(len(input_data))
for i in range(L, test_N):
    test_input_data[i] = test_zscore_temp_rec[i - L : i]
    test_target_data[i] = test_zscore_temp_rec[i]

train_size = int((N - L) * 0.8)
train_input_data = input_data[L : L + train_size]
train_target_data = target_data[L : L + train_size]

val_input_data = input_data[L + train_size : ]
val_target_data = target_data[L + train_size : ]

t_input_data = test_input_data[L :]
t_target_data = test_target_data[L :]

input_data = Input(shape = (L, ), name = "input_data")
lstm_input_reshaped = Reshape((L, 1))(input_data)
LSTM_L1 = GRU(units = 1, activation = 'sigmoid', return_sequences = False, dropout = 0.2)(lstm_input_reshaped)
predict_val = Dense(1)(LSTM_L1)
gru = Model(inputs = input_data, outputs = predict_val, name = "lstm")

def custom_loss(B):
    def loss(y_true, y_pred):
        prediction_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred)) / B

        return prediction_loss
    return loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
gru_loss = custom_loss(batch_size)
gru.compile(optimizer = optimizer, loss = lstm_loss)
start_time = time.time()
history = gru.fit(
    train_input_data,
    train_target_data, 
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(val_input_data, val_target_data)
)
end_time = time.time()


GRU_test_predictions = gru.predict(t_input_data)
lstm_rmse = np.sqrt(mean_squared_error(t_target_data, GRU_test_predictions))
lstm_r2 = r2_score(t_target_data, GRU_test_predictions)
DS = []
for i in range (1, len(t_target_data)):
    if ((t_target_data[i] - t_target_data[i - 1]) * (GRU_test_predictions[i] - t_target_data[i - 1]) >= 0):
        DS.append(1)
    elif ((t_target_data[i] - t_target_data[i - 1]) * (GRU_test_predictions[i] - t_target_data[i - 1]) < 0):
        DS.append(0)
DS_value = (100 / len(DS)) * sum(DS)
print("DS val : ", DS_value)
index = range(0, 2000)
t_target_data = t_target_data * np.std(test_temp_rec) + np.mean(test_temp_rec)
GRU_test_predictions = GRU_test_predictions * np.std(test_temp_rec) + np.mean(test_temp_rec)


select_ori = t_target_data[0:2000]
select_rec = vae_lstm_test_predictions[0:2000]
select_GRU_rec = GRU_test_predictions[0:2000]
select_lstm_rec = lstm_test_predictions[0:2000]

plt.plot(index, select_ori, color='blue', label='testing data')  
plt.plot(index, select_rec, color='green', label='VAE-LSTM predict data')  
plt.plot(index, select_GRU_rec, color='red', label='GRU predict data')
plt.plot(index, select_lstm_rec, color='grey', label='LSTM predict data')    
plt.xlabel('Sample')
#plt.ylabel('Temparature (Celsuis)')
plt.ylabel('Temperature(Celsius)')
plt.legend()
plt.grid(True)
plt.show()
'''
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(index, select_ori, color='blue', label='Testing data')
ax.plot(index, select_rec, color='green', label='Model predicted data')
ax.set_xlabel('Sample')
ax.set_ylabel(r'Electrical Conductivity $\mu$S/cm')
ax.legend()

# 创建插图

ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper center')
ax_inset.plot(index, select_ori, color='blue', label='Testing data')
ax_inset.plot(index, select_rec, color='green', label='Model predicted data')
ax_inset.set_xlim(600, 700)
ax_inset.set_ylim(min(select_ori[600:700].min(), select_rec[600:700].min()), max(select_ori[600:700].max(), select_rec[600:700].max()))
ax_inset.set_xticklabels([])  # 隐藏x轴标签
#ax_inset.set_yticklabels([])  # 隐藏y轴标签

# 在主图上框选区域并连接到插图
from matplotlib.patches import ConnectionPatch

# 框选区域
rect = plt.Rectangle((600, min(select_ori[600:700].min(), select_rec[600:700].min())),
                     100, max(select_ori[600:700].max(), select_rec[600:700].max()) - min(select_ori[600:700].min(), select_rec[600:700].min()),
                     linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect)

# 添加连接线并设置颜色和样式
start_point = (700, max(select_ori[600:700].max(), select_rec[600:700].max()))
end_point = (600, max(select_ori[600:700].min(), select_rec[600:700].min()))

# 创建连接线并设置颜色和样式
con = ConnectionPatch(xyA=start_point, xyB=(0, 0), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)

# 指定连接线的起始点和终止点
start_point = (700, min(select_ori[600:700].min(), select_rec[600:700].min()))
end_point = (600, min(select_ori[600:700].max(), select_rec[600:700].max()))

# 创建连接线并设置颜色和样式
con = ConnectionPatch(xyA=start_point, xyB=(1, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)

plt.show()
'''
'''
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(index, select_ori, color='blue', label='Testing data')
ax.plot(index, select_rec, color='green', label='Model predicted data')
ax.set_xlabel('Sample')
ax.set_ylabel('pH')
ax.legend()

# 创建插图并放置在图的底部中心
ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower center')
ax_inset.plot(index, select_ori, color='blue', label='Testing data')
ax_inset.plot(index, select_rec, color='green', label='Model predicted data')
ax_inset.set_xlim(1000, 1100)
ax_inset.set_ylim(min(select_ori[1000:1100].min(), select_rec[1000:1100].min()), max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))


ax_inset.grid(False)


ax_inset.set_xticklabels([])  
#ax_inset.set_yticklabels([])  


rect = plt.Rectangle((1000, min(select_ori[1000:1100].min(), select_rec[1000:1100].min())),
                     100, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()) - min(select_ori[1000:1100].min(), select_rec[1000:1100].min()),
                     linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect)


start_point = (1100, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))
end_point = (1000, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))


con = ConnectionPatch(xyA=start_point, xyB=(1, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)


start_point = (1100, min(select_ori[1000:1100].min(), select_rec[1000:1100].min()))
end_point = (1000, min(select_ori[1000:1100].min(), select_rec[1000:1100].min()))


con = ConnectionPatch(xyA=start_point, xyB=(0, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)

plt.show()

'''
'''
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(index, select_ori, color='blue', label='Testing data')
ax.plot(index, select_rec, color='green', label='Model predicted data')
ax.set_xlabel('Sample')
ax.set_ylabel('Temperature (Celsius)')
ax.legend()

# 创建插图并放置在图的底部中心
ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower right')
ax_inset.plot(index, select_ori, color='blue', label='Testing data')
ax_inset.plot(index, select_rec, color='green', label='Model predicted data')
ax_inset.set_xlim(1000, 1100)
ax_inset.set_ylim(min(select_ori[1000:1100].min(), select_rec[1000:1100].min()), max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))


ax_inset.grid(False)


ax_inset.set_xticklabels([])  
#ax_inset.set_yticklabels([])  


rect = plt.Rectangle((1000, min(select_ori[1000:1100].min(), select_rec[1000:1100].min())),
                     100, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()) - min(select_ori[1000:1100].min(), select_rec[1000:1100].min()),
                     linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect)


start_point = (1100, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))
end_point = (1000, max(select_ori[1000:1100].max(), select_rec[1000:1100].max()))


con = ConnectionPatch(xyA=start_point, xyB=(1, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)


start_point = (1100, min(select_ori[1000:1100].min(), select_rec[1000:1100].min()))
end_point = (1000, min(select_ori[1000:1100].min(), select_rec[1000:1100].min()))


con = ConnectionPatch(xyA=start_point, xyB=(0, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=ax_inset, color="black", lw=1.5, linestyle='--')
ax.add_artist(con)

plt.show()
'''

print("lstm rmse : ", lstm_rmse)
print("lstm r2 : ", lstm_r2)
training_time = end_time - start_time
print(f"Model training time: {training_time:.2f} seconds")