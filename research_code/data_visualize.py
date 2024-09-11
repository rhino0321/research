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
from scipy.fft import fft, fftfreq
import tensorflow as tf
from keras import layers, Model
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, LSTM, Concatenate, LeakyReLU, TimeDistributed, Layer, GRU
from keras import backend as K
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import ConnectionPatch
file_path = 'C:\\Users\\USER\\Desktop\\1220.csv'
data = pd.read_csv(file_path)

param_columns = data.iloc[:, 1:4].astype(float)
third_column = data.iloc[:, 3]
first_nonzero_index = (third_column != 0).idxmax()
subseq_after_first_nonzero = third_column.iloc[first_nonzero_index + 1:]
first_zero_after_nonzero_index = data.shape[0]
print(first_nonzero_index)


temp = param_columns.iloc[first_nonzero_index : first_zero_after_nonzero_index - 1, 0].values
temp = np.nan_to_num(temp)
ph = param_columns.iloc[first_nonzero_index : first_zero_after_nonzero_index - 1, 1].values
ph = np.nan_to_num(ph)
ec = param_columns.iloc[first_nonzero_index : first_zero_after_nonzero_index - 1, 2].values
ec = np.nan_to_num(ec)


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
    temp_rec, ph_rec, ec_rec= wavelet_reconstruction(temp_coeffs, temp_noise, ph_coeffs, ph, ec_coeffs, ec)
    print(len(temp_rec))
    temp_snr_list.append(10 * np.log10(np.sum(np.abs(temp_noise)) / np.sum(np.abs(temp_rec - temp))))
    temp_rmse_list.append(np.sqrt(mean_squared_error(temp, temp_rec)))
    ph_snr_list.append(10 * np.log10(np.sum(np.abs(ph)) / np.sum(np.abs(ph_rec - ph))))
    ph_rmse_list.append(np.sqrt(mean_squared_error(ph, ph_rec)))
    ec_snr_list.append(10 * np.log10(np.sum(np.abs(ec)) / np.sum(np.abs(ec_rec - ec))))
    ec_rmse_list.append(np.sqrt(mean_squared_error(ec, ec_rec)))
    temp_appro_entropy_list.append(approx_entropy(temp_coeffs))
    ph_appro_entropy_list.append(approx_entropy(ph_coeffs))
    ec_appro_entropy_list.append(approx_entropy(ec_coeffs))

#plot original data
#original_data(temp_noise, ph, ec, first_nonzero_index)

#plot snr entropy rmse 
snr_entropy_rmse(max_decomposition_level, ec_snr_list, ec_rmse_list, ec_appro_entropy_list, temp_snr_list, temp_rmse_list, temp_appro_entropy_list, ph_snr_list, ph_rmse_list, ph_appro_entropy_list)

'''
index = range(0, 1250)
select_ori = ph_noise[250:1500]
temp_coeffs , ph_coeffs, ec_coeffs = wavelet_decomposition(zscore_temp, zscore_ph, zscore_ec, 1)

temp_coeffs, ph_coeffs, ec_coeffs = semi_soft_thresholding(temp_coeffs, ph_coeffs, ec_coeffs)

temp_rec, ph_rec, ec_rec= wavelet_reconstruction(temp_coeffs, temp_noise, ph_coeffs, ph_noise, ec_coeffs, ec_noise)
scalar = MinMaxScaler()
zscore_temp_rec = zscore(temp_rec)
'''
select_rec = ph_rec[250:1500]


plt.plot(index, select_ori)
plt.xlabel('Original Data')
plt.ylabel('Temperature')
plt.grid(True)
plt.show()

plt.plot(index, select_rec)
plt.xlabel('after Wavelet Transform (Level 8)')
plt.ylabel('Temperature')
plt.grid(True)
plt.show()
'''

#VAE-RESLSTM MODEL
L = 15
m = 2
batch_size = 128
epochs = 200
train_ratio = 0.8
val_test_ratio = 0.2
latent_dim = 15
H1_cell = 60
H2_cell = 90
#VAE Encoder
input_refluctuation = Input(shape = (L, m), name = "input_refluctuation")
input_data = Input(shape = (L, ), name = "input_data")
leaky_relu = LeakyReLU(alpha=0.25)
flatten_x = Flatten()(input_refluctuation)
H1 = Dense(H1_cell * 2, activation = leaky_relu, name = "H1")(flatten_x)
H2 = Dense(H1_cell * 4, activation = leaky_relu, name = "H2")(H1)
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
lstm_layers = [LSTM(units=128, return_sequences=True)(lstm_input) for lstm_input in lstm_inputs]
concatenated_output = Concatenate()(lstm_layers)
flattened_output = Flatten()(concatenated_output)
last_hidden_state = Lambda(lambda x: x[:, 128:])(flattened_output)
LSTM_H1 = Dense(H1_cell * 2, activation = leaky_relu, name = "LSTM_H1")(last_hidden_state)
LSTM_H2 = Dense(H1_cell * 4, activation = leaky_relu, name = "LSTM_H2")(LSTM_H1)
predict_val = Dense(1)(LSTM_H2)
#lstm decoder....predict...output

#VAE Decoder
decoder_input = Input(shape = (latent_dim, ), name = "Decoder_input")
H6 = Dense(H1_cell * 4, activation = leaky_relu, name = "H6")(decoder_input)
H7 = Dense(H1_cell * 2, activation = leaky_relu, name = "H7")(H6)
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
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
vae_lstm.compile(optimizer = optimizer, loss = vae_loss)
vae_lstm.summary()

recent_fluctuation_matrix = np.zeros((N, L, m))

for num in range(L, N):
    bottom = num - L
    for i in range(0, L):
        Max = bottom + i
        for j in range(0, m):
            if (Max - (m - 1 - j) >= bottom and Max - (m - 1 - j) >= 0):
                recent_fluctuation_matrix[num, i, j] = zscore_temp_rec[Max - (m - 1 - j)]
input_data = np.zeros((N, L))
target_data = np.zeros(N)
print(len(input_data))
for i in range(L, N):
    input_data[i] = zscore_temp_rec[i - L : i]
    target_data[i] = zscore_temp_rec[i]

train_size = int((N - L) * 0.6)
train_refluctuation_matrix = recent_fluctuation_matrix[L : L + train_size]
train_input_data = input_data[L : L + train_size]
train_target_data = target_data[L : L + train_size]

val_refluctuation_matrix = recent_fluctuation_matrix[L + train_size : L + int((N - L) * 0.8)]
val_input_data = input_data[L + train_size : L + int((N - L) * 0.8)]
val_target_data = target_data[L + train_size : L + int((N - L) * 0.8)]

test_refluctuation_matrix = recent_fluctuation_matrix[L + int((N - L) * 0.8):]
test_input_data = input_data[L + int((N - L) * 0.8):]
test_target_data = target_data[L + int((N - L) * 0.8):]


'''
vae_lstm.save('./model/my_model.h5')

train_loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(1, len(train_loss) + 1)


plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training, validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
'''
R2_score = []
rmse_score = []
vae_lstm_start_time = time.time()
history = vae_lstm.fit(
    [train_refluctuation_matrix, train_input_data],
    [train_refluctuation_matrix, train_target_data], 
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([val_refluctuation_matrix, val_input_data], [val_refluctuation_matrix, val_target_data])
)
vae_lstm_end_time = time.time()
decoded_data, test_predictions = vae_lstm.predict([test_refluctuation_matrix, test_input_data])
vae_lstm_r2 = r2_score(test_target_data, test_predictions)
vae_lstm_rmse = np.sqrt(mean_squared_error(test_target_data, test_predictions))
R2_score.append(vae_lstm_r2)
rmse_score.append(vae_lstm_rmse)

DS = []
for i in range (1, len(test_target_data)):
    if ((test_target_data[i] - test_target_data[i - 1]) * (test_predictions[i] - test_target_data[i - 1]) > 0):
        DS.append(1)
    elif ((test_target_data[i] - test_target_data[i - 1]) * (test_predictions[i] - test_target_data[i - 1]) < 0):
        DS.append(0)
DS_value = (100 / len(DS)) * sum(DS)
print("DS val : ", DS_value)
index = range(0, 1900)
test_target_data = test_target_data * np.std(temp_rec) + np.mean(temp_rec)
test_predictions = test_predictions * np.std(temp_rec) + np.mean(temp_rec)

#########################################################################
input_data = np.zeros((N, L))
target_data = np.zeros(N)
print(len(input_data))
for i in range(L, N):
    input_data[i] = zscore_temp_rec[i - L : i]
    target_data[i] = zscore_temp_rec[i]

train_size = int((N - L) * 0.6)

train_input_data = input_data[L : L + train_size]
train_target_data = target_data[L : L + train_size]


val_input_data = input_data[L + train_size : L + int((N - L) * 0.8)]
val_target_data = target_data[L + train_size : L + int((N - L) * 0.8)]


test_input_data = input_data[L + int((N - L) * 0.8):]
test_target_data = target_data[L + int((N - L) * 0.8):]

input_data = Input(shape = (L, ), name = "input_data")
lstm_input_reshaped = Reshape((L, 1))(input_data)
GRU_L1 = GRU(units = 16, activation = 'sigmoid', return_sequences = False)(lstm_input_reshaped)
predict_val = Dense(1)(GRU_L1)
gru = Model(inputs = input_data, outputs = predict_val, name = "lstm")

def custom_loss(B):
    def loss(y_true, y_pred):
        prediction_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred)) / B

        return prediction_loss
    return loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
gru_loss = custom_loss(batch_size)
gru.compile(optimizer = optimizer, loss = gru_loss)

GRU_start_time = time.time()
gru_history = gru.fit(
    train_input_data,
    train_target_data, 
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(val_input_data, val_target_data)
)
GRU_end_time = time.time()
'''
train_loss = gru_history.history['loss']
val_loss = gru_history.history['val_loss']


gru_epochs = range(1, len(train_loss) + 1)


plt.plot(gru_epochs, train_loss, 'b', label='Training loss')
plt.plot(gru_epochs, val_loss, 'r', label='Validation loss')
plt.title('Training, validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
'''
test_GRU_predictions = gru.predict(test_input_data)
GRU_r2 = r2_score(test_target_data, test_GRU_predictions)
GRU_rmse = np.sqrt(mean_squared_error(test_target_data, test_GRU_predictions))

index = range(0, 1900)
test_target_data = test_target_data * np.std(temp_rec) + np.mean(temp_rec)
test_GRU_predictions = test_GRU_predictions * np.std(temp_rec) + np.mean(temp_rec)
#############################################################################
input_data = np.zeros((N, L))
target_data = np.zeros(N)
print(len(input_data))
for i in range(L, N):
    input_data[i] = zscore_temp_rec[i - L : i]
    target_data[i] = zscore_temp_rec[i]

train_size = int((N - L) * 0.6)

train_input_data = input_data[L : L + train_size]
train_target_data = target_data[L : L + train_size]


val_input_data = input_data[L + train_size : L + int((N - L) * 0.8)]
val_target_data = target_data[L + train_size : L + int((N - L) * 0.8)]


test_input_data = input_data[L + int((N - L) * 0.8):]
test_target_data = target_data[L + int((N - L) * 0.8):]

input_data = Input(shape = (L, ), name = "input_data")
lstm_input_reshaped = Reshape((L, 1))(input_data)
GRU_L1 = LSTM(units = 16, activation = 'sigmoid', return_sequences = False)(lstm_input_reshaped)
predict_val = Dense(1)(GRU_L1)
lstm = Model(inputs = input_data, outputs = predict_val, name = "lstm")

def custom_loss(B):
    def loss(y_true, y_pred):
        prediction_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred)) / B

        return prediction_loss
    return loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
lstm_loss = custom_loss(batch_size)
lstm.compile(optimizer = optimizer, loss = lstm_loss)

LSTM_start_time = time.time()
history = lstm.fit(
    train_input_data,
    train_target_data, 
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(val_input_data, val_target_data)
)
LSTM_end_time = time.time()
'''
train_loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(1, len(train_loss) + 1)


plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training, validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
'''
test_lstm_predictions = lstm.predict(test_input_data)
LSTM_r2 = r2_score(test_target_data, test_lstm_predictions)
LSTM_rmse = np.sqrt(mean_squared_error(test_target_data, test_lstm_predictions))

index = range(0, 1900)
test_target_data = test_target_data * np.std(temp_rec) + np.mean(temp_rec)
test_lstm_predictions = test_lstm_predictions * np.std(temp_rec) + np.mean(temp_rec)
#############################################################################
select_ori = test_target_data[0:1900]
select_rec = test_predictions[0:1900]
select_GRU_rec = test_GRU_predictions[0:1900]
select_lstm_rec = test_lstm_predictions[0:1900]

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

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(index, select_ori, color='blue', label='Testing data')
ax.plot(index, select_rec, color='green', label='Model predicted data')
ax.set_xlabel('Sample')
ax.set_ylabel('pH')
ax.legend(loc='lower left')

# 创建插图并放置在图的右下角
ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower right')
ax_inset.plot(index, select_ori, color='blue', label='Testing data')
ax_inset.plot(index, select_rec, color='green', label='Model predicted data')
ax_inset.set_xlim(1100, 1200)
ax_inset.set_ylim(min(select_ori[1100:1200].min(), select_rec[1100:1200].min()), max(select_ori[1100:1200].max(), select_rec[1100:1200].max()))

# 隐藏插图的网格和坐标轴标签
ax_inset.grid(False)
ax_inset.set_xticklabels([])
#ax_inset.set_yticklabels([])

# 添加矩形框以突出插图中的区域
rect = plt.Rectangle((1100, min(select_ori[1100:1200].min(), select_rec[1100:1200].min())),
                     100, max(select_ori[1100:1200].max(), select_rec[1100:1200].max()) - min(select_ori[1100:1200].min(), select_rec[1100:1200].min()),
                     linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect)

# 连接矩形框的左下角到插图的左上角
con1 = ConnectionPatch(xyA=(1100, min(select_ori[1100:1200].min(), select_rec[1100:1200].min())), xyB=(0, 1),
                       coordsA="data", coordsB="axes fraction", axesA=ax, axesB=ax_inset,
                       color="black", lw=1.5, linestyle='--')
ax.add_artist(con1)

# 连接矩形框的右下角到插图的右上角
con2 = ConnectionPatch(xyA=(1200, min(select_ori[1100:1200].min(), select_rec[1100:1200].min())), xyB=(1, 1),
                       coordsA="data", coordsB="axes fraction", axesA=ax, axesB=ax_inset,
                       color="black", lw=1.5, linestyle='--')
ax.add_artist(con2)

plt.show()


vae_lstm_training_time = vae_lstm_end_time - vae_lstm_start_time
gru_training_time = GRU_end_time - GRU_start_time
lstm_training_time = LSTM_end_time - LSTM_start_time
print(f"VAE-LSTM Model training time: {vae_lstm_training_time:.2f} seconds")
print(f"GRU Model training time: {gru_training_time:.2f} seconds")
print(f"LSTM Model training time: {lstm_training_time:.2f} seconds")

print("vae lstm r2 : ", vae_lstm_r2)
print("GRU r2 : ", GRU_r2)
print("LSTM r2 : ", LSTM_r2)
print("vae lstm rmse : ", vae_lstm_rmse)
print("GRU rmse : ", GRU_rmse)
print("LSTM rmse : ", LSTM_rmse)