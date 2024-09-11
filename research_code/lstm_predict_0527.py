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
file_path = 'C:\\Users\\USER\\Desktop\\1220.csv'
data = pd.read_csv(file_path)

param_columns = data.iloc[:, 1:4].astype(float)
third_column = data.iloc[:, 3]
first_nonzero_index = (third_column != 0).idxmax()
subseq_after_first_nonzero = third_column.iloc[first_nonzero_index + 1:]
first_zero_after_nonzero_index = data.shape[0]
print(first_nonzero_index)



temp = param_columns.iloc[first_nonzero_index : data.shape[0] - 1, 0].values
temp = np.nan_to_num(temp)
ph = param_columns.iloc[first_nonzero_index : data.shape[0] - 1, 1].values
ph = np.nan_to_num(ph)
ec = param_columns.iloc[first_nonzero_index : data.shape[0] - 1, 2].values
ec = np.nan_to_num(ec)


leaky_relu = LeakyReLU(alpha=0.01)

N = data.shape[0] - 1 - first_nonzero_index
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

temp_coeffs , ph_coeffs, ec_coeffs = wavelet_decomposition(zscore_temp, zscore_ph, zscore_ec, 1)

temp_coeffs, ph_coeffs, ec_coeffs = semi_soft_thresholding(temp_coeffs, ph_coeffs, ec_coeffs)

temp_rec, ph_rec, ec_rec= wavelet_reconstruction(temp_coeffs, temp, ph_coeffs, ph, ec_coeffs, ec)



zscore_temp_rec = zscore(temp_rec)
test_zscore_temp_rec = zscore(temp_rec)
L = 30
batch_size = 128
epochs = 200

input_data = np.zeros((N, L))
target_data = np.zeros(N)


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

test_predictions = lstm.predict(test_input_data)
print(len(test_target_data), len(test_predictions))
vae_lstm_r2 = r2_score(test_target_data, test_predictions)
vae_lstm_rmse = np.sqrt(mean_squared_error(test_target_data, test_predictions))
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


select_ori = test_target_data[0:1900]
select_rec = test_predictions[0:1900]

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
plt.xlabel('Sample')
plt.ylabel('pH')
plt.legend()
plt.grid(True)
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(index, select_ori, color='blue', label='Testing data')
ax.plot(index, select_rec, color='green', label='Model predicted data')
ax.set_xlabel('Sample')
ax.set_ylabel('Temperature')
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

print("vae lstm rmse : ", vae_lstm_rmse)
print("vae lstm r2 : ", vae_lstm_r2)