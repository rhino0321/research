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
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, LSTM, Concatenate, LeakyReLU, TimeDistributed, Layer
from keras import backend as K
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
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

zscore_ec_rec = zscore(ec_rec)
test_zscore_ec_rec = zscore(test_ec_rec)
L = 2
m = 2
batch_size = 1024
epochs = 500

input_data = np.zeros((N, L))
target_data = np.zeros(N)

for i in range(L, N):
    input_data[i] = zscore_ec_rec[i - L : i]
    target_data[i] = zscore_ec_rec[i]

test_input_data = np.zeros((test_N, L))
test_target_data = np.zeros(test_N)
print(len(input_data))
for i in range(L, test_N):
    test_input_data[i] = test_zscore_ec_rec[i - L : i]
    test_target_data[i] = test_zscore_ec_rec[i]

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
t_target_data = t_target_data * np.std(test_ec_rec) + np.mean(test_ec_rec)
test_predictions = lstm_test_predictions * np.std(test_ec_rec) + np.mean(test_ec_rec)


select_ori = t_target_data[0:2000]
select_rec = test_predictions[0:2000]

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

print("vae lstm rmse : ", lstm_rmse)
print("vae lstm r2 : ", lstm_r2)
training_time = end_time - start_time
print(f"Model training time: {training_time:.2f} seconds")
