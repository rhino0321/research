import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import ConnectionPatch
from scipy.stats import zscore
from functions import semi_soft_thresholding, wavelet_decomposition, wavelet_reconstruction, approx_entropy
# 读取数据
file_path = 'C:\\Users\\USER\\Desktop\\nycu-data\\log.csv'
file_test_path = 'C:\\Users\\USER\\Desktop\\nycu-data\\log(0529).csv'
data = pd.read_csv(file_path)
data_test = pd.read_csv(file_test_path)

# 提取特征和处理
param_columns = data.iloc[:, 1:4].astype(float)
third_column = data.iloc[:, 3]
first_nonzero_index = (third_column != 0).idxmax()
subseq_after_first_nonzero = third_column.iloc[first_nonzero_index + 1:]
first_zero_after_nonzero_index = (subseq_after_first_nonzero == 0).idxmax()

test_param_columns = data_test.iloc[:, 1:4].astype(float)
test_third_column = data_test.iloc[:, 3]
test_first_nonzero_index = (test_third_column != 0).idxmax()
test_first_zero_after_nonzero_index = data_test.shape[0]

# 处理温度、pH值和电导率
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

# 标准化数据
N = first_zero_after_nonzero_index - 1 - first_nonzero_index
noise = np.random.normal(0, 0.005, size=N)
temp_noise = temp + noise
ph_noise = ph + noise
ec_noise = ec + noise
zscore_temp = zscore(temp_noise)
zscore_ph = zscore(ph_noise)
zscore_ec = zscore(ec_noise)

test_N = test_first_zero_after_nonzero_index - 1 - test_first_nonzero_index
test_noise = np.random.normal(0, 0.005, size=test_N)
test_temp_noise = test_temp + test_noise
test_ph_noise = test_ph + test_noise
test_ec_noise = test_ec + test_noise
test_zscore_temp = zscore(test_temp_noise)
test_zscore_ph = zscore(test_ph_noise)
test_zscore_ec = zscore(test_ec_noise)

temp_coeffs , ph_coeffs, ec_coeffs = wavelet_decomposition(zscore_temp, zscore_ph, zscore_ec, 1)

temp_coeffs, ph_coeffs, ec_coeffs = semi_soft_thresholding(temp_coeffs, ph_coeffs, ec_coeffs)

temp_rec, ph_rec, ec_rec= wavelet_reconstruction(temp_coeffs, temp, ph_coeffs, ph, ec_coeffs, ec)

temp_coeffs , ph_coeffs, ec_coeffs = wavelet_decomposition(test_zscore_temp, test_zscore_ph, test_zscore_ec, 1)

temp_coeffs, ph_coeffs, ec_coeffs = semi_soft_thresholding(temp_coeffs, ph_coeffs, ec_coeffs)

test_temp_rec, test_ph_rec, test_ec_rec= wavelet_reconstruction(temp_coeffs, test_temp_noise, ph_coeffs, test_ph, ec_coeffs, test_ec)

zscore_temp_rec = zscore(temp_rec)
test_zscore_temp_rec = zscore(test_temp_rec)

# 创建输入特征
L = 150
input_data = np.zeros((N - L, L))
target_data = np.zeros(N - L)

for i in range(L, N):
    input_data[i - L] = zscore_temp[i - L : i]
    target_data[i - L] = zscore_temp[i]

test_input_data = np.zeros((test_N - L, L))
test_target_data = np.zeros(test_N - L)

for i in range(L, test_N):
    test_input_data[i - L] = test_zscore_temp[i - L : i]
    test_target_data[i - L] = test_zscore_temp[i]

# 划分训练集和验证集
train_size = int((N - L) * 0.8)
train_input_data = input_data[:train_size]
train_target_data = target_data[:train_size]

val_input_data = input_data[train_size:]
val_target_data = target_data[train_size:]

# SVR建模
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(train_input_data)
X_val = scaler_X.transform(val_input_data)
X_test = scaler_X.transform(test_input_data)

y_train = scaler_y.fit_transform(train_target_data.reshape(-1, 1)).flatten()
y_val = scaler_y.transform(val_target_data.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(test_target_data.reshape(-1, 1)).flatten()

svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

# 预测
y_train_pred = svr.predict(X_train).reshape(-1, 1)
y_val_pred = svr.predict(X_val).reshape(-1, 1)
y_test_pred = svr.predict(X_test).reshape(-1, 1)

y_test = y_test * np.std(test_temp_rec) + np.mean(test_temp_rec)
y_test_pred = y_test_pred * np.std(test_temp_rec) + np.mean(test_temp_rec)
# 反标准化预测结果
y_train_pred = scaler_y.inverse_transform(y_train_pred).flatten()
y_val_pred = scaler_y.inverse_transform(y_val_pred).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred).flatten()
y_train = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_val = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 计算性能指标
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Train RMSE: {train_rmse}, Train R²: {train_r2}')
print(f'Val RMSE: {val_rmse}, Val R²: {val_r2}')
print(f'Test RMSE: {test_rmse}, Test R²: {test_r2}')

# 绘图
plt.figure(figsize=(14, 6))
plt.plot(range(len(y_test)), y_test, label='Test data', color='blue')
plt.plot(range(len(y_test)), y_test_pred, label='SVR Predicted (Test)', color='green')
plt.xlabel('Sample')
plt.ylabel('Temperature (Celsius)')
plt.legend()
plt.grid(True)
plt.show()

index = range(0, 2000)
select_ori = y_test[0:2000]
select_rec = y_test_pred[0:2000]
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
