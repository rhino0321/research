import numpy as np
import matplotlib.pyplot as plt

# 假設有3組模型，每組有10次測量的RMSE和R²值
r2_scores_model1 = [0.9653, 0.9635, 0.9712, 0.9714, 0.9735, 0.968, 0.9616, 0.9703, 0.9701, 0.9563]
r2_scores_model2 = [0.8552, 0.889, 0.9612, 0.8934, 0.9428, 0.9481, 0.9429, 0.9174, 0.9167, 0.8965]
r2_scores_model3 = [0.629, 0.914, 0.9509, 0.9624, 0.8527, 0.9021, 0.9632, 0.947, 0.9352, 0.9339]

rmse_scores_model1 = [0.1209, 0.128, 0.0931, 0.0922, 0.0802, 0.1089, 0.1352, 0.0979, 0.0988, 0.1536]
rmse_scores_model2 = [0.353, 0.3, 0.137, 0.294, 0.1927, 0.1783, 0.1925, 0.25, 0.2513, 0.1355, 0.2888]
rmse_scores_model3 = [0.6086, 0.2557, 0.1702, 0.1323, 0.3566, 0.279, 0.1294, 0.1813, 0.2114, 0.2146]


ds_scores_model1 = [75.32, 75.282, 75.111, 75.13, 75.03, 75.06, 75.29, 75.13, 75.27, 75.13]
ds_scores_model2 = [75.02, 76.81, 76.29, 76.077, 76.83, 77.04, 76.87, 76.93, 76.95, 76.79]
ds_scores_model3 = [76.4, 0.7694, 76.28, 77.43, 76.41, 76.63, 77.02, 76.65, 76.77, 75.58]

# 整理數據
r2_scores = [r2_scores_model1, r2_scores_model2, r2_scores_model3]
rmse_scores = [rmse_scores_model1, rmse_scores_model2, rmse_scores_model3]
ds_scores = [ds_scores_model1, ds_scores_model2, ds_scores_model3]

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.boxplot(r2_scores, vert=True, patch_artist=True, labels=['VAE-LSTM', 'LSTM', 'GRU'])
ax1.set_title('R² Box-Whisker Plot')
ax1.set_ylabel('R² Scores')
plt.show()

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.boxplot(rmse_scores, vert=True, patch_artist=True, labels=['VAE-LSTM', 'LSTM', 'GRU'])
ax2.set_title('RMSE Box-Whisker Plot')
ax2.set_ylabel('RMSE Scores')
plt.show()

temp_scores = [39.191, 39.177, 39.159, 39.137, 39.114, 39.08, 39.064, 39.039, 39.013]
pH_scores = [35.366, 35.365, 35.363, 35.3609, 35.3575, 35.3536, 35.3488, 35.343, 35.336]
ec_scores = [31.878, 31.875, 31.872, 31.869, 31.867, 31.8642, 31.8613, 31.858, 31.855]
index = range(1, 10)

plt.plot(index, temp_scores, 'b-o')
plt.xlabel('Delta Value')
plt.ylabel('Temperature SNR(dB)')
plt.show()
plt.plot(index, pH_scores, 'r-d')
plt.xlabel('Delta Value')
plt.ylabel('pH SNR(dB)')
plt.show()
plt.plot(index, ec_scores, 'g-^')
plt.xlabel('Delta Value')
plt.ylabel('Electrival Conductivity SNR(dB)')
plt.show()

VAE_LSTM_scores = [0.0930, 0.1361, 0.1354, 0.1566, 0.2159]
GRU_scores = [0.1500, 0.1991, 0.2257, 0.2590, 0.2745]
LSTM_scores = [0.1218, 0.2003, 0.1995, 0.2196, 0.2437]
index = [64, 128, 256, 512, 1024]

plt.plot(range(len(index)), VAE_LSTM_scores, 'b-o', label = 'VAE-LSTM')
plt.plot(range(len(index)), GRU_scores, 'r-d', label = 'GRU')
plt.plot(range(len(index)), LSTM_scores, 'g-^', label = 'LSTM')
plt.xlabel('Batch Size')
plt.ylabel('RMSE')
plt.xticks(ticks=range(len(index)), labels = index)
plt.legend()
plt.show()

