import numpy as np
from flask import Flask, request, jsonify, send_file
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import tensorflow as tf
import pandas as pd
from scipy.stats import zscore
from functions import semi_soft_thresholding, wavelet_decomposition, wavelet_reconstruction
import math

app = Flask(__name__)
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
noise = np.random.normal(0, 1, size=N)
zscore_temp = zscore(temp)
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

temp_coeffs , ph_coeffs, ec_coeffs = wavelet_decomposition(zscore_temp, zscore_ph, zscore_ec, 1)

temp_coeffs, ph_coeffs, ec_coeffs = semi_soft_thresholding(temp_coeffs, ph_coeffs, ec_coeffs)

temp_rec, ph_rec, ec_rec= wavelet_reconstruction(temp_coeffs, temp, ph_coeffs, ph, ec_coeffs, ec)

zscore_temp_rec = zscore(temp_rec)

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


vae_lstm = tf.keras.models.load_model(
    './model/my_model_timedistributes_9019.h5',
    custom_objects={'loss': wrapped_custom_loss}
)
decoded_data, test_predictions = vae_lstm.predict([test_refluctuation_matrix, test_input_data])

def load_model_and_data():
    
    vae_lstm = tf.keras.models.load_model(
    './model/my_model_timedistributes_9019.h5',
    custom_objects={'loss': wrapped_custom_loss}
    )
    decoded_data, test_predictions = vae_lstm.predict([test_refluctuation_matrix, test_input_data])
    return test_predictions

def generate_plot(test_predictions):
    plt.plot(test_predictions)
    plt.xlabel('Index')
    plt.ylabel('Prediction Value')
    plt.title('Top 1000 Predictions')
    plt.grid(True)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer

@app.route('/predict_and_plot')
def predict_and_plot():
    # 加载模型和数据
    test_predictions = load_model_and_data()
    # 取前1000个预测值
    top_1000_predictions = test_predictions[:1000]
    # 生成图像
    plot_buffer = generate_plot(top_1000_predictions)
    # 将图像保存为文件
    # 这里可以保存为文件，也可以直接返回图像数据
    # 为了方便，我们在内存中生成图像并返回图像数据
    img = Image.open(plot_buffer)
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
