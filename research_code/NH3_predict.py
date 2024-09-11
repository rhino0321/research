import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, Reshape, LSTM, Concatenate, AdditiveAttention
from keras.models import load_model, Sequential
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
file_path = 'C:\\Users\\USER\\Desktop\\dihuadata.csv'
data = pd.read_csv(file_path)
param_columns = data.iloc[:, 0:4].astype(float)

dataSize = data.shape[0]
print(dataSize)
temp = param_columns.iloc[0: dataSize, 0].values
ph = param_columns.iloc[0 : dataSize, 1].values
#ec = param_columns.iloc[0 : dataSize, 2].values
#nitrate = param_columns.iloc[0 : dataSize, 6].values
#DO = param_columns.iloc[0 : dataSize, 3].values
Ammonia = param_columns.iloc[0 : dataSize, 3].values
#NH3 = zscore(NH3)
data_heatmap = {
    'temp': temp,
    'ph': ph,
    'Ammonia': Ammonia
}
df = pd.DataFrame(data_heatmap)

# 绘制热图
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='crest', fmt=".2f")
plt.show()

index = range(0, dataSize)
plt.subplot(4,1,1)
plt.plot(index, temp)
plt.xlabel('temp')
plt.grid(True)

plt.subplot(4,1,2)
plt.plot(index, ph)
plt.xlabel('ph')
plt.grid(True)
'''
plt.subplot(4,1,3)
plt.plot(index, nitrate)
plt.xlabel('nitrate')
plt.grid(True)
'''
plt.subplot(4,1,4)
plt.plot(index, Ammonia)
plt.xlabel('Ammonia')
plt.grid(True)

plt.show()
'''
batch_size = 8
epochs = 100

input_temp = Input(shape = (1, ), name = "input_temp")
input_ph = Input(shape = (1, ), name = "input_ph")

combined_lstm_input = Concatenate(axis = 1)([input_ph, input_temp])
combined_lstm_input_reshaped = Reshape((1, 2))(combined_lstm_input)

lstm_layer = LSTM(units = 512, input_shape=(1, 2), return_sequences = False, dropout=0.2)(combined_lstm_input_reshaped)
predict_val = Dense(1)(lstm_layer)

NH3_predict_model = Model(inputs = [input_temp, input_ph], outputs = [predict_val], name = "NH3_predict_model")
NH3_predict_model.summary()

#loss finction
def custom_loss():
    def loss(y_true, y_pred):
    
        prediction_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true[0], y_pred[0]))

        return prediction_loss
    return loss

#compile model
NH3_predict_loss = custom_loss()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.002)
NH3_predict_model.compile(optimizer = optimizer, loss = NH3_predict_loss)
NH3_predict_model.summary()
'''
train_size = int(dataSize * 0.4)
sc = StandardScaler()

train_temp_data = temp[0 : 0 + train_size]
train_ph_data = ph[0 : 0 + train_size]
#train_nitrate_data = nitrate[0 : 0 + train_size]
train_NH3_data = Ammonia[0 : 0 + train_size]

val_temp_data = temp[train_size : int(dataSize * 0.6)]
val_ph_data = ph[train_size : int(dataSize * 0.6)]
#val_nitrate_data = nitrate[train_size : int(dataSize * 0.8)]
val_NH3_data = Ammonia[train_size : int(dataSize * 0.6)]

test_temp_data = temp[int(dataSize * 0.6) : ]
test_ph_data = ph[int(dataSize * 0.6) : ]
#test_nitrate_data = nitrate[int(dataSize * 0.8) : ]
test_NH3_data = Ammonia[int(dataSize * 0.6) : ]



X_train = np.stack((train_ph_data, train_temp_data, train_NH3_data), axis=-1)
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, 3))
y_train = train_NH3_data

X_val = np.stack((val_ph_data, val_temp_data, val_NH3_data), axis=-1)
X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 1, 3))
y_val = val_NH3_data

X_test = np.stack((test_ph_data, test_temp_data, test_NH3_data), axis=-1)
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, 3))
y_test = test_NH3_data


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

NH3_predict_model = Sequential()
NH3_predict_model.add(LSTM(units=128, return_sequences=False, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), dropout = 0.1))
NH3_predict_model.add(Dense(1))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
NH3_predict_model.compile(optimizer = optimizer, loss = 'mean_squared_error')
#data split

#model fit
history = NH3_predict_model.fit(
    X_train_reshaped, 
    y_train, 
    epochs = 200, 
    batch_size = 32,
    validation_data=(X_val_reshaped, y_val)
)

#plot loss function
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

#testing data
test_predictions = NH3_predict_model.predict(X_test_reshaped)
NH3_predict_r2 = r2_score(y_test, test_predictions)
NH3_predict_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print("NH3 predict rmse : ", NH3_predict_rmse)
print("NH3 predict r2 : ", NH3_predict_r2)

index = range(0, 95)
select_rec = test_predictions[0:95]
select_ori = y_test[0:95]
plt.subplot(2,1,1)
plt.plot(index, select_ori, color='blue')
plt.xlabel('original_data')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(index, select_rec, color='green') 
plt.xlabel('predict nitrate')
plt.grid(True)
plt.show()

plt.plot(index, select_ori, color='blue', label='testing data')  # 蓝色线
plt.plot(index, select_rec, color='green', label='model predict data')  # 绿色线
plt.xlabel('Sample')
plt.ylabel(r'Ammonia (NH$_3$)')
plt.legend()
plt.grid(True)
plt.show()