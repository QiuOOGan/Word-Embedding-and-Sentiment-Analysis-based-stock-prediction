import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.metrics import RootMeanSquaredError
import numpy as np
from keras.callbacks import LambdaCallback
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.optimizers import Adam


method_name = 'finbert_with_summarize'
with open('./LSTM_data/' + method_name + '_x' + '.npy', 'rb') as f:
    data_x = np.load(f)
with open('./LSTM_data/' + method_name + '_y' + '.npy', 'rb') as f:
    data_y = np.load(f)
testing_split = 0.2
testing_split = int(len(data_x)*testing_split)
train_x = data_x [:-testing_split]
test_x = data_x[-testing_split:]
train_y = data_y [:-testing_split]
test_y = data_y[-testing_split:]
dim = data_x.shape

# Create Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (dim[1], dim[2])))  # 2 time step, 3 features
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1)) # 1 output: Price


# Train
epochs = 100
train_scores = []
test_scores = []

train_loss = LambdaCallback(on_epoch_end=lambda batch, logs: train_scores.append(logs['loss']))
earlystopper = EarlyStopping(monitor='loss', patience=epochs/10)
model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss='mean_squared_error', metrics=[RootMeanSquaredError()])
test_loss = LambdaCallback(on_epoch_end=lambda batch, logs: test_scores.append(model.evaluate(test_x, test_y)[0]))
model.fit(train_x, train_y, batch_size=2000, epochs=epochs, callbacks=[train_loss, test_loss])

result = model.evaluate(test_x,test_y)[1]

print(result)

plt.figure()
plt.title("Testing RMSE: " + str(result))
plt.grid()
plt.suptitle(method_name + " Learning Curve")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.ylim(top=max(train_scores),bottom=min(train_scores))
plt.plot(np.linspace(0,len(train_scores),len(train_scores)), train_scores, linewidth=1, color="r",
         label="Training loss")
plt.plot(np.linspace(0,len(test_scores),len(test_scores)), test_scores, linewidth=1, color="b",
          label="Testing loss")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
legend.get_frame().set_facecolor('C0')

plt.show()

