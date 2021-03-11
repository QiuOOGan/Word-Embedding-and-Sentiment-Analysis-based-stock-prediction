import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.metrics import RootMeanSquaredError
import numpy as np
from keras.callbacks import LambdaCallback
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.optimizers import Adam
data_x = np.array([
    # Datapoint 1
    [
        # Input features at timestep 1, we have 30 timesteps
        [1, 2, 3], # our features contain price at this time step, and senntiment(s) at this time step
        # Input features at timestep 2
        [4, 5, 6]
    ],
    # Datapoint 2
    [
        # Features at timestep 1
        [7, 8, 9],
        # Features at timestep 2
        [10, 11, 12]
    ]
])

# The desired model outputs.
# We will create two data points, just for the example.
data_y =  np.array([7, 13])
method_name = 'mood'
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
train_loss = LambdaCallback(on_epoch_end=lambda batch, logs: train_scores.append(logs['loss']))
earlystopper = EarlyStopping(monitor='loss', patience=epochs/10)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[RootMeanSquaredError()])
model.fit(train_x, train_y, batch_size=2000, epochs=epochs, callbacks=[train_loss, earlystopper])

result = model.evaluate(test_x,test_y)[1]

print(result)

plt.figure()
plt.title("RMSE: " + str(result))
plt.grid()
plt.suptitle(method_name + " Learning Curve")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.ylim(top=max(train_scores),bottom=min(train_scores))
plt.plot(np.linspace(0,len(train_scores),len(train_scores)), train_scores, linewidth=1, color="r",
         label="Training score")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
legend.get_frame().set_facecolor('C0')

plt.show()

