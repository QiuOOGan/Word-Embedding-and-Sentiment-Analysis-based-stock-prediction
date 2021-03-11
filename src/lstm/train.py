import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.metrics import RootMeanSquaredError
import numpy as np
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

dim = data_x.shape

# Create Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (dim[1], dim[2])))  # 2 time step, 3 features
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1)) # 1 output: Price


# Train
model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss='mean_squared_error', metrics=[RootMeanSquaredError()])
model.fit(data_x, data_y, batch_size=2000, epochs=100)

result = model.evaluate(data_x,data_y)
# Predict and get error
# preds = model.predict(data_x)
# rmse = np.sqrt(np.mean(preds-data_y) ** 2)
print(result)

