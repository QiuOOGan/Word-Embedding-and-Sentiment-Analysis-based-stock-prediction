from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras.callbacks import LambdaCallback
from preprocessing import *
import numpy as np
from tensorflow.python.client import device_lib
from keras.models import save_model
print(device_lib.list_local_devices())
# Hpyerparameter Optimization:
# round 1: lr=0.0033,num_Nodes=12,dropout=1,loss='mean_Squared_error'
# round 2: lr=0.0066,num_Nodes=18, dropout=0.3,final_activation='sigmoid', loss='categorical_crossentropy'
def create_network(lr=1.0, num_Nodes=2, dropout=0, final_activation='sigmoid',
                   loss_function='mean_squared_error', n=n):

    model = Sequential()
    model.add(Dense(num_Nodes, input_dim=n, activation='sigmoid', kernel_initializer='random_uniform'))
    model.add(Dropout(dropout))
    model.add(Dense(num_Nodes, activation='sigmoid', kernel_initializer='random_uniform'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation=final_activation, kernel_initializer='random_uniform'))
    model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss=loss_function,
                  metrics=['accuracy'])
    return model

epochs = 100

# Build an ANN with 20% validation set using create_network. Record the validation loss.
# wrapper = KerasClassifier(build_fn=create_network, epochs=epochs)
final_model = create_network()
train_scores = []
val_scores = []
train_loss = LambdaCallback(on_epoch_end=lambda batch, logs: train_scores.append(logs['loss']))
val_loss = LambdaCallback(on_epoch_end=lambda batch, logs: val_scores.append(logs['val_loss']))
earlystopper = EarlyStopping(monitor='val_loss', patience=epochs/10)
final_model.fit(x_train,y_train,epochs=epochs, validation_split=0.2, batch_size=1000, verbose=1,
                callbacks=[train_loss, val_loss])

#retrain for all training data and save the model
test_scores = []
final_model = create_network()
test_loss = LambdaCallback(on_epoch_end=lambda batch, logs: test_scores.append(final_model.evaluate(x_test, y_test)[0]))
final_model.fit(x_train, y_train, epochs=epochs, batch_size=1000, verbose=0,callbacks=[test_loss])
print("testing accuracy:",final_model.evaluate(x_test, y_test)[1])
#save_model(final_model, 'model509b.h5')


# Plotting loss curve of training loss, validation loss and test loss
plt.figure()
plt.title("Learning Curve")
plt.grid()

# plt.fill_between(np.linspace(0,len(train_scores),len(train_scores)), train_scores,
#                   alpha=0.1, color="r")
# plt.fill_between(np.linspace(0,len(val_scores),len(val_scores)), val_scores,
#              alpha=0.1, color="g")
# plt.fill_between(np.linspace(0,len(test_scores),len(test_scores)), test_scores,
#              alpha=0.1, color="b")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.ylim(top=max(train_scores),bottom=min(train_scores))
plt.plot(np.linspace(0,len(train_scores),len(train_scores)), test_scores, linewidth=1, color="r",
         label="Training score")
plt.plot(np.linspace(0,len(val_scores),len(val_scores)), val_scores, linewidth=1, color="g",
          label="validation score")
plt.plot(np.linspace(0,len(test_scores),len(test_scores)), train_scores, linewidth=1, color="b",
          label="test score")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
legend.get_frame().set_facecolor('C0')

plt.show()