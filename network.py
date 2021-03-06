from data import Data
import numpy as np
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
import timeit
import numpy as np

#### Cette partie est réservée au chargement des données une première fois ####
# data = Data()
# training_set = data.training()
#
# (X_train, Y_train), (X_test, Y_test) = data.training()
#
# np.array(X_train).dump(open('X_train.npy','wb'))
# np.array(X_test).dump(open('X_test.npy','wb'))
# np.array(Y_train).dump(open('Y_train.npy','wb'))
# np.array(Y_test).dump(open('Y_test.npy','wb'))

X_train  = np.load(open('X_train.npy','rb'))
Y_train  = np.load(open('Y_train.npy','rb'))
X_test   = np.load(open('X_test.npy','rb'))
Y_test   = np.load(open('Y_test.npy','rb'))


# one hot encode outputs
y_train = np_utils.to_categorical(Y_train)
y_test  = np_utils.to_categorical(Y_test)

num_classes = y_test.shape[1]
plt.imshow(X_train[1,:,:,0])
X_train     = X_train.reshape(904,512,50,1)
input_shape =(512,50,1)
# Create model
model = Sequential()
model.add(Conv2D(512,kernel_size=(3,3),activation='relu',input_shape= input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4,activation='softmax',kernel_initializer='normal'))

#compile model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# Fit the model
NAME = "CNN_Dense_100"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME), histogram_freq=1)
start_time = timeit.default_timer()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4, batch_size=200,callbacks=[tensorboard])
print("\n Duree entrainement : %.2f s" % (timeit.default_timer() - start_time))
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, batch_size=200,verbose=2)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#print("CNN Error: \%.2f\%\%",\%(100-scores[1]*100))



# Plot ad hoc mnist instances

