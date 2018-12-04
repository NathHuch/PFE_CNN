# Plot ad hoc mnist instances
import os
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
from keras      import backend as K
import matplotlib.pyplot as plt
import timeit





# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

# reshape to be [samples][channels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

input_shape = (1,28,28)
# Create model
model = Sequential()
model.add(Conv2D(28,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(112,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax',kernel_initializer='normal'))

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

