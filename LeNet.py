from data import Data
from LeNetModel import LeNet
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

data = Data(True)
(X_train, y_train), (X_test, y_test) = data.training(width=24)

num_classes = len(y_train[0])
batch_size = 32
img_rows, img_cols, z = X_train.shape[1:]
epochs = 100

model = LeNet((img_rows, img_cols, z), y_train.shape[1])
print('Initiating LeNet model with', (img_rows, img_cols, z), 'and', y_train.shape[1])

filepath = "CNN_checkpoint/model_and_weights-{epoch:02d}-{val_acc:.4f}.hdf5"
# only save model and weights if validation accuracy is better
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
score = model.evaluate(X_test, y_test, batch_size=batch_size)

print()
print('Test Loss= ', score)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train, Test'], loc=0)
plt.show()
