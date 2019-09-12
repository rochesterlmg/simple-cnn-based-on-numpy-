import numpy as np
from  mnist_lu import load_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.optimizers import SGD

x_train, y_train = load_mnist("./", kind="train")
x_test, y_test = load_mnist("./", kind="t10k")

train_images = x_train[:4000].reshape(-1,28,28)
train_labels = y_train[:4000]
test_images = x_test[:1000].reshape(-1,28,28)
test_labels = y_test[:1000]

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

model = Sequential([
  Conv2D(8, 3, input_shape=(28, 28, 1), use_bias=False),
  MaxPooling2D(pool_size=2),
  Flatten(),
  Dense(10, activation='softmax'),
])

model.compile(SGD(lr=.005), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
  train_images,
  to_categorical(train_labels),
  batch_size=1,
  epochs=3,
  validation_data=(test_images, to_categorical(test_labels)),
)

'''
Epoch 1
46s 760us/step - loss: 0.2433 - acc: 0.9276 - val_loss: 0.1176 - val_acc: 0.9634
Epoch 2
46s 771us/step - loss: 0.1184 - acc: 0.9648 - val_loss: 0.0936 - val_acc: 0.9721
Epoch 3
48s 797us/step - loss: 0.0930 - acc: 0.9721 - val_loss: 0.0778 - val_acc: 0.9744
'''
