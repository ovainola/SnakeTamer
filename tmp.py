from keras.models import load_model, Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense
from keras.optimizers import RMSprop

import numpy as np

x_train = np.random.randn(100, 10, 10, 2)
y_train = np.zeros((100, 2))
y_train[:, np.argmax(np.median(x_train, axis=(1, 2)), axis=1)] = 1.

x_val = np.random.randn(30, 10, 10, 2)
y_val = np.zeros((30, 2))
y_val[:, np.argmax(np.median(x_val, axis=(1, 2)), axis=1)] = 1.

model = Sequential()
model.add(Conv2D(16, (4, 4), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))
optimizer = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10)
print(model.evaluate(x_val, y_val))
model.save('test.h5', overwrite=True)

model = load_model('test.h5')
print(model.evaluate(x_val, y_val))
