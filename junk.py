from comet_ml import Experiment
experiment = Experiment('OKhPlin1BVQJFzniHu1f3K1t3')

import keras
import os
from skimage import io
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Hyperparameters
img_width, img_height = 128, 128
num_classes = 19
# train_samples = 1000
# val_samples = 100
epochs = 50
batch_size = 16

ts =  datetime.datetime.now().timestamp()
train_data = './data/train/'
val_data = './data/val/'
test_data = './data/test/'
saved_model = './saved_models/saved_model-' + str(ts) + '.h5'
saved_weights = './saved_weights/saved_weight-' + str(ts) + '.h5'

# save the entire model with weights
save_model = True
# save the weights separately
save_weights = True

load_weights = False
load_weight_path = './saved_weights/saved_weight-1576127088.591156.h5'

###############################################################

# create a data generator
datagen = ImageDataGenerator(rescale=1./255, zoom_range=0, validation_split=0.3)

train_gen = datagen.flow_from_directory(train_data, class_mode='categorical',
            batch_size=batch_size, target_size=(img_width, img_height), subset='training')
# load and iterate validation dataset
val_gen = datagen.flow_from_directory(val_data, class_mode='categorical',
            batch_size=batch_size, target_size=(img_width, img_height), subset='validation')
# load and iterate test dataset
test_gen = datagen.flow_from_directory(test_data, class_mode='categorical',
            batch_size=batch_size, target_size=(img_width, img_height))

x_train, y_train = train_gen.next()
x_test, y_test = test_gen.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (x_train.shape, x_train.min(), x_train.max()))

# set up default input shape parameters from keras default
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

################################################################
model = Sequential()


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# CIFAR 10 CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
            input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)


if load_weights:
    model.load_weights(load_weight_path)

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
# STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

# fit model
model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=16,
        validation_data=val_gen, validation_steps=8, shuffle=True, verbose=2)

# test_error_rate = model.evaluate(X)

if save_weights:
    model.save_weights(saved_weights)
    print('Saved trained weights as %s ' % saved_weights)

if save_model:
    model.save(saved_model)
    print('Saved trained model as %s ' % saved_model)
