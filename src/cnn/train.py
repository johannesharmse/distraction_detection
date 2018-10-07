# # data augmenting / preprocessing
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# # from tensorflow.python.client import device_lib

# # model architecture and training
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Convolution2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense


# # finding files
# import os

# # array stuff
# import numpy as np

# def graph():
#     model = Sequential()
#     model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
#                             name='image_array', input_shape=(64, 64, 3)))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(.5))

#     model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(.5))

#     model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(.5))

#     model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(.5))

#     model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=2, kernel_size=(3, 3), padding='same'))
#     model.add(GlobalAveragePooling2D())
#     model.add(Activation('softmax',name='predictions'))

#     return model


# def data_prep(train_dir, validate_dir, batch_size=16):
#     # number of images fed into CNN at a time
#     # not too big - will take forever to train
#     # not too small - model will struggle to get a good idea of 
#     # the classes in general
#     # batch_size = 16

#     # augment settings for training data
#     # this is the augmentation configuration we will use for training
#     train_datagen = ImageDataGenerator(
#             rescale=1./255, #RGB colours (change values to 0 to 1)
#             brightness_range=(0.1, 0.9), 
#             # shear_range=0.2, #tilt random images (20% of images)
#             # zoom_range=0.2, # zoom random iamges (20% of images)
#             horizontal_flip=True
#             ) # flip images

#     # augment settings for validation data
#     # this is the augmentation configuration we will use for validation:
#     # only rescaling
#     validate_datagen = ImageDataGenerator(rescale=1./255) #RGB colours (change values to 0 to 1)

#     # this is a generator that will read pictures found in
#     # subfolers of 'data/train', and indefinitely generate
#     # batches of augmented image data
#     train_generator = train_datagen.flow_from_directory(
#             train_dir,  # this is the target directory
#             target_size=(64, 64),  # all images will be resized to 164x164 (same as input shape in architecture)
#             batch_size=batch_size,
#             class_mode='categorical')  # bug with binary - https://github.com/keras-team/keras/issues/6499- since we use binary_crossentropy loss, we need binary labels

#     # this is a similar generator, for validation data
#     validation_generator = validate_datagen.flow_from_directory(
#             validate_dir,
#             target_size=(64, 64),
#             batch_size=batch_size,
#             class_mode='categorical') # bug with 'binary' - https://github.com/keras-team/keras/issues/6499

#     return train_generator, validation_generator


# if __name__ == "__main__":
#     batch_size = 16
#     # print(device_lib.list_local_devices())
#     model = graph()

#      # model training configuration (https://keras.io/models/model/)
#     # loss function - binary crossentropy - ideal for 2 classes # bug - https://github.com/keras-team/keras/issues/6499
#     # optimizer - rmsprop (http://ruder.io/optimizing-gradient-descent/index.html#rmsprop) - for gradient descent (finding best weights)
#     # metrics - accuracy (proportion of correrctly classified images over number of total images)

#     model.compile(optimizer='adam', loss='categorical_crossentropy',
#                 metrics=['accuracy'])
#     model.summary()

#     # model.compile(loss='categorical_crossentropy',
#     #             optimizer='Adam', # rmsprop
#     #             metrics=['accuracy'])


#     train_generator, validation_generator = data_prep(train_dir='../../data/train', validate_dir='../../data/validate', batch_size=batch_size)
#     model.fit_generator(
#         train_generator,
#         steps_per_epoch=300, # total number of images processed is batch_size*steps_per_epoch*epochs
#         epochs=1,
#         validation_data=validation_generator,
#         validation_steps=60)
#     # save model
#     model.save('distraction_model.hdf5')




'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = '../../data/train'
validation_data_dir = '../../data/validate'
nb_train_samples = 422
nb_validation_samples = 90
epochs = 2
batch_size = 12

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# model.save_weights('first_try.h5')
model.save('distraction_model.hdf5')