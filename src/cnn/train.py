# data augmenting / preprocessing
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# model architecture and training
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


# finding files
import os

# array stuff
import numpy as np

def graph():
    # fully connected model (opposed to model with different input / branches)
    model = Sequential()

    # input and first hidden layer
    # - 2D images (Conv2D - https://keras.io/layers/convolutional/#conv2d)
    # - 32 filters applied to input images (meaning 32 outputs per image)
    # - filter/kernel has size (3x3)
    # - strides is by default (1,1), meaning the filter moves one pixel at a time (both directions)
    # - padding is by default 'valid', meaning if the image doesn't meet input shape, padding will be added
    model.add(Conv2D(32, (3, 3), input_shape=(164, 164, 3)))
    # activation function (ensures non-linearity)
    model.add(Activation('relu'))
    # max pooling reduces dimensionality
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # second hidden layer
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # third hidden layer
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # the model so far outputs 3D feature maps (height, width, features)
    # now we need to flatten it to allow computation
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    # a normal 1D hidden layer with less neurons (not necessary)
    model.add(Dense(64))
    model.add(Activation('relu'))

    # dropout removes influence of neurons that don't add value
    model.add(Dropout(0.5))

    # output layer (2 neurons) (bug with binary - now matching num of classes - https://github.com/keras-team/keras/issues/6499)
    # this will give us our 'probability' value
    model.add(Dense(2))

    # change to a value between 0 and 1
    model.add(Activation('softmax')) # binary bug (binary use 'sigmoid')

    # model training configuration (https://keras.io/models/model/)
    # loss function - binary crossentropy - ideal for 2 classes # bug - https://github.com/keras-team/keras/issues/6499
    # optimizer - rmsprop (http://ruder.io/optimizing-gradient-descent/index.html#rmsprop) - for gradient descent (finding best weights)
    # metrics - accuracy (proportion of correrctly classified images over number of total images)
    model.compile(loss='categorical_crossentropy',
                optimizer='Adam', # rmsprop
                metrics=['accuracy'])

    return model


def data_prep(train_dir, validate_dir, batch_size=16):
    # number of images fed into CNN at a time
    # not too big - will take forever to train
    # not too small - model will struggle to get a good idea of 
    # the classes in general
    # batch_size = 16

    # augment settings for training data
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255, #RGB colours (change values to 0 to 1)
            shear_range=0.2, #tilt random images (20% of images)
            zoom_range=0.2, # zoom random iamges (20% of images)
            horizontal_flip=True) # flip images

    # augment settings for validation data
    # this is the augmentation configuration we will use for validation:
    # only rescaling
    validate_datagen = ImageDataGenerator(rescale=1./255) #RGB colours (change values to 0 to 1)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            train_dir,  # this is the target directory
            target_size=(164, 164),  # all images will be resized to 164x164 (same as input shape in architecture)
            batch_size=batch_size,
            class_mode='categorical')  # bug with binary - https://github.com/keras-team/keras/issues/6499- since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = validate_datagen.flow_from_directory(
            validate_dir,
            target_size=(164, 164),
            batch_size=batch_size,
            class_mode='categorical') # bug with 'binary' - https://github.com/keras-team/keras/issues/6499

    return [train_generator, validation_generator]


if __name__ == "__main__":
    batch_size = 16
    model = graph()
    train_generator, validation_generator = data_prep(train_dir='../../data/train', validate_dir='../../data/validate', batch_size=batch_size)
    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size, # total number of images processed is batch_size*steps_per_epoch*epochs
        epochs=4,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
