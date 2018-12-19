import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint
from IPython.display import display, Image

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 256
N_CLASSES = 3036
LR = 0.001
N_EPOCHS = 50
IMG_SIZE = 96

train_generator = train_datagen.flow_from_directory(
        './data/training_data',  # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to 150x150
        batch_size=BATCH_SIZE,
        class_mode='sparse')

validation_generator = test_datagen.flow_from_directory(
        './data/testing_data',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse')

model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=True, classes=N_CLASSES, weights=None)
model.summary()

model_file = "models_4/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=576840 // BATCH_SIZE,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=30360 // BATCH_SIZE,
        callbacks=callbacks_list)