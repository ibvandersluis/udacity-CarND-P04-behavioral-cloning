import csv
import cv2
import numpy as np
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Conv2D, MaxPooling2D

def loadData(correction=0.2):
    rows = []
    with open('data/track1/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            rows.append(line)

    images = []
    measurements = []
    for row in rows:
        for index in range(3):
            path = row[index]
            image = ndimage.imread(path)
            images.append(image)
            measurement = float(row[3])
            if index == 1:
                measurement += correction
            elif index == 2:
                measurement -= correction
            measurements.append(measurement)
    
    return images, measurements

def augmentData(images, measurements):
    augmented_images, augmented_measurements = [], []
    for image,measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(-measurement)
    return np.array(augmented_images), np.array(augmented_measurements)
    
def createPreprocessingLayers():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((60,25),(0,0))))
    return model

def LeNet():
    model = createPreprocessingLayers()
    model.add(Conv2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def Nvidia():
    model = createPreprocessingLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def trainModel(model, X_train, y_train, epochs=5):
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    model.save('model.h5')

if __name__ == "__main__":
    X_train, y_train = augmentData(loadData())
    model = Nvidia()
    trainModel(model, X_train, y_train, epochs=5)
