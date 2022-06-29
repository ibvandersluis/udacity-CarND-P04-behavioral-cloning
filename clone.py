import csv
import cv2
import numpy as np
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D

rows = []
with open('data/track1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        rows.append(line)
        
correction = 0.2
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
    
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(-measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
