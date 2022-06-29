import csv
import cv2
import numpy as np
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

rows = []
with open('data/track1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        rows.append(line)

images = []
measurements = []
for row in rows:
    path = row[0]
    image = ndimage.imread(path)
    images.append(image)
    measurement = float(row[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20)

model.save('model.h5')
