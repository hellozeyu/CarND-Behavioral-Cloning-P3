from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, GaussianNoise, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import plot_model
import numpy as np
import cv2
import csv


csvpath = 'data/driving_log.csv'
samples = []
with open(csvpath) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=1314)

# preprocess the images to what is in Nvidia's paper and crop the top 1/3.
def load_image(imagepath):
    imagepath = 'data/IMG/' + imagepath.split('/')[-1]
    image = cv2.imread(imagepath)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

def generator(samples, batch_size=32):
    correction = 0.5
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            steering_angles = []

            for batch_sample in batch_samples:
                batch_images = []
                batch_angles = []
                img_center = load_image(batch_sample[0])
                img_left = load_image(batch_sample[1])
                img_right = load_image(batch_sample[2])

                steering_center = float(batch_sample[3])
                steering_left = (steering_center + correction)
                steering_right = (steering_center - correction)

                batch_images.extend([img_center, img_left, img_right])
                batch_angles.extend([steering_center, steering_left, steering_right])
                # data augmentation
                # flip images left and right
                batch_images.extend(list(map(lambda x: cv2.flip(x,1), batch_images)))
                batch_angles.extend(list(map(lambda x: -x, batch_angles)))

                car_images.extend(batch_images)
                steering_angles.extend(batch_angles)

            X_train = np.array(car_images)
            y_train = np.array(steering_angles)

            yield X_train, y_train

batch_size = 32
nb_epoch = 10
nb_classes = 1
samples_per_epoch = len(train_samples)/batch_size
nb_val_samples = int(samples_per_epoch/10.0)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

def nvidia_net():
    model = Sequential()
    # Dropout probablity
    # Didn't use dropout at the beginning and it gives me a pretty good model.
    p=0.5

    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Conv2D(24, (5, 5), name="conv1", strides=(2, 2), padding="valid", kernel_initializer="normal", activation="relu"))
    model.add(Conv2D(36, (5, 5), name="conv2", strides=(2, 2), padding="valid", kernel_initializer="normal", activation="relu"))
    model.add(Conv2D(48, (5, 5), name="conv3", strides=(2, 2), padding="valid", kernel_initializer="normal", activation="relu"))
    model.add(Conv2D(64, (3, 3), name="conv4", strides=(1, 1), padding="valid", kernel_initializer="normal", activation="relu"))
    model.add(Conv2D(64, (3, 3), name="conv5", strides=(1, 1), padding="valid", kernel_initializer="normal", activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164, name = "dense_1"))
    model.add(Dense(100, name = "dense_2"))
    model.add(Dense(50, name = "dense_3"))
    model.add(Dense(10, name = "dense_4"))
    model.add(Dense(1, name = "dense_5"))

    return model


model = nvidia_net()
model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])

# Print out summary of the model
model.summary()
plot_model(model, to_file='model.png', show_shapes=True)

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

# use fit generator to train.
history = model.fit_generator(train_generator, nb_epoch=nb_epoch,
                    validation_data=validation_generator,
                    samples_per_epoch=samples_per_epoch,
                    nb_val_samples=nb_val_samples,
                    verbose=1)

# save model
model.save('model.h5')
