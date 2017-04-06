import numpy as np
import cv2
import csv
import pickle

csv_file = 'data/driving_log.csv'

with open(csv_file, 'r') as f:
    car_images = []
    steering_angles = []
    reader = csv.reader(f)

    def process_image(path):
        return cv2.imread('data/IMG/' + path.split('/')[-1])
    i = 0
    for row in reader:
        i += 1
        if i % 1000 == 0:
            print ('Processed {} lines'.format(i))
        steering_center = float(row[3])

        # if i >= 10000:
        #     break
        # create adjusted steering measurements for the side camera images
        correction = 0.4 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras

        img_center = process_image(row[0])
        img_left = process_image(row[1])
        img_right = process_image(row[2])

        # add images and angles to data set
        car_images.extend([img_center, img_left, img_right])
        steering_angles.extend([steering_center, steering_left*1.1, steering_right*1.1])

print("csv file done")

# for i in range(len(car_images)):
#     car_images.append(cv2.flip(car_images[i],1))
#     steering_angles.append(steering_angles[i] * -1.0)

X_train = np.array(car_images)
y_train = np.array(steering_angles)

del car_images
del steering_angles


print("start pickle")
# Save the data for easy access
pickle_file = 'data/camera.pickle'
with open(pickle_file, 'w+b') as pfile:
    pickle.dump({
                    'train_dataset': np.array(X_train),
                    'train_labels': np.array(y_train),
                }, pfile)
