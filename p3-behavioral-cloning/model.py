import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D

def get_sample_data(path, ignore_first_line=True):
    """
    This function returns the data columns of the csv file at PATH
    """
    samples = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        first_line = ignore_first_line
        for line in reader:
            if first_line:
                first_line = False
                continue
            samples.append(line)
    return samples

def analyze_samples(data):
    """
    Return a dictionary containing count of all steering angles in DATA
    """
    steering = dict()
    data = np.array(data)
    angles = data[:, 3]
    for i in angles:
        i = float(i)
        if i not in steering:
            steering[i] = 1
        else:
            steering[i] += 1
    return steering

def redistribute_samples(data, count_map):
    """
    Return a new dataset by resampling DATA
    """
    average = np.average(list(count_map.values()))
    keep_prob = dict()
    for i in count_map:
        if count_map[i] <= average:
            keep_prob[i] = 1 # keep angles whose count <= than average
        else:
            # keep sample with a probability
            keep_prob[i] = average / count_map[i]
    # Resample Data
    new_data = list()
    for line in data:
        angle = float(line[3])
        if np.random.random() <= keep_prob[angle]:
            new_data.append(line)
    return np.array(new_data)

def adjust_brightness_RGB(img):
    """
    Adjust brightness of the IMG, whose color is in RGB
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ratio = 0.9 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def process_image(img, kernel_size=5):
    """
    Preprocess IMG with a Gaussian Blur and convert it from BGR color channel
    to RGB color channel
    """
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def random_translate(image, steering_angle, range_x=100, range_y=10):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def random_shadow(image):
    """
    Generates and adds random shadow
    """
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = image.shape
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)
    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

# Generator that yield data a batch size at a time (to save memoery) 
def generator(samples, directory, batch_size=32, side_image_correction=0.25, preprocess_image=lambda x: x, augment=True):

    batch_size = batch_size // 6

    def read_img(path):
        return preprocess_image(cv2.imread(directory + path.split('/')[-1]))

    def flip(img):
        return np.fliplr(img)

    def append_data(lst1, item1, lst2, item2):
        lst1.append(item1)
        lst2.append(item2)

    def use_augmentation(prob):
        return True if np.random.random() <= prob else False

    def augment(new_img, new_angle):
        augmented = False
        if augment and use_augmentation(0.4):
            new_img, new_angle = adjust_brightness_RGB(new_img), new_angle
            augmented = True
        if augment and use_augmentation(0.2):
            new_img, new_angle = random_shadow(new_img), new_angle
            augmented = True
        if augment and use_augmentation(0.3):
            new_img, new_angle = random_translate(new_img, new_angle)
            augmented = True
        if not augmented or use_augmentation(0.4):
            new_img, new_angle = flip(new_img), -new_angle

        return new_img, new_angle

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Get Data
                center, left, right, angle, _, _, _ = batch_sample
                # Convert Images
                center = read_img(center)
                left = read_img(left)
                right = read_img(right)
                # Convert Angles
                center_angle = float(angle)
                left_angle = center_angle + side_image_correction
                right_angle = center_angle - side_image_correction

                # Add Center Image and Augmentation
                append_data(images, center, angles, center_angle)
                new_img, new_angle = augment(center, center_angle)
                append_data(images, new_img, angles, new_angle)

                # Add Left Image and Augmentation
                append_data(images, left, angles, left_angle)
                new_img, new_angle = augment(left, left_angle)
                append_data(images, new_img, angles, new_angle)

                # Add Right Image and Augmentation
                append_data(images, right, angles, right_angle)
                new_img, new_angle = augment(right, right_angle)
                append_data(images, new_img, angles, new_angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def build_model(cropping_box=((70,25), (0,0)), input_shape=(160,320,3), normalize_fn=lambda x: x / 255.0 - 0.5):
    model = Sequential()
    model.add(Cropping2D(cropping=cropping_box))
    model.add(Lambda(normalize_fn, input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="elu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="elu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="elu"))
    model.add(Convolution2D(64, 3, 3, activation="elu"))
    model.add(Convolution2D(64, 3, 3, activation="elu"))
    keep_prob = 0.5
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation("elu"))
    model.add(Dropout(keep_prob))
    model.add(Dense(50))
    model.add(Activation("elu"))
    model.add(Dropout(keep_prob))
    model.add(Dense(10))
    model.add(Activation("elu"))
    model.add(Dropout(keep_prob))
    model.add(Dense(1))
    return model

save_path = "model.h5"
image_data_file_path = "data/IMG/"

log_file_path1 = "data/driving_log.csv"
log_file_path2 = "data/driving_log2.csv"
samples = get_sample_data(log_file_path1)
samples.extend(get_sample_data(log_file_path2))

# Split Data to Training and Validation Sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Remove 0-angle bias for samples
train_samples_angle_count_map = analyze_samples(train_samples)
train_samples = redistribute_samples(train_samples, train_samples_angle_count_map)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=180, directory=image_data_file_path, preprocess_image=process_image, side_image_correction=0.25)
validation_generator = generator(validation_samples, batch_size=180, directory=image_data_file_path, preprocess_image=process_image, side_image_correction=0.25, augment=False)

# Uncomment following line if you want to rebuild a new model to train
# model = build_model()

# Uncomment following line if you want to keep fine tuning existing model (model.h5)
model = load_model("model.h5")

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch= len(train_samples) * 5, validation_data=validation_generator, nb_val_samples=len(validation_samples) * 5, nb_epoch=4, verbose=1) 

# Save Model After Training
model.save(save_path)
print("model saved as", save_path)
