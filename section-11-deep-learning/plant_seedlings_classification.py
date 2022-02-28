import pandas as pd
import numpy as np

from glob import glob
import os

import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings("ignore")

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam 
from keras.metrics import categorical_crossentropy 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing import image
from keras.utils import np_utils

DATA_FOLDER = 'v2-plant-seedlings-dataset'

images_df = []

for class_folder_name in os.listdir(DATA_FOLDER):
    class_folder_path = os.path.join(DATA_FOLDER, class_folder_name)

    for image_path in glob(os.path.join(class_folder_path, "*.png")):
        tmp = pd.DataFrame([image_path, class_folder_name]).T
        images_df.append(tmp)

images_df = pd.concat(images_df, axis=0, ignore_index=True)
images_df.columns = ['image', 'target']

# Check distribution
images_df['target'].value_counts()

# Split between train and test
X_train, X_test, y_train, y_test = train_test_split(images_df['image'], images_df['target'],
test_size=0.2, random_state=0)

# Reset indices
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# One-hot encode target
encoder = LabelEncoder()
encoder.fit(y_train)

train_y = np_utils.to_categorical(encoder.transform(y_train))
test_y = np_utils.to_categorical(encoder.transform(y_test))

# Reshape images
IMAGE_SIZE = 150

def im_resize(df, n):
    im = cv2.imread(df[n])
    im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
    return im

tmp = im_resize(X_train, 7)

# Create dataset for CNN of shape (n1, n2, n3, n4)
# n1: number of observations
# n2 and n3: image width and length
# n4: indicates that it is a color image so 3 planes per image

def create_dataset(df, image_size):
    tmp = np.zeros((len(df), image_size, image_size, 3), dtype='float32')

    for n in range(0, len(df)):
        im = im_resize(df, n)
        tmp[n] = im
    
    print('Dataset Images shape: {} size: {:,}'.format(tmp.shape, tmp.size))

X_train = create_dataset(X_train, IMAGE_SIZE)
X_test = create_dataset(X_test, IMAGE_SIZE)

# CNN
kernel_size = (3,3)
pool_size = (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3

model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu',
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(second_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(third_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(12, activation = "softmax"))

model.summary()

model.compile(Adam(lr=0.0001), loss='binary_crossentropy',
metrics=['accuracy'])

batch_size = 10
epochs = 8

filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1,
save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=1,
verbose=1, mode='max', min_lr=0.00001)

# Make a prediction
predictions = model.predict_class(X_test, verbose=1)
cnf_matrix = confusion_matrix(encoder.transform(y_test), predictions)

# Create a dict to map back the numbers onto the classes
tmp = pd.concat([y_test, pd.Series(encoder.transform(y_test))], axis=1)
tmp.columns = ['class_name', 'class_number']
tmp.drop_duplicates(inplace=True)
tmp.reset_index(drop=True, inplace=True)
tmp = pd.Series(tmp.class_name.values,index=tmp.class_number).to_dict()

# Accuracy on test
accuracy_score(encoder.transform(y_test), predictions, normalize=True, sample_weight=None)





