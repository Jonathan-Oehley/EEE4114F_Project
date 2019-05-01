import numpy as np
import csv
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, Flatten, Concatenate, BatchNormalization
import h5py
import seqGen
import matplotlib.pyplot as plt

data_path = "../Saved Datasets.nosync/2 - Jonti/"
batch_size = 8
img_width = 640
img_height = 240

retrain_model = False

# Read in teh CSV
dataCSV = pd.read_csv(data_path + "Labels.csv")
#print(dataCSV.head())

# Split for training and testing (and validation?)
train = dataCSV.sample(frac=0.8) #Split the dataset, using random seed for reppeatability
test = dataCSV.drop(train.index)
validation = train.sample(frac=0.1)
train = train.drop(validation.index)

# Set up the image flow from disk
trainSequence = seqGen.mySequenceGenerator(train, data_path, batch_size)
validSequence = seqGen.mySequenceGenerator(validation, data_path, batch_size)
testSequence  = seqGen.mySequenceGenerator(test, data_path, batch_size, mode='test')


# Design the model
leftInput  = Input(shape=(img_height, img_width, 1))
rightInput = Input(shape=(img_height, img_width, 1))

leftConv1  = MaxPooling2D(pool_size=(3,3))(Conv2D(32, (3,3), activation='relu')(leftInput))
rightConv1 = MaxPooling2D(pool_size=(3,3))(Conv2D(32, (3,3), activation='relu')(rightInput))

leftConv2  = MaxPooling2D(pool_size=(3,3))(Conv2D(32, (3,3), activation='relu')(leftConv1))
rightConv2 = MaxPooling2D(pool_size=(3,3))(Conv2D(32, (3,3), activation='relu')(rightConv1))

leftConv3  = MaxPooling2D(pool_size=(3,3))(Conv2D(32, (3,3), activation='relu')(leftConv2))
rightConv3 = MaxPooling2D(pool_size=(3,3))(Conv2D(32, (3,3), activation='relu')(rightConv2))

leftFlat  = Flatten()(leftConv3)
rightFlat = Flatten()(rightConv3)

concat = Concatenate()([leftFlat, rightFlat])
#dense1 = Dense(32, activation='relu')(concat)
output = Dense(18, activation='linear')(concat)

model = Model(inputs=[leftInput, rightInput], outputs=output)
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mean_squared_error'])
print(model.summary())

if (retrain_model):
    # Fit the model using the training data
    print("Fitting the model")
    h = model.fit_generator(trainSequence, epochs=50, verbose=1, validation_data=validSequence, validation_steps=validSequence.__len__(), use_multiprocessing=True, workers=2)
    model.save_weights(data_path + "weights.hdf5")
    plt.plot(h.history['mean_squared_error'])
    plt.show()
else:
    model.load_weights(data_path + "weights.hdf5")

# Evaluate the model
print("Evaluating the model")
print(model.evaluate_generator(testSequence, verbose=1))

# Predicting
# print("Predicting")
# index = np.random.randint(0, batch_size)
# print(testSequence.get_batch_labels(index))
# print(model.predict(testSequence.get_batch_features(index)))