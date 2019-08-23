import numpy as np
import pickle
import warnings
import cv2
from os import listdir
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)

data = pickle.load(open("data_array.pkl", "rb"))
X = np.array(data["images"])
y = np.array(data["labels"])
n_classes = len(set(data["labels"]))
chanDim = -1
del data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)
pickle.dump(encoder, open("encoder.pkl", "wb"))

del X
del y
chanDim=-1
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same",input_shape=(256, 256, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))


opt = Adam(lr=0.1)
# distribution
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,metrics=["accuracy"])
model.fit(X_train, y_train,
    validation_data=(X_test, y_test),
    epochs = 5
)

model.save("model.h5")

