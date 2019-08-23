import numpy as np
import pickle
import cv2
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder


class Model:
    def __init__(self, model_path = "model.h5", encoder_path = "encoder.pkl"):
        self.model = load_model(model_path)
        self.encoder = pickle.load(open(encoder_path, "rb"))

    def predcit(self, image_path = ""):
        image = cv2.imread(image_path)
        image_resize = np.array(cv2.resize(image, (256, 256)))
        image_resize = np.reshape(image_resize, (1, 256, 256, 3))
        prediciton = self.model.predict(image_resize)
        return self.encoder.inverse_transform(prediciton)


