import cv2
import os
import pickle
import numpy as np
from tqdm import tqdm
from os import listdir

DATA_PATH = "PlantVillage-Dataset/raw/color"

image_list, label_list = [], []
for folder_path in tqdm(listdir(DATA_PATH)):
    num_images = 0
    for image_path in listdir(os.path.join(DATA_PATH, folder_path)):
        if num_images > 400:
            break
        num_images+=1
        image_list.append(cv2.imread(os.path.join(DATA_PATH, folder_path, image_path)))
        label_list.append(folder_path)

data = {"images": image_list, "labels": label_list}
pickle.dump(data, open("data_array.pkl", "wb"))
