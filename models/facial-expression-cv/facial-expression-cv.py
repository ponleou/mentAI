import cv2
import tensorflow as tf
import os
import numpy as np
import matplotlib

# execute inside the folder
datasets_dir = "../../datasets/"

labels = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
fe_dataset_dir = os.path.join(datasets_dir, "facial-expression-dataset")

# img = cv2.imread(os.path)

# there should be 2400 different images for each lable/category
max_data_per_label = 2400
img_size = 224

fe_dataset = []

for label in labels:
    label_dir = os.path.join(fe_dataset_dir, label)
    label_index = labels.index(label)  # the label in numbers for the images
    i = 0
    for img_file in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_file)
        img_array = cv2.imread(img_path)
        img_resize_array = cv2.resize(img_array, (img_size, img_size))
        fe_dataset.append([img_resize_array, label_index])
        i += 1
        if i == max_data_per_label:
            break

# print(len(fe_dataset))
# print(fe_dataset[23][0].shape) # (224, 224, 3) cus rgb

# print(np.array(fe_dataset))
# print(fe_dataset.shape)

feature_ds = []
label_ds = []
for feature, label in fe_dataset:
    feature_ds.append(feature)
    label_ds.append(label)

# print(np.array(fe_dataset))
# print(np.array(fe_dataset))

print(np.array(feature_ds))
print(np.array(label_ds))
print(len(feature_ds), len(label_ds))
