import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

# execute inside the folder
datasets_dir = "../../datasets/"

labels = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
fe_dataset_dir = os.path.join(datasets_dir, "facial-expression-dataset")

# img = cv2.imread(os.path)

# there should be 2400 different images for each lable/category
max_data_per_label = 500
img_size = 224

fe_dataset = []

for label in labels:
    label_dir = os.path.join(fe_dataset_dir, label)
    label_index = labels.index(label)  # the label in numbers for the images
    i = 0
    for img_file in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_file)
        img_array = cv2.imread(
            img_path
        )  # converting images to multi dimension array with opencv
        img_array_rgb = cv2.cvtColor(
            img_array, cv2.COLOR_BGR2RGB
        )  # changing color from bgr to rgb (default is bgr)
        img_resize_array = cv2.resize(
            img_array_rgb, (img_size, img_size)
        )  # changing the image resolution from 96 to 224, because our pretrain-model inputs (224,224,3) shape
        fe_dataset.append([img_resize_array, label_index])
        i += 1
        if i == max_data_per_label:
            break

feature_ds = []
label_ds = []
for feature, label in fe_dataset:
    feature_ds.append(feature / 255)  # normalizing the data
    label_ds.append(label)

batch_size = 16

raw_ds = (
    tf.data.Dataset.from_tensor_slices((np.array(feature_ds), np.array(label_ds)))
    .batch(batch_size)
    .shuffle(1000)
)

train_size = int(0.64 * len(raw_ds))
val_size = int(0.16 * len(raw_ds))
test_size = int(0.2 * len(raw_ds))

train_ds = raw_ds.take(train_size)
test_ds = raw_ds.skip(train_size)
val_ds = raw_ds.skip(test_size)
test_ds = raw_ds.take(test_size)

pretrained_model = tf.keras.applications.MobileNetV2()

pretrained_model_output = pretrained_model.layers[-2].output
pretrained_model_output = layers.Dense(128, activation="relu")(pretrained_model_output)
pretrained_model_output = layers.Dense(64, activation="relu")(pretrained_model_output)
prediction_output = layers.Dense(8, activation="softmax")(pretrained_model_output)

model = tf.keras.Model(
    inputs=pretrained_model.layers[0].input, outputs=prediction_output
)

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

epochs = 10
model.fit(train_ds, epochs=epochs, validation_data=val_ds)

loss, acc = model.evaluate(test_ds)
print("Loss: ", loss, "Accuracy: ", acc)