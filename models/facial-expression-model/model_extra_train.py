import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

# execute inside the folder
datasets_dir = "datasets/"


labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
fe_dataset_dir = os.path.join(datasets_dir, "facial-expression-dataset2")

# img = cv2.imread(os.path)

# there should be 2400 different images for each lable/category
max_data_per_label = 100000
img_size = 48

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
        img_resize_array = cv2.resize(img_array_rgb, (img_size, img_size))
        # changing the image resolution from 96 to 224, because our pretrain-model inputs (224,224,3) shape
        fe_dataset.append([img_resize_array, label_index])
        i += 1
        if i == max_data_per_label:
            break

print(len(fe_dataset))

feature_ds = []
label_ds = []
for feature, label in fe_dataset:
    feature_ds.append(feature / 255)  # normalizing the data
    label_ds.append(label)

batch_size = 32

raw_ds = (
    tf.data.Dataset.from_tensor_slices((np.array(feature_ds), np.array(label_ds)))
    .shuffle(len(fe_dataset))
    .batch(batch_size)
)


train_size = int(0.8 * len(raw_ds))
test_size = int(0.2 * len(raw_ds))

train_ds = raw_ds.take(train_size)
test_ds = raw_ds.skip(train_size)

val_size = int(0.2 * len(train_ds))

val_ds = train_ds.take(val_size)
train_ds = train_ds.skip(val_size)


model = tf.keras.models.load_model("saved-model/5.pmodel-fer-150e.h5")
# model.pop()

# model.add(layers.Dense(64, activation="relu", name="added"))
# model.add(layers.Dense(len(labels), activation="softmax", name="output"))

print(model.summary())

epochs = 25

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    metrics=["accuracy"],
)

# history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

loss, acc = model.evaluate(test_ds)
print("Loss: ", loss, "Accuracy: ", acc)


def plot_graph(train, val, title):
    epochs = range(1, len(train) + 1)
    plt.plot(epochs, train, "b", label="Training " + title)
    plt.plot(epochs, val, "y", label="Validation " + title)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(title)
    plt.legend(loc="upper right")
    plt.show()


# train_acc = history.history["accuracy"]
# val_acc = history.history["val_accuracy"]
# plot_graph(train_acc, val_acc, "Accuracy")

# train_loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# plot_graph(train_loss, val_loss, "Loss")

# model.save("model.keras")
