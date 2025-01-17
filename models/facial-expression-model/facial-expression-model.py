import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

# execute inside the folder
datasets_dir = "datasets/"

# labels = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# fe_dataset_dir = os.path.join(datasets_dir, "facial-expression-dataset")

labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
fe_dataset_dir = os.path.join(datasets_dir, "FER2013")

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

# overfitted, train acc: 0.899 loss 0.29, test acc: 0.778 loss 0.909
# pretrained_model = tf.keras.applications.MobileNetV2(
#     input_shape=(img_size, img_size, 3),
# )

# pretrained_model_output = pretrained_model.layers[-2].output
# pretrained_model_output = layers.Dense(128, activation="relu")(pretrained_model_output)
# pretrained_model_output = layers.Dense(64, activation="relu")(pretrained_model_output)
# prediction_output = layers.Dense(len(labels), activation="softmax")(
#     pretrained_model_output
# )

# model = tf.keras.Model(
#     inputs=pretrained_model.layers[0].input, outputs=prediction_output
# )

# this model needs more epochs >25, acc: 68 loss 0.86
model = tf.keras.Sequential(
    [
        layers.Input(shape=(48, 48, 3)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPool2D(2),
        layers.Dropout(0.5),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPool2D(2),
        layers.Dropout(0.5),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPool2D(2),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(labels), activation="softmax"),
    ]
)

print(model.summary())

epochs = 150

# initial_learning_rate = 0.00005
# final_learning_rate = 0.00000005
# learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (
#     1 / epochs
# )
# steps_per_epoch = int(train_size)

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=steps_per_epoch,
#     decay_rate=learning_rate_decay_factor,
#     staircase=True,
# )

model.compile(
    loss="sparse_categorical_crossentropy",
    # optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
    # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"],
)

history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

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


train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plot_graph(train_acc, val_acc, "Accuracy")

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
plot_graph(train_loss, val_loss, "Loss")

model.save("model.keras")

# model = tf.keras.models.load_model("model.h5")
