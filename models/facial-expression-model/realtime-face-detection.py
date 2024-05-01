import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

face_model = tf.keras.models.load_model("saved-model/5.pmodel-fer-150e.h5")
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

path = "saved-model/haarcascade_frontalface_default.xml"
font_scale = 1.5

face_roi = []
final_image = []


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open camera")

while True:

    ret, frame = cap.read()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )  # this model detects for faces
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB
    )
    # faces = face_cascade.detectMultiScale(frame_rgb, 1.1, 4)
    faces = face_cascade.detectMultiScale(frame_gray, 1.1, 4)

    for x, y, w, h in faces:
        # frame_rgb_cropface = frame_rgb[y : y + h, x : x + w]
        frame_gray_cropface = frame_gray[y : y + h, x : x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # faces_2 = face_cascade.detectMultiScale(frame_rgb_cropface)
        faces_2 = face_cascade.detectMultiScale(frame_gray_cropface)
        if len(faces_2) == 0:
            print("face not detected")
            break

        for x2, y2, w2, h2 in faces_2:
            # face_roi = frame_rgb_cropface[y2 : y2 + h2, x2 : x2 + w2]
            face_roi = frame_gray_cropface[y2 : y2 + h2, x2 : x2 + w2]
            # second face crop

    if not len(face_roi) == 0:
        final_image = np.expand_dims(cv2.resize(np.array(face_roi), (48, 48)), 0) / 255
        # resize the face and normalize
        predictions = face_model.predict(final_image, verbose=0)
        # print(labels[np.argmax(predictions)])
        # print(predictions)
        cv2.putText(
            frame,
            labels[np.argmax(predictions)],
            (x, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_4,
        )
        cv2.putText(
            frame,
            "confidence: "
            + str(round(predictions[0][np.argmax(predictions)] * 100, 2)),
            (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_4,
        )

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(2)
    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
