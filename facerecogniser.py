import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

path = 'dataset'

font = cv2.FONT_HERSHEY_SIMPLEX


def getImagesandLabels(path, i):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    for imagePath in imagePaths:
        if i == int(os.path.split(imagePath)[-1].split(".")[1]):
            return os.path.split(imagePath)[-1].split(".")[0]

cam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = cam.read()
    gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = detector.detectMultiScale(gray_scaled_img, 1.2, 5)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray_scaled_img[y:y + h, x:x + w])

        if confidence < 100:
            name = str(getImagesandLabels(path, id))
            confidence = "{0}%".format(round(confidence))
        else:
            name = "Unknown"
            confidence = "{0}%".format(round(confidence))
        cv2.putText(frame, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

cam.release()
cv2.destroyAllWindows()
