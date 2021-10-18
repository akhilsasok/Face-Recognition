import cv2
import numpy as np
import os
from PIL import Image

path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def getImagesandLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSample = []
    ids = []

    for imagePath in imagePaths:
        PIL_image = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_image, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSample.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return faceSample, ids


print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesandLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml')

print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
