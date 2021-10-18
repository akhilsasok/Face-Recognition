import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n Enter user id end press <return> ==>  ')
print('\n Initializing face capture. Look the camera and wait...')
name = input('\nEnter your name:')

count = 0

while True:
    successful_frame_read, frame = cam.read()
    gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img, 1.3, 4)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1
        cv2.imwrite("dataset/" + name + "." + str(face_id) + "." + str(count) + ".jpg",
                    gray_scaled_img[y:y + h, x:x + w])
        cv2.imshow('Face Read', frame)
    key = cv2.waitKey(1)
    if count >= 20:
        break
    elif key == 27:
        break

print("Exiting program")
cam.release()
cv2.destroyAllWindows()
