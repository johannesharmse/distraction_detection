from keras.preprocessing.image import img_to_array
import imutils
import cv2 as cv
from keras.models import load_model
import numpy as np

face_cascade = cv.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')

# video streaming
cv.namedWindow('Watcha Looking At?')
camera = cv.VideoCapture(0)
while True:
    frame = camera.read()[1]

    frame = imutils.resize(frame,width=300)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv.CASCADE_SCALE_IMAGE)
    
    if len(faces) > 0:
        frameClone = frame.copy()

        for (x,y,w,h) in faces:
            cv.rectangle(frameClone,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frameClone[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv.imshow('Watcha Looking At?', frameClone)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv.destroyAllWindows()
