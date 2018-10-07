from keras.preprocessing.image import img_to_array
import imutils
import cv2 as cv
from keras.models import load_model
import numpy as np

face_cascade = cv.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')
distract_model = load_model('cnn/distraction_model.hdf5', compile=False)

# frame
frame_w = 1200
border_w = 2
min_size_w = 240
min_size_h = 240
min_size_w_eye = 60
min_size_h_eye = 60
scale_factor = 1.1
min_neighbours = 5

prediction_classes = ['distract', 'focus']

# video streaming
cv.namedWindow('Watcha Looking At?')
camera = cv.VideoCapture(0)
while True:
    frame = camera.read()[1]

    frame = imutils.resize(frame,width=frame_w)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frameClone = frame.copy()

    faces = face_cascade.detectMultiScale(gray,scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w,min_size_h),flags=cv.CASCADE_SCALE_IMAGE)
    
    if len(faces) > 0:
        frameClone = frame.copy()

        for (x,y,w,h) in faces:
            cv.rectangle(frameClone,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frameClone[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w_eye,min_size_w_eye))

            probs = list()
            label = list()

            for (ex,ey,ew,eh) in eyes:
                cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),border_w)
                roi = roi_color[ey+border_w:ey+eh-border_w, ex+border_w:ex+ew-border_w]

                roi = cv.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # print(roi)

                prediction = distract_model.predict(roi)
                # print(prediction)
                probs.append(prediction[0])
                # label.append(prediction_classes[prediction.argmax()])

            probs_mean = np.mean(probs)

            if probs_mean <= 0.5:
                label = 'distracted'
            else:
                label = 'focussed'

            cv.putText(frameClone,label,(x,y-5), cv.FONT_HERSHEY_SIMPLEX, 
            1, (0,0,255), 3, cv.LINE_AA)
            # print(prediction)
            # print(label)

    cv.imshow('Watcha Looking At?', frameClone)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv.destroyAllWindows()
