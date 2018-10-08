from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# models
# face and eyes are templates from opencv
# distract model is a TF CNN model trained using Keras (see /src/cnn/train.py) 
face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')
distract_model = load_model('cnn/distraction_model.hdf5', compile=False)

# frame params
frame_w = 1200
border_w = 2
min_size_w = 240
min_size_h = 240
min_size_w_eye = 60
min_size_h_eye = 60
scale_factor = 1.1
min_neighbours = 5

# Video writer
# IMPORTANT:
# - frame width and height must match output frame shape
# - avi works on ubuntu. mp4 doesn't :/
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video_out = cv2.VideoWriter('video_out.avi', fourcc, 10.0,(1200, 900))

# init camera window
cv2.namedWindow('Watcha Looking At?')
camera = cv2.VideoCapture(0)

# Check if camera opened successfully
if (camera.isOpened() == False): 
    print("Unable to read camera feed")

while True:
    # get frame
    ret, frame = camera.read()

    # if we have a frame, do stuff
    if ret:

        # make frame bigger
        frame = imutils.resize(frame, width=frame_w)

        # use grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect face(s)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w,min_size_h),flags=cv2.CASCADE_SCALE_IMAGE)
        
        # for each face, detect eyes and distraction
        if len(faces) > 0:
            # loop through faces
            for (x,y,w,h) in faces:
                # draw face rectangle
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                # get gray face for eye detection
                roi_gray = gray[y:y+h, x:x+w]
                # get colour face for distraction detection (model has 3 input channels - probably redundant)
                roi_color = frame[y:y+h, x:x+w]
                # detect gray eyes
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w_eye,min_size_w_eye))

                # init probability list for each eye prediction
                probs = list()

                # loop through detected eyes
                for (ex,ey,ew,eh) in eyes:
                    # draw eye rectangles
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),border_w)
                    # get colour eye for distraction detection
                    roi = roi_color[ey+border_w:ey+eh-border_w, ex+border_w:ex+ew-border_w]
                    # match CNN input shape
                    roi = cv2.resize(roi, (64, 64))
                    # normalize (as done in model training)
                    roi = roi.astype("float") / 255.0
                    # change to array
                    roi = img_to_array(roi)
                    # correct shape
                    roi = np.expand_dims(roi, axis=0)

                    # distraction classification/detection
                    prediction = distract_model.predict(roi)
                    # save eye result
                    probs.append(prediction[0])

                # get average score for all eyes
                probs_mean = np.mean(probs)

                # get label
                if probs_mean <= 0.5:
                    label = 'distracted'
                else:
                    label = 'focused'
                
                # insert label on frame
                cv2.putText(frame,label,(x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255), 3, cv2.LINE_AA)
        
        # Write the frame to video
        video_out.write(frame)

        # display frame in window
        cv2.imshow('Watcha Looking At?', frame)

        # quit with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # no frame, don't do stuff
    else:
        break

# close
camera.release()
video_out.release()
cv2.destroyAllWindows()