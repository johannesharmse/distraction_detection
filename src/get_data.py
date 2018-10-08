from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# models
# face and eyes are templates from opencv
face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')

# frame params
frame_w = 1200
border_w = 2
min_size_w = 240
min_size_h = 240
min_size_w_eye = 60
min_size_h_eye = 60
scale_factor = 1.1
min_neighbours = 5


# image iterators
# i = image filename number
# j = controls how often images should be saved
i = 0
j = 0

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
        frame = imutils.resize(frame,width=frame_w)

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
                # get colour face for saving colour eye images for CNN (probs not necessary)
                roi_color = frame[y:y+h, x:x+w]
                # detect gray eyes
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w_eye,min_size_w_eye))

                # loop through detected eyes
                for (ex,ey,ew,eh) in eyes:
                    # draw eye rectangles
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),border_w)
                    # keep track of eyes detected
                    j += 1
                    # write every second detected eye to file (should probably make 
                    # this an odd number, to prevent only one eye being captured)
                    if j%2 == 0:
                        # create new filename
                        i += 1
                        # specify save location
                        filename = '../data/eye'+str(i)+'.jpg'
                        # print(filename)

                        # write image to file
                        cv2.imwrite(filename, roi_color[ey+border_w:ey+eh-border_w, ex+border_w:ex+ew-border_w])

        # show frame in window
        cv2.imshow('Watcha Looking At?', frame)

        # quit with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# close
camera.release()
cv2.destroyAllWindows()
