from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')
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

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video_out = cv2.VideoWriter('video_out.avi', fourcc, 10.0,(1200, 900))

# video streaming
cv2.namedWindow('Watcha Looking At?')
camera = cv2.VideoCapture(0)

# Check if camera opened successfully
if (camera.isOpened() == False): 
    print("Unable to read camera feed")

while True:
    ret, frame = camera.read()

    if ret:
        frame = imutils.resize(frame,width=frame_w)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frameClone = frame.copy()

        faces = face_cascade.detectMultiScale(gray,scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w,min_size_h),flags=cv2.CASCADE_SCALE_IMAGE)
        
        if len(faces) > 0:
            frameClone = frame.copy()

            for (x,y,w,h) in faces:
                cv2.rectangle(frameClone,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frameClone[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w_eye,min_size_w_eye))

                probs = list()
                label = list()

                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),border_w)
                    roi = roi_color[ey+border_w:ey+eh-border_w, ex+border_w:ex+ew-border_w]

                    roi = cv2.resize(roi, (64, 64))
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
                    label = 'focused'

                cv2.putText(frameClone,label,(x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255), 3, cv2.LINE_AA)
                # print(prediction)
                # print(label)
        
        # Write the frame into the file 'output.avi'
        video_out.write(frameClone)

        print(frameClone.shape)

        cv2.imshow('Watcha Looking At?', frameClone)

        

        # quit with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

camera.release()
video_out.release()
cv2.destroyAllWindows()
