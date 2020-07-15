import cv2              
import os

def check_path(path):            
    dir = os.path.dirname(path)  
    if not os.path.exists(dir):
        os.makedirs(dir)

vid_cam = cv2.VideoCapture(0) 

face_cascade = cv2.CascadeClassifier('C:\\Users\\Riyansika\\Desktop\\Deep-learning\\Computer-vision\\Lock-Unlock-Laptop-PC-Screen-Using-Face-Recognition\\haarcascade_frontalface_default.xml')

#face_id = 2 
face_id = input('\n enter user id end press <return> ==>  ')

count = 0   

check_path("dataset/")

while(True):
    _,image_frame = vid_cam.read()       # Capture video frame _, is used to ignored first value because vid_cam.read() is returning 2 values
    
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY) # Convert frame to grayscale
    
    faces = face_cascade.detectMultiScale(gray, 1.4, 5)# Detect faces using Cascade Classifier(xml file)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2) # Crop the image frame into rectangle
        
        count += 1               # Increment face image

        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w]) # Save the captured image into the datasets folder

        cv2.imshow('Creating Dataset!!!', image_frame)  # Display the video frame, with rectangular box on the person's face

    if cv2.waitKey(100) & 0xFF == 27:                   # To stop taking video, press 'Esc'
        break

    elif count>100:                                     # If image taken reach 100, stop taking video
        break

vid_cam.release()                                       # Stop video

cv2.destroyAllWindows()                                 # Close all windows

