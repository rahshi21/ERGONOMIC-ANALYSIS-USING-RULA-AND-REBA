import cv2
import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

import time
import pyautogui as pgi
from tkinter import *  
from tkinter import messagebox  
from tkinter import filedialog

mimetypes.init()
root=Tk()
variable1=StringVar()    
variable2=StringVar()    

root.geometry("1000x800")

root.title("Ergonomic Analysis using RULA and REBA")
root.configure(bg='#F1EB90')

# Update the title label with internal padding
l1 = Label(root, text="Ergonomic Analysis using RULA and REBA", font=('Helvetica', 25, 'bold'), fg='blue', bg='#F1EB90')
l1.pack(pady=100)  # Use pady for internal vertical padding

from angle_calc import angle_calc
import os
import mimetypes

def image_pose_estimation(name):
    img = cv2.imread(name)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    pose1=[]
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            x_y_z=[]
            h, w,c = img.shape
            x_y_z.append(lm.x)
            x_y_z.append(lm.y)
            x_y_z.append(lm.z)
            x_y_z.append(lm.visibility)
            pose1.append(x_y_z)
            cx, cy = int(lm.x*w), int(lm.y*h)
            if id%2==0:
                cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
            else:
                cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)
    img = cv2.resize(img, (700, 700))
    cv2.imshow("Image", img)
    rula,reba=angle_calc(pose1)
    if rula and reba:
        if int(rula)>3:
            pgi.alert("Posture not proper in upper body","Warning")
        elif int(reba)>4:
            pgi.alert("Posture not proper in your body","Warning")
    variable1.set("Rapid Upper Limb Assessment Score : "+rula)
    variable2.set("Rapid Entire Body Score : "+reba)
    root.update()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_pose_estimation(name):
    count=1
    cap = cv2.VideoCapture(name)
    while count:
        frame_no=count*20
        cap.set(1,frame_no);
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        pose1=[]
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                x_y_z=[]
                h, w,c = img.shape
                x_y_z.append(lm.x)
                x_y_z.append(lm.y)
                x_y_z.append(lm.z)
                x_y_z.append(lm.visibility)
                pose1.append(x_y_z)
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id%2==0:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
                else:
                    cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)
        img = cv2.resize(img, (600, 800))
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        angle_calc(pose1)
        time.sleep(1)
        count+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        rula,reba=angle_calc(pose1)
        print(rula,reba)
        if (rula != "NULL") and (reba != "NULL"):
            if int(rula)>3:
                variable1.set("Rapid Upper Limb Assessment Score : "+rula+"Posture not proper in upper body")
                pgi.alert("Posture not proper in upper body","Warning")
            else:
                variable1.set("Rapid Upper Limb Assessment Score : "+rula)
            if int(reba)>4:
                variable2.set("Rapid Entire Body Score : "+reba+"Posture not proper in your body")
                pgi.alert("Posture not proper in your body","Warning")
            else:
                variable2.set("Rapid Entire Body Score : "+reba)
            root.update()
        else:
            pgi.alert("Posture Incorrect")

def webcam():
   video_pose_estimation(0)

def browsefunc():
   filename =filedialog.askopenfilename()
   mimestart = mimetypes.guess_type(str(filename))[0]

   if mimestart != None:
      mimestart = mimestart.split('/')[0]

   if mimestart == 'video':
      video_pose_estimation(str(filename))
   elif mimestart == 'image':
      image_pose_estimation(str(filename))
   else:
      pass

# Text paragraph about the project
project_text = """A model utilizing Pose Estimation, incorporating RULA and REBA for effective ergonomic analysis."""
project_paragraph = Text(root, wrap=WORD, width=70, height=2, font=('Helvetica', 12), bg='#f0f0f0')
project_paragraph.insert('1.0', project_text)
project_paragraph.place(relx=.5, rely=.6, anchor=N)

# Container frame for buttons with increased width and height
button_frame = Frame(root, bg='#87C4FF')
button_frame.place(relx=.5, rely=.3, anchor=N)

# Set pack_propagate to False to control the size of the container
button_frame.pack_propagate(False)

# Set the desired width and height for the container
button_frame.config(width=500, height=170)

# Webcam button
b2 = Button(button_frame, text="LIVE POSTURE ANALYSIS", font=('Comic Sans MS', 14), command=webcam)
b2.pack(pady=20)

# Browse button
b1 = Button(button_frame, text="SELECT AN IMAGE OR A VIDEO", font=('Comic Sans MS', 14), command=browsefunc)
b1.pack(pady=20)

# Container frame for labels
label_frame = Frame(root, bg='#87C4FF')
label_frame.place(relx=.5, rely=.8, anchor=S)

# Label 1
l2 = Label(label_frame, textvariable=variable1, font=('Century Gothic', 10, 'bold'))
l2.pack(pady=5)

# Label 2
l3 = Label(label_frame, textvariable=variable2, font=('Century Gothic', 10, 'bold'))
l3.pack(pady=5)

root.mainloop()
