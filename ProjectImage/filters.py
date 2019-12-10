from commonfunctions import *
from appliedfunctions import *
from skimage.filters import gaussian 
from skimage.color import rgb2ycbcr
from skimage.transform import resize
import numpy as np
import math
from skimage.morphology import binary_erosion, binary_dilation
from skimage import exposure
import cv2

######################################################################################################
######################################################################################################
######################################################################################################
windowsize=[3,3]
orgin=[1,1]


######################################################################################################
######################################################################################################
######################################################################################################


cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

#creating adaboost trained model using cv2
face_cascade = cv2.CascadeClassifier('C:\\Users\\xps\\Downloads\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')

#val = input("press 1 for camera image press 2 for choosing an image from your computer: ")

while True:
    #reading frame from camera
    #ret, frame = cam.read()
    #cv2.imshow("test", frame)
    #img =frame

    #reading an exisiting image

    img =  io.imread("C:\\Users\\xps\\Desktop\\hh\\ImageProcessing\\ProjectImage\\friends1.jpg").astype('uint8')
    show_images([img])


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,yt,w,h) in faces:

        
        #crop the face
        cropped_img = img[yt:yt+h,x:x+w ,:]

        #removing noise

        filtered_img = gaussian(cropped_img, sigma=0.1,multichannel=False)

        #resize image

        resized_image=resize(cropped_img,(200,180))

        
        #finding the eyemap
        
        

        
        arrnp= np.array([])

        factor=0.7
        while arrnp.size==0:
            eyemap=geteyemap(resized_image)
            
            thresh =np.max(eyemap)*factor
            print(eyemap)
            print(thresh)
            print(np.max(eyemap))
            print("the ")
            eyemap[eyemap<thresh]=0
            eyemap[eyemap>thresh]=1
            print(eyemap)

            show_images([resized_image,eyemap])


            eyemap= erode(eyemap,windowsize,orgin)
            eyemap= dilate(eyemap,windowsize,orgin)
                
            eyemap= erodesmall(eyemap,windowsize,orgin)
                

            arrnp=distance(eyemap, 125,90)
            factor-=0.1

        #print(arrnp)
        #print(arrnp.count)
        show_images([resized_image,eyemap])

        righti = np.mean(arrnp[:,0])
        rightj = np.mean(arrnp[:,1])
        lefti = np.mean(arrnp[:,2])
        leftj = np.mean(arrnp[:,3])
        eyearr=np.asarray([righti,rightj,lefti,leftj])
        degree=0
        if(righti>lefti and righti-lefti>10):
            degree=20
        elif(righti<lefti and lefti-righti>10):
            degree=-20

        print("mirna")
        print(eyearr)
        eyearr=eyearr.astype('uint16')
        
        new_croppedimg=np.zeros(eyemap.shape)

        new_croppedimg[eyearr[0],eyearr[1]]=1

        new_croppedimg[eyearr[2],eyearr[3]]=1
        
        show_images([new_croppedimg])

        distanceofeye = math.sqrt((eyearr[0]-eyearr[2])**2+(eyearr[3]-eyearr[1])**2)
        midpointx= int((eyearr[0]+eyearr[2])/2)
        midpointy= int((eyearr[1]+eyearr[3])/2)
        #nose
        nosex = int((distanceofeye+midpointx) *0.78)
        nosey = midpointy

        new_croppedimg[nosex,nosey]=1
       
        #img[yt:yt+h,x:x+w ,:]=sunglassesfilter(resized_image,midpointx,midpointy,h,w,degree)
        img[yt:yt+h,x:x+w ,:]=clown_nose_filter(resized_image,nosex,nosey,h,w,degree)

        #cv2.imshow("test", img)

        show_images([img])



    #cv2.imshow('img',img)
   
    print(img.shape)
    print(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()