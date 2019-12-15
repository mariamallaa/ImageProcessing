from commonfunctions import *
from appliedfunctions import *
from skimage.filters import gaussian 
from skimage.color import rgb2ycbcr
from skimage.transform import resize
import numpy as np
import math
from skimage.morphology import binary_erosion, binary_dilation
from skimage import exposure
from skimage.morphology import erosion, dilation
from skimage.morphology import disk
import cv2

######################################################################################################
######################################################################################################
######################################################################################################
windowsize=[3,3]
orgin=[1,1]
Structure = disk(2)



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
    '''
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    img =frame
    '''
    #reading an exisiting image

    img =  io.imread("C:\\Users\\xps\\Desktop\\hh\\ImageProcessing\\IMG_5403.jpg").astype('uint8')
    show_images([img])


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,yt,w,h) in faces:

        #print("x,yt")
        #crop the face
        cropped_img = img[yt:yt+h,x:x+w ,:]
        cropped_img2=img[yt-80:yt+h,x:x+w ,:]
    
        #resize image

        resized_image=resize(cropped_img,(200,180))
        resized_image2=resize(cropped_img2,(200,180))
        #show_images([resized_image2])
        #finding the eyemap
        '''
        cropped_img3=img[yt-20:yt+h+20,x-20:x+w+20 ,:]
        img_hair=np.copy(resized_image3)
        img_hair=hair_colour(resized_image3)
        show_images([ resized_image3,img_hair])
        '''
        eye_location= np.array([])
       
        factor=0.7
        while eye_location.size==0:
            eyemap=geteyemap(resized_image)
            
            thresh =np.max(eyemap)*factor
            
            eyemap[eyemap<thresh]=0
            eyemap[eyemap>thresh]=1
           
            #show_images([resized_image,eyemap])
                
            #show_images([resized_image,eyemap])
            eye_location=distance(eyemap, 125,90)
            factor-=0.1

        #show_images([resized_image,eyemap])

        righti = np.mean(eye_location[:,0])
        rightj = np.mean(eye_location[:,1])
        lefti = np.mean(eye_location[:,2])
        leftj = np.mean(eye_location[:,3])
        eyearr=np.asarray([righti,rightj,lefti,leftj])
        
        img_eyes=resized_image[int(eyearr[2]-10):int(eyearr[0]+10),int(eyearr[1]-20):int(eyearr[3]+20)]
        #show_images([img_eyes])
        degree=0
       
        if(righti>lefti and righti-lefti>10):
            degree=20
        elif(righti<lefti and lefti-righti>10):
            degree=-20

        eyearr=eyearr.astype('uint16')
        
        new_croppedimg=np.zeros(eyemap.shape)

        new_croppedimg[eyearr[0],eyearr[1]]=1

        new_croppedimg[eyearr[2],eyearr[3]]=1
        
        
        midpointx= int((eyearr[0]+eyearr[2])/2)
        midpointy= int((eyearr[1]+eyearr[3])/2)
        #nose
        image_nose=resized_image[int(eyearr[0]):int(eyearr[0]*2),int(eyearr[1]):int(eyearr[3])]
        #show_images([image_nose])
        nosex,nosey= getnose(midpointx,midpointy,righti,lefti)
        mouthx,mouthy=getmouth(midpointx,midpointy,righti,lefti)
        new_croppedimg[nosex,nosey]=1
        new_croppedimg[mouthx,mouthy]=1
        
        #show_images([new_croppedimg])
        img[yt:yt+h,x:x+w ,:]=sunglassesfilter(resized_image,midpointx,midpointy,h,w,degree)
        #show_images([img])
        img[yt:yt+h,x:x+w,:]=clown_nose_filter(resized_image,nosex,nosey,h,w,degree)
        #show_images([img])
        #img[yt:yt+h,x:x+w,:]=mouth_filter(resized_image,mouthx,mouthy,h,w,degree)

        #img[yt-80:yt+h,x:x+w,:]=hatfilter(resized_image2,h,w)
        #cv2.imshow("test", img)

        #show_images([img])


    show_images([img])
    cv2.imshow('img',img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    