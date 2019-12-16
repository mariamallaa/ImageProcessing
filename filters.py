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




######################################################################################################
######################################################################################################
######################################################################################################


def apply(img,filtername):
    windowsize=[3,3]
    orgin=[1,1]
    Structure = disk(2)

    #creating adaboost trained model using cv2
    face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,yt,w,h) in faces:

    
        #crop the face
        cropped_img = img[yt:yt+h,x:x+w ,:]
        #cropped_img2=img[yt-80:yt+h,x:x+w ,:]

        hatflag=0
        if yt>90:
            cropped_img2=img[yt-80:yt+h,x:x+w ,:]
            hatflag=1
        else :
            cropped_img2=cropped_img.copy()
    
        #resize image

        resized_image=resize(cropped_img,(200,180))
        resized_image2=resize(cropped_img2,(200,180))
        
        #finding the eyemap
        
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
        if nosex<img.shape[0] and nosey<img.shape[1]:
            new_croppedimg[nosex,nosey]=1
        if mouthx<img.shape[0] and mouthy<img.shape[1]:
            new_croppedimg[mouthx,mouthy]=1
        
        if filtername=="eye":
            img[yt:yt+h,x:x+w ,:]=sunglassesfilter(resized_image,midpointx,midpointy,h,w,degree)
        elif filtername=="nose":
            img[yt:yt+h,x:x+w,:]=clown_nose_filter(resized_image,nosex,nosey,h,w,degree)
        elif filtername=="mouth":
            img[yt:yt+h,x:x+w,:]=mouth_filter(resized_image,mouthx,mouthy,h,w,degree)
        elif filtername=="hat":
            if hatflag==1:
                img[yt-80:yt+h,x:x+w,:]=hatfilter(resized_image2,h,w)


        show_images([img])
        cv2.imshow('img',img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        