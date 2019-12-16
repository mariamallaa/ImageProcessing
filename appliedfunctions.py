from commonfunctions import *
from skimage.filters import gaussian 
from skimage.color import rgb2ycbcr
from skimage.color import rgb2yuv
from skimage.transform import resize
import numpy as np
import math
from skimage.morphology import binary_erosion, binary_dilation
from skimage import exposure
import cv2
from skimage.filters import threshold_otsu
from skimage.transform import rotate


######################################################################################################
######################################################################################################
######################################################################################################

def erode(img, window_size, origin_position):
    new_img = np.ones((img.shape[0], img.shape[1]))
    for i in range(origin_position[0], img.shape[0] - window_size[0] + origin_position[0]):
        for j in range(origin_position[1], img.shape[1] - window_size[1] + origin_position[1]):
            x_start = i-origin_position[0]
            y_start = j-origin_position[1]
            mini = 256
            for x in range( x_start , x_start + window_size[0]):
                for y in range( y_start, y_start + window_size[1]):
                    if img[x][y] < mini:
                        mini = img[x][y]
                        
            if(x_start + window_size[0]+1 < img.shape[0] and img[x_start + window_size[0]+1][y_start]<mini):
                mini = img[x_start + window_size[0]+1][y_start]
            if(y_start + window_size[1]+1 < img.shape[1] and img[x_start ][y_start+ window_size[1]+1]<mini):
                mini = img[x_start ][y_start+ window_size[1]+1]
                
            if(x_start + window_size[0]-1 > 0 and img[x_start + window_size[0]-1][y_start]<mini):
                mini = img[x_start + window_size[0]-1][y_start]
            if(y_start + window_size[1]-1 > 0 and img[x_start ][y_start+ window_size[1]-1]<mini):
                mini = img[x_start ][y_start+ window_size[1]-1]
            
            new_img[i,j] = mini 
    return new_img



def erodesmall(img, window_size, origin_position):
    new_img = np.ones((img.shape[0], img.shape[1]))
    for i in range(origin_position[0], img.shape[0] - window_size[0] + origin_position[0]):
        for j in range(origin_position[1], img.shape[1] - window_size[1] + origin_position[1]):
            x_start = i-origin_position[0]
            y_start = j-origin_position[1]
            mini = 256
            for x in range( x_start , x_start + window_size[0]):
                for y in range( y_start, y_start + window_size[1]):
                    if img[x][y] < mini:
                        mini = img[x][y]
            
            new_img[i,j] = mini 
    return new_img



def dilate(img, window_size, origin_position):
    new_img = np.ones((img.shape[0], img.shape[1]))
    for i in range(origin_position[0], img.shape[0] - window_size[0] + origin_position[0]):
        for j in range(origin_position[1], img.shape[1] - window_size[1] + origin_position[1]):
            x_start = i-origin_position[0]
            y_start = j-origin_position[1]
            maxi = -1
            for x in range( x_start , x_start + window_size[0]):
                for y in range( y_start, y_start + window_size[1]):
                    if img[x][y] > maxi:
                        maxi = img[x][y]
            if(x_start + window_size[0]+1 < img.shape[0] and img[x_start + window_size[0]+1][y_start]>maxi):
                maxi = img[x_start + window_size[0]+1][y_start]
            if(y_start + window_size[1]+1 < img.shape[1] and img[x_start ][y_start+ window_size[1]+1]>maxi):
                   maxi = img[x_start ][y_start+ window_size[1]+1]
                
            if(x_start + window_size[0]-1 > 0 and img[x_start + window_size[0]-1][y_start]>maxi):
                maxi = img[x_start + window_size[0]-1][y_start]
            if(y_start + window_size[1]-1 > 0 and img[x_start ][y_start+ window_size[1]-1]>maxi):
                maxi = img[x_start ][y_start+ window_size[1]-1]
            new_img[i,j] = maxi
    return new_img


def distance(img, ic,jc):
    arr=[]
    for i in range(25,ic):
        for j in range(30,jc):
            if(img[i][j]==1):
                dist= math.sqrt((i-ic)**2 + (j-jc)**2)
                for n in range(25,175):
                    for m in range(jc,150):
                        if(img[n][m]==1):
                            dist2= math.sqrt((n-ic)**2 + (m-jc)**2)
                            if(abs(dist-dist2)/dist<30/100):
                                if(abs(i-n)<20):
                                    if(abs(j-m)<80):
                                        arr.append([i,j,n,m])                                
    arrnp =np.asarray(arr)
    
    return arrnp
                    


def sunglassesfilter(img,midpointx,midpointy,h,w,degree):
            sunglass_image = io.imread('sun.jpg')


            resized_sunglass=resize(sunglass_image[0:600,:],(100,150))
            #print(degree)
            if(degree!=0):
                resized_sunglass=rotate(resized_sunglass, degree,cval=1)
            #show_images([resized_sunglass])

            resized_sunglass[resized_sunglass>=0.99607843]=1
            resized_sunglass[resized_sunglass<0.99607843]=0
            
            for i in range(resized_sunglass.shape[0]):
                for j in range(resized_sunglass.shape[1]):
                    if resized_sunglass[i,j,1]==0:
                        if midpointx-50+i<img.shape[0] and midpointy-75+j<img.shape[1]:
                            img[midpointx-50+i,midpointy-75+j]=(resized_sunglass[i,j])
               
            img=resize(img,(h,w))
            
            finalimg=(img*255).astype('uint8')
            return finalimg

def hatfilter(img,h,w):
            hat_image = io.imread('hat2.jpg')


            resized_hat=resize(hat_image[100:500,50:500],(90,140))
            
            show_images([resized_hat])
            '''
            if(degree!=0):
                resized_hat=rotate(resized_hat, degree,cval=1)
            ''' 
            resized_hat2=np.copy(resized_hat)
            resized_hat[resized_hat>=0.7]=1
            resized_hat[resized_hat<0.7]=0
            
            for i in range(resized_hat.shape[0]):
                for j in range(resized_hat.shape[1]):
                    if resized_hat[i,j,1]==0:
                        img[i,j+40]=(resized_hat2[i,j])
            img=resize(img,(h+80,w))
            
            finalimg=(img*255).astype('uint8')
            return finalimg

def clown_nose_filter(img,nosex,nosey,h,w,degree):
        clown_nose= io.imread('clown-nose.jpg')
        resized_clown_nose=resize(clown_nose[0:500,0:500],(70,100))
        if(degree!=0):
                resized_clown_nose=rotate(resized_clown_nose, degree,cval=1)
        resized_clown_nose[resized_clown_nose>=0.7]=1
        resized_clown_nose[resized_clown_nose<0.7]=0

        for i in range(resized_clown_nose.shape[0]):
            for j in range(resized_clown_nose.shape[1]):
                if resized_clown_nose[i,j,1]==0:
                    if nosex-30+i<img.shape[0] and nosey-50+j<img.shape[1]:
                        img[nosex-30+i,nosey-50+j]=(resized_clown_nose[i,j])


        img=resize(img,(h,w))
            
        finalimg=(img*255).astype('uint8')
        return finalimg
    

def mouth_filter(img,mouthx,mouthy,h,w,degree):
        mouth= io.imread('mouth.jpg')
        resized_mouth=resize(mouth[20:470,0:700],(65,70))
        if(degree!=0):
                resized_mouth=rotate(resized_mouth, degree,cval=1)
        resized_mouth2=np.copy(resized_mouth)
        resized_mouth[resized_mouth>=0.7]=1
        resized_mouth[resized_mouth<0.7]=0

        for i in range(resized_mouth.shape[0]):
            for j in range(resized_mouth.shape[1]):
                if resized_mouth[i,j,1]==0:
                    if mouthx-20+i<img.shape[0] and mouthy-35+j<img.shape[1]:
                        img[mouthx-20+i,mouthy-35+j]=(resized_mouth2[i,j])
        img=resize(img,(h,w))
            
        finalimg=(img*255).astype('uint8')
        return finalimg

def face_filter(img,x,y,h,w,degree):
        face= io.imread('dog.jpg')
        resized_face=resize(face[0:400,0:500],(160,150))
        if(degree!=0):
                resized_face=rotate(resized_face, degree,cval=1)
        resized_face2=np.copy(resized_face)
        resized_face[resized_face>=0.7]=1
        resized_face[resized_face<0.7]=0
        
        for i in range(resized_face.shape[0]):
            for j in range(resized_face.shape[1]):
                if resized_face[i,j,1]==0:
                    img[i,j]=(resized_face2[i,j])
        img=resize(img,(h+100,w))
            
        finalimg=(img*255).astype('uint8')
        return finalimg


def hair_colour(img):
        r=img[2,50,0]*255
        g=img[2,50,1]*255
        b=img[2,50,2]*255
        # 3- extract R, G and B channels (as float)
        image=(img*255).astype(float)
        red=image[:,:,0]
        green=image[:,:,1]
        blue=image[:,:,2]
        # 4- calculate differences FOR EACH CHANNEL (between the image and the required pixel value)
        red=np.subtract(red,r)
        green=np.subtract(green,g)
        blue=np.subtract(blue,b)

        red=np.square(red)
        green=np.square(green) 
        blue=np.square(blue)

        distance=np.sqrt(red+green+blue)

        print(distance)
        # 6-  create a mask by thresholding the differences
        mask=distance.copy()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if(mask[i,j]>30):
                    mask[i,j]=1
                else:
                    mask[i,j]=0
        image[:,:,0]=np.multiply(image[:,:,0],mask)
        image[:,:,1]=np.multiply(image[:,:,1],mask)
        image[:,:,2]=np.multiply(image[:,:,2],mask)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if(image[i,j,0]==0):
                    image[i,j,0]=255
                if(image[i,j,1]==0):
                    image[i,j,1]=255
                if(image[i,j,2]==0):
                    image[i,j,2]=0
        finalimg=(image).astype('uint8')
        return finalimg

def geteyemap(img):
        ycbcr_image = rgb2ycbcr(img).astype('uint8')
        y = ycbcr_image[:,:,0]
        cb = ycbcr_image[:,:,1]
        cr = ycbcr_image[:,:,2]

        chromamap= 1/3 *((cb/cr) + cb**2 + (1-cr)**2)

        newchromamap = exposure.equalize_hist(chromamap)

        windowsize=[3,3]
        orgin=[1,1]
        ydilate= dilate(y,windowsize,orgin)
        yerode= erode(y,windowsize,orgin)

        lumamap = ydilate / (yerode+1)


        eyemap =(1-newchromamap)+lumamap
        return eyemap

def getnose(midpointx,midpointy,righti,lefti):
    nosex = int((midpointx) *1.5)
    if(righti>lefti and righti-lefti>10):
        nosey=int(midpointy*1.2)
    elif(righti<lefti and lefti-righti>10):
        nosey=int(midpointy*0.8)
    else:
        nosey = midpointy
    return nosex,nosey

def getmouth(midpointx,midpointy,righti,lefti):
    mouthx = int((midpointx) *1.96)
    if(righti>lefti and righti-lefti>10):
        mouthy=int(midpointy*1.2)
    elif(righti<lefti and lefti-righti>10):
        mouthy=int(midpointy*0.8)
    else:
        mouthy = midpointy
    return mouthx,mouthy


######################################################################################################
######################################################################################################
######################################################################################################
