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
                                arr.append([i,j,n,m])
                                
    arrnp =np.asarray(arr)
    
    return arrnp
                    


def sunglassesfilter(img,midpointx,midpointy,h,w):
            sunglass_image = io.imread('sun.jpg')


            resized_sunglass=resize(sunglass_image[0:600,:],(100,150))
            

            resized_sunglass[resized_sunglass>=0.99607843]=1
            resized_sunglass[resized_sunglass<0.99607843]=0
            
            for i in range(resized_sunglass.shape[0]):
                for j in range(resized_sunglass.shape[1]):
                    if resized_sunglass[i,j,1]==0:
                        img[midpointx-50+i,midpointy-75+j]=(resized_sunglass[i,j])
               
            img=resize(img,(h,w))
            
            finalimg=(img*255).astype('uint8')
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

######################################################################################################
######################################################################################################
######################################################################################################
