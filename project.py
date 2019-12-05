
from commonfunctions import *
from skimage.filters import gaussian 
from skimage.color import rgb2ycbcr
from skimage.transform import resize
import numpy as np
import math
from skimage.morphology import binary_erosion, binary_dilation
from skimage import exposure



##################################################################################################################

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
            new_img[i,j] = maxi
    return new_img






##################################################################################################

input_img = io.imread('testimage.jpg')
filtered_img3 = gaussian(input_img, sigma=0.2)

grey_image =  rgb2gray(filtered_img3)



resized_image=resize(filtered_img3,(200,180))

ycbcr_image = rgb2ycbcr(resized_image).astype('uint8')









y = ycbcr_image[:,:,0]



cb = ycbcr_image[:,:,1]
cr = ycbcr_image[:,:,2]
new_imageycbcr[cr<140]=0
new_imageycbcr[cr>140]=resized_image[cr>140]

#ycbcr_image = rgb2ycbcr(new_imageycbcr).astype('uint8')



#y = ycbcr_image[:,:,0]



#cb = ycbcr_image[:,:,1]
#cr = ycbcr_image[:,:,2]



chromamap= 1/3 *((cb/cr) + cb**2 + (1-cr)**2)

newchromamap = exposure.equalize_hist(chromamap)


windowsize=[2,2]
orgin=[1,1]
ydilate= dilate(y,windowsize,orgin)
yerode= erode(y,windowsize,orgin)

lumamap = ydilate / (yerode+1)


eyemap =np.bitwise_and(newchromamap,lumamap)

print(eyemap)

show_images([ycbcr_image,new_imageycbcr,newchromamap,lumamap,eyemap])
