from commonfunctions import *
from skimage.filters import gaussian 
from skimage.color import rgb2ycbcr
from skimage.color import rgb2hsv
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.morphology import erosion, dilation,binary_closing,binary_opening,binary_erosion,binary_dilation
from skimage.measure import find_contours
import numpy as np
import math
import cv2


def getfaces(img):

    imgRGB=img
    imgHSV=rgb2hsv(img)
    imgYCbCr=rgb2ycbcr(img)

    

    ##RGB
    R = imgRGB[:,:,0]
    G = imgRGB[:,:,1]
    B = imgRGB[:,:,2]
    ##HSV
    H = imgHSV[:,:,0]
    S = imgHSV[:,:,1]
    V = imgHSV[:,:,2]
    ##YCbCr
    y = imgYCbCr[:,:,0]
    cb = imgYCbCr[:,:,1]
    cr = imgYCbCr[:,:,2]

    #(R>50) && (G>40) && (B>20) && [max{max(R,G),B}– min{min(R,G),B}]>10) && (R – G >= 10) && (R>G) && (R>B)
    

    #Eq1
    part1=np.zeros((imgRGB.shape[0],imgRGB.shape[1]))
    part2=np.zeros((imgRGB.shape[0],imgRGB.shape[1]))
    part3=np.zeros((imgRGB.shape[0],imgRGB.shape[1]))
    part4=np.zeros((imgRGB.shape[0],imgRGB.shape[1]))
    part5=np.zeros((imgRGB.shape[0],imgRGB.shape[1]))


    part1[R>50]=1
    part2[B>20]=1
    #part1[abs(max(max(R,G),B)-min(min(R,G),B))>10]=1
    part3[abs(R-G)>=10]=1
    part4[abs(R>G)]=1
    part5[abs(R>B)]=1
    out12=np.logical_and(part1,part2)
    out34=np.logical_and(part3,part4)
    out125=np.logical_and(out12,part5)
    Equation1=np.logical_and(out125,out34)



    #eq2
    part1=np.zeros((imgRGB.shape[0],imgRGB.shape[1]))
    part2=np.zeros((imgRGB.shape[0],imgRGB.shape[1]))
    part3=np.zeros((imgRGB.shape[0],imgRGB.shape[1]))
    part4=np.zeros((imgRGB.shape[0],imgRGB.shape[1]))
    part5=np.zeros((imgRGB.shape[0],imgRGB.shape[1]))
    part6=np.zeros((imgRGB.shape[0],imgRGB.shape[1]))


    part1[R>220]=1
    part2[B>170]=1
    part3[G>210]=1
    part4[abs(G>B)]=1
    part5[abs(R>B)]=1
    part6[abs(R-G)<=15]=1

    out12=np.logical_and(part1,part2)
    out34=np.logical_and(part3,part4)
    out56=np.logical_and(part6,part5)
    out1234=np.logical_and(out12,out34)
    Equation2=np.logical_and(out1234,out56)


    ruleA=np.logical_or(Equation1,Equation2)



    #eq3
    part1=np.zeros((imgYCbCr.shape[0],imgYCbCr.shape[1]))
    part2=np.zeros((imgYCbCr.shape[0],imgYCbCr.shape[1]))


    part1[cb<=130]=1
    part2[cr>=140]=1

    RuleB=np.logical_and(part1,part2)




    #eq7
    part1=np.zeros((imgHSV.shape[0],imgHSV.shape[1]))
    part2=np.zeros((imgHSV.shape[0],imgHSV.shape[1]))

    part1[H>=0]=1
    part2[H<=50]=1

    Equation7=np.logical_and(part1,part2)


    #eq8
    part1=np.zeros((imgHSV.shape[0],imgHSV.shape[1]))
    part2=np.zeros((imgHSV.shape[0],imgHSV.shape[1]))

    part1[S>=0.1]=1
    part2[S<=0.9]=1

    Equation8=np.logical_and(part1,part2)
    RuleC=np.logical_and(Equation8,Equation7)


    RuleBC=np.logical_and(RuleC,RuleB)
    skin=np.logical_and(RuleBC,ruleA)

    #show_images([skin])

    selem=np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])

    #skin=binary_closing(skin,selem=selem); 
    '''
    show_images([skin])
    skin=binary_erosion(skin,selem=selem); 
    show_images([skin])
    skin=binary_erosion(skin,selem=selem);  
    
    
    show_images([skin])
    skin=binary_dilation(skin,selem=selem); 
    show_images([skin])
    skin=binary_dilation(skin,selem=selem);  
    '''
    #show_images([skin])
    #print(skin)


    output = cv2.connectedComponentsWithStats(skin.astype('uint8'), 4)


    num_labels = output[0]
    labels = output[1]
    #for i in range()
    stats = output[2]
    lblareas = stats[1:,cv2.CC_STAT_AREA]

    #index = max(enumerate(lblareas), key=(lambda x: x[1]))[0] + 1
    #print(index)
    lblareas[lblareas<1500]=0
    #print(lblareas)
    #print(remaining)
    #index2= np.argmax(lblareas)+1
    areas=[]
    for i in range(len(lblareas)):
        if lblareas[i]!=0:
            #print(i)
            x = stats[i+1, cv2.CC_STAT_LEFT]
            y = stats[i+1, cv2.CC_STAT_TOP]
            w = stats[i+1, cv2.CC_STAT_WIDTH]
            h = stats[i+1, cv2.CC_STAT_HEIGHT]
            ratio=w/h
            #(ratio<0.4 || ratio>1.1)
            if ratio>0.4 and ratio<1.1:
                #out=cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                areas.append([int(w),int(h),int(x),int(y)])
                #show_images([out])
    #print(stats.shape)

    contours, hierarchy = cv2.findContours(skin.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #print(contours)
    thickness = -1
    faces=[]
    for ind, cont in enumerate(contours):
        if(len(cont)>5):
            (x,y),(MA,ma),angle = cv2.fitEllipse(cont)
            if(MA/ma >0.25 and MA/ma<0.97):
                for i in range(len(areas)):
                    if MA*ma/(areas[i][0]*areas[i][1]) >0.65 and areas[i][2]<x<areas[i][2]+areas[i][0] and areas[i][3]<y<(areas[i][3]+areas[i][1]):
                        #print("hi")
                        #out=cv2.ellipse(img,(int(x),int(y)),(int(MA/2), int(ma/2)),angle,0,360,(0,0,255),thickness)
                        #out=cv2.rectangle(img, (int(x-MA/2), int(y-ma/2)), (int(x + MA/2), int(y + ma/2)), (255, 0, 0), 2)
                        #show_images([out])
                        faces.append([areas[i][2],areas[i][3],areas[i][0], areas[i][1]])
    return faces
