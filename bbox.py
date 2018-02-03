# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 19:41:08 2017

@author: Shobhit
"""

from PIL import Image
import pandas as pd

path = "/diskb/tmp/CXR8BBox_List_2017.csv"
df = pd.read_csv(path)
index = df['Image Index']

def cropImages(array, cropIteration, cropUnit):
    imgcrop = []
    for k in range(0, 200):
       imgcrop.append(Image.new('RGBA', [901, 874]))
       try:     
            if(cropIteration == 0):
                p1 = (5.4179894179999994, 12.83793403)
                xadj = 906.537989418 
                yadj = 886.21782823
                p2 = (xadj - cropUnit, yadj)
            
            elif(cropIteration == 1):
                p1 = (5.4179894179999994, 12.83793403)
                xadj = 906.537989418 
                yadj = 886.21782823
                p2 = (xadj, yadj - cropUnit)
            
            elif(cropIteration == 2):
                p1 = (5.4179894179999994, 12.83793403 + cropUnit)
                xadj = 906.537989418 
                yadj = 886.21782823
                p2 = (xadj, yadj)
            
            elif(cropIteration == 3):
                p1 = (5.4179894179999994+ cropUnit, 12.83793403)
                xadj = 906.537989418 
                yadj = 886.21782823
                p2 = (xadj, yadj)    
            
            print("Cropping Image: " +str(k))
            #print("p1 = "+ str(p1) + ", p2 = " +str(p2))
            xy = p1[0], p1[1], p2[0], p2[1]
            imgcrop[k] = array[k].crop(xy)
            imgcrop[k].save("/diskb/tmp/CXR8/labelled_bbox/"+str(index[k]))
            
       except (SystemError):
            print("Cannot crop the image any more")
            k = 199
       except (FileNotFoundError):
            print("Please check the path syntax pecified")
