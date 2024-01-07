from ultralytics import YOLO
import cv2
import numpy as np
import math
import random 
import os 
import numpy as np 
import requests
import matplotlib.pyplot as plt 
from  PIL import Image
import torch
import pandas as pd 
import time
from datetime import datetime
import torch
import tensorflow as tf

dict_from_arr={0: '0',
 1: '1',
 2: '2',
 3: '3',
 4: '4',
 5: '5',
 6: '6',
 7: '7',
 8: '8',
 9: '9',
 10: 'A',
 11: 'B',
 12: 'D',
 13: 'E',
 14: 'G',
 15: 'H',
 16: 'J',
 17: 'K',
 18: 'L',
 19: 'N',
 20: 'R',
 21: 'S',
 22: 'T',
 23: 'U',
 24: 'V',
 25: 'X',
 26: 'Z'}
def calculate_iou(box1, box2):
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

def isin_box(xb1,yb1,xb2,yb2,xs1,ys1,xs2,ys2,img):
    imgb=img[yb1:yb2,xb1:xb2].copy()
    img_merged=imgb[ys1-yb1:ys2-yb1,xs1-xb1:xs2-xb1].copy()
    imgs=img[ys1:ys2,xs1:xs2].copy()
    if np.array_equal(img_merged, imgs):
        return True

    elif xs1>xb1 and xs2<xb2:
        box1=[xs1,yb1,xb2,yb2]
        box2=[xs1,ys1,xs2,ys2]
        iou = calculate_iou(box1, box2)
        if iou >0:
            return True
        else:
            return False
    

def num_latter(predtected_img_coor,croped_img):
    list_xyxy=[]
    space_dict={}
    img_latters=[]
    image_num=[]
    spaces=[]

    for i,item in enumerate(predtected_img_coor.xyxy):
        item=item.int()[:4].tolist()
        space=(item[2]-item[0])*(item[3]-item[1])
        
        if space not in spaces:
            space_dict[str(space)]=item
            spaces.append(space)
            list_xyxy.append(space)
        else:
            space_dict[str(space+1)]=item
            list_xyxy.append(space+1)
            spaces.append(space+1)
    
    list_xyxy.sort()
    try:
        list1=space_dict[ str(list_xyxy[-1])]
        list2=space_dict[ str(list_xyxy[-2])]
        lis=[list2]+[list1]
        lis.sort(key=lambda x: x[0])
        numbers=lis[0]
        latters=lis[1]

        # print(numbers)
        # print(latters)
        for j,box in enumerate( [numbers,latters]):
            xb1,yb1,xb2,yb2=box

            for z in  range(len( list_xyxy[:-2])):
                
                coordenates=space_dict[ str(list_xyxy[z])]
                xs1,ys1,xs2,ys2=coordenates
                
                if isin_box(xb1,yb1,xb2,yb2,xs1,ys1,xs2,ys2,croped_img.copy()):
                    if j == 0:
                        image_num.append(coordenates)
                    elif j ==1:
                        img_latters.append(coordenates)
        
        img_latters.sort(key=lambda x: x[-2]) 
        image_num.sort(key=lambda x: x[-2]) 
    except:
        pass
    return space_dict,list_xyxy,img_latters , image_num