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


def isin_box(xb1,yb1,xb2,yb2,xs1,ys1,xs2,ys2,img):
    imgb=img[yb1:yb2,xb1:xb2].copy()
    img_merged=imgb[ys1-yb1:ys2-yb1,xs1-xb1:xs2-xb1].copy()
    imgs=img[ys1:ys2,xs1:xs2].copy()
    if np.array_equal(img_merged, imgs):
        return True
    else:
        return False
    

def num_latter(predtected_img_coor,croped_img):
    list_xyxy=[]
    space_dict={}
    img_latters=[]
    image_num=[]

    for i,item in enumerate(predtected_img_coor.xyxy[0]):
        item=item.int()[:4].tolist()
        space=(item[2]-item[0])*(item[3]-item[1])
        list_xyxy.append(space)
        space_dict[str(space)]=item
    
    list_xyxy.sort()
    try:
        list1=space_dict[ str(list_xyxy[-1])]
        list2=space_dict[ str(list_xyxy[-2])]

        if list1[-2] > list2[-2]:
            numbers=list2.copy()
            latters=list1.copy()
        else:
            numbers=list1.copy()
            latters=list2.copy()

        # print(numbers)
        # print(latters)
        for j,box in enumerate( [numbers,latters]):
            xb1,yb1,xb2,yb2=box
            print(list_xyxy)
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
    except :
        pass
    return img_latters , image_num



def xywh_to_xyxy(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2

def xyxy_to_xywh(xmin, ymin, xmax, ymax):
    width = xmax - xmin
    height = ymax - ymin
    x = xmin
    y = ymin
    return x, y, width, height


def xywh_from_yolo_label_to_xyxy(x,y,w,h,width,height):
    xmin = int((x - w/2) * width)
    ymin = int((y - h/2) * height)
    xmax = int((x + w/2) * width)
    ymax = int((y + h/2) * height)
    return xmin,ymin,xmax,ymax

def xyxy_to_yolo_label(xmin, ymin, xmax, ymax, image_width, image_height):
    box_width = xmax - xmin
    box_height = ymax - ymin
    x_center = (xmin + xmax) / 2 / image_width
    y_center = (ymin + ymax) / 2 / image_height
    width = box_width / image_width
    height = box_height / image_height
    return x_center, y_center, width, height


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