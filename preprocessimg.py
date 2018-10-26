# -*- coding: utf-8 -*-
# @Time    : 2018/9/26 13:35
# @Author  : sunxk
# @Email   : 944977981@qq.com
# @File    : imgenforce.py

import cv2
import numpy as np
from ol.openpose import config

#[x * 368. / float(img.shape[0]) for x in scale_search]

#图片处理先放大到设置的最大大小，然后除以factor缩小到factor整数倍
def scaleAndCropImg(img,dstSize,factor=config.factor,isCeil=config.isCeil):
    imgSizeMin=np.min(img.shape[:2])

    imgScale=float(dstSize)/imgSizeMin
    imgS=cv2.resize(img,None,fx=imgScale,fy=imgScale)

    h,w,c=imgS.shape
    newH=int(np.ceil(float(h)/factor) if isCeil else np.floor(float(h)/factor))*factor
    newW=int(np.ceil(float(w)/factor) if isCeil else np.floor(float(h)/factor))*factor

    newImg=np.zeros([newH,newW,c],img.dtype)
    newImg[:h,:w,:]=imgS

    return newImg,imgScale,imgS.shape

def normalizeImg(img):
    img=img.astype(np.float32)
    img=img/256.-0.5
    return img