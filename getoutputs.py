# -*- coding: utf-8 -*-
# @Time    : 2018/9/26 14:43
# @Author  : sunxk
# @Email   : 944977981@qq.com
# @File    : getoutputs.py

import numpy as np
import cv2
from ol.openpose import preprocessimg,config,model
import tensorflow as tf


def getInputBatch(img,scales):
    maxImgShape0 = img.shape[0] * scales[-1]
    maxCorpedImg, _, _ = preprocessimg.scaleAndCropImg(img, maxImgShape0)

    batchImgs = np.zeros([len(config.scales), maxCorpedImg.shape[0], maxCorpedImg.shape[1], 3])

    for index, scale in enumerate(scales):
        dstSize = img.shape[0] * scale
        imgCroped, imgScale, realShape = preprocessimg.scaleAndCropImg(img, dstSize)
        imgData = preprocessimg.normalizeImg(imgCroped)
        batchImgs[index, :imgData.shape[0], :imgData.shape[1], :] = imgData

    return batchImgs.astype(np.float32)

def mapToOrigin(img,scale,heatMap,paf):
    imgCroped,imgScale,realShape=preprocessimg.scaleAndCropImg(img,scale*img.shape[0])
    #除以8是因为神经网络将输入缩小八倍
    heatMap=heatMap[:int(imgCroped.shape[0]/config.factor),:int(imgCroped.shape[1]/config.factor),:]
    heatMap=cv2.resize(heatMap,None,fx=config.factor,fy=config.factor,interpolation=cv2.INTER_CUBIC)
    heatMap=heatMap[:realShape[0],:realShape[1],:]
    heatMap=cv2.resize(heatMap,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_CUBIC)

    paf=paf[:int(imgCroped.shape[0]/config.factor),:int(imgCroped.shape[1]/config.factor),:]
    paf=cv2.resize(paf,None,fx=config.factor,fy=config.factor,interpolation=cv2.INTER_CUBIC)
    paf=paf[:realShape[0],:realShape[1],:]
    paf=cv2.resize(paf,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_CUBIC)

    return heatMap,paf

def calcAverage(img,inputHolder,net1,net2,sess):
    scales=[x * 368. / float(img.shape[0]) for x in config.scales]
    batchImgs=getInputBatch(img,scales)
    pafs,heatMaps=sess.run([net1,net2],feed_dict={inputHolder:batchImgs})

    heatMapsAvg = np.zeros((img.shape[0], img.shape[1], 19))
    pafsAvg = np.zeros((img.shape[0], img.shape[1], 38))
    for index,scale in enumerate(scales):
        heatMap,paf=mapToOrigin(img,scale,heatMaps[index],pafs[index])
        heatMapsAvg+=heatMap/len(scales)
        pafsAvg+=paf/len(scales)

    return heatMapsAvg,pafsAvg

def flipMaps(flippedHeatMapsAvg,flippedPafsAvg):
    """
    预测的坐标值是翻转后的图片的坐标值，翻转回原图应该对应原图的坐标->坐标旋转
    假设第1张heatMap预测的是翻转后的左手腕，那么应该对应原图的右手腕->通道翻转

    1、先将每张paf特征图坐标旋转到与原图对应(列维度-1)
    2、坐标对应像素值旋转
        2.1、pafs对应关节方向的单位向量，起始图代表x坐标(对应矩阵列维度)，结束图道标y坐标（对应矩阵行维度）
        2.2、操作将矩阵列维度取反，相当于单位方向向量x坐标取反
    3、通道旋转，左边还是对应左边，右边对应右边(config.swapPaf)
    """

    flippedPafsAvg=flippedPafsAvg[:,::-1,:]
    flippedPafsAvg[:,:,config.swapPaf[1::2]]=flippedPafsAvg[:,:,config.swapPaf[1::2]]
    flippedPafsAvg[:,:,config.swapPaf[::2]]=-flippedPafsAvg[:,:,config.swapPaf[::2]]
    pafsAvg_=flippedPafsAvg[:,:,config.swapPaf]

    heatMapsAvg_=flippedHeatMapsAvg[:,::-1,:][:,:,config.swapHeat]

    return heatMapsAvg_,pafsAvg_




if __name__=='__main__':
    imgPath='./ski.jpg'
    img=cv2.imread(imgPath)
    heatMapsAvg,pafsAvg=calcAverage(img)
    print('heatMapsShape:',heatMapsAvg.shape)
    print('pafsShape:',pafsAvg.shape)