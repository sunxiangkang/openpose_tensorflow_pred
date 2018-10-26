# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 13:35
# @Author  : sunxk
# @Email   : 944977981@qq.com
# @File    : pipline.py

import model
import tensorflow as tf
import numpy as np
import getoutputs
import config
import cv2
import NMS
import findconnectedjoints
import grouplimbs
import plotlimbs
import restoremodel

def main(img):
    inputHolder,net1,net2,sess=restoremodel.restoreModel()
    heatMapsAvgO,pafsAvgO=getoutputs.calcAverage(img,inputHolder,net1,net2,sess)
    flippedHeatMapsAvg,flippedPafsAvg=getoutputs.calcAverage(img[:,::-1,:],inputHolder,net1,net2,sess)
    heatMapsAvg_,pafsAvg_=getoutputs.flipMaps(flippedHeatMapsAvg,flippedPafsAvg)

    heatMapsAvg=(heatMapsAvgO+heatMapsAvg_)/2
    pafsAvg=(pafsAvgO+pafsAvg_)/2

    jointListPerType=NMS.NMS(heatMapsAvg)

    jointList = np.array([tuple(peak) + (jointType,) for jointType,joint_peaks in
                          enumerate(jointListPerType) for peak in joint_peaks])

    #drawPoints(jointList)

    pafsReshaped=cv2.resize(pafsAvg,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_CUBIC)
    connectedLimbs=findconnectedjoints.findConnectedJoints(jointListPerType,pafsReshaped)
    personJointAssoc=grouplimbs.groupLimb(connectedLimbs,jointList)
    canvs=plotlimbs.plotLimbs(img,jointList,personJointAssoc)

    cv2.imshow('res',canvs)
    cv2.waitKey()

    return jointList,personJointAssoc,canvs


def drawPoints(joint_list):
    img=cv2.imread('./ski.jpg')
    for item in joint_list:
        cv2.circle(img,(int(item[0]),int(item[1])),2,(0,0,255),8)
    cv2.imshow("test",img)
    cv2.waitKey()


if __name__=='__main__':
    img=cv2.imread('./ski.jpg')
    main(img)
