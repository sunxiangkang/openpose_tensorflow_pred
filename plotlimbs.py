# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 14:05
# @Author  : sunxk
# @Email   : 944977981@qq.com
# @File    : plotlimbs.py

import cv2
from ol.openpose import config
import numpy as np
import math


def plotLimbs(img,jointList,personJointAssoc,plotEarToShoulder=False):
    canvas=img.copy()
    limbToPlot=len(config.jointToLimb_heatmapRelationship) if plotEarToShoulder \
        else len(config.jointToLimb_heatmapRelationship)-2
    for limbType in range(limbToPlot):
        for personLimbInfo in personJointAssoc:
            jointInd=personLimbInfo[config.jointToLimb_heatmapRelationship[limbType]].astype(int)
            if -1 in jointInd:
                continue
            jointCoords=jointList[jointInd,0:2]
            for joint in jointCoords:
                cv2.circle(canvas,tuple(joint.astype(np.int)),4,(0,0,255),thickness=-1)
            coordCenter=tuple(np.round(np.mean(jointCoords,0)).astype(int))
            limbVec=jointCoords[0]-jointCoords[1]
            limbLength=np.linalg.norm(limbVec)
            angle=math.degrees(math.atan2(limbVec[1],limbVec[0]))
            polygon=cv2.ellipse2Poly(coordCenter,(int(limbLength/2),4),int(angle),0,360,1)
            cv2.fillConvexPoly(canvas,polygon,config.colors[limbType])

    return canvas