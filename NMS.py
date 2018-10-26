# -*- coding: utf-8 -*-
# @Time    : 2018/9/11 9:28
# @Author  : sunxk
# @Email   : 944977981@qq.com
# @File    : NMS.py

import cv2
import numpy as np
from ol.openpose import config
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure


def NMS(heatMaps,upsampFactor=config.upsampFactor,
        refineCenter=config.refineCenter,
        gaussianFilt=config.gaussianFilt):
    jointListPerType=[];jointsCnt=0
    for jointType in range(config.numJoints):
        heatMap=heatMaps[:,:,jointType]
        peaksCoords=calcPeakCoords(heatMap,generate_binary_structure(2, 1))
        peaks=np.zeros((len(peaksCoords),4))
        for index,peak in enumerate(peaksCoords):
            if refineCenter:
                refinedCenter,peakScore=calcUpsampledItems(peak,heatMap,upsampFactor,gaussianFilt)
            else:
                refinedCenter=[0,0]
                peakScore=heatMap[tuple(peak[::-1])]
            peakCenter=(np.array(peaksCoords[index],dtype=np.float32)+0.5)*upsampFactor-0.5
            peaks[index,:]=tuple([int(round(x)) for x in peakCenter]+refinedCenter[::-1])+(peakScore,jointsCnt)
            jointsCnt+=1
        jointListPerType.append(peaks)

    return jointListPerType


def calcPeakCoords(heatMap,footPrint):
    maxFilted = maximum_filter(heatMap, footprint=footPrint)
    isPeaks = (heatMap == maxFilted) * (heatMap > config.jointsThres)
    peaksCoords = np.array(np.nonzero(isPeaks)[::-1]).T

    return peaksCoords


def calcUpsampledItems(peak,heatMap,upsampFactor,gaussianFilt):
    xmin, ymin = np.maximum(peak - config.winSize, 0)
    xmax, ymax = np.minimum(np.array(heatMap.T.shape) - 1, peak + config.winSize)
    patch = heatMap[ymin:ymax + 1, xmin:xmax + 1]
    upsampled = cv2.resize(patch, None, fx=upsampFactor, fy=upsampFactor, interpolation=cv2.INTER_CUBIC)
    upsampled = gaussian_filter(upsampled, sigma=3) if gaussianFilt else upsampled
    # 放大后值最大点的坐标
    locationOfMax = np.unravel_index(np.argmax(upsampled), upsampled.shape)
    # 放大后peak的坐标
    upsampledCenter = (np.array(peak[::-1] - [ymin, xmin], np.float) + 0.5) * upsampFactor - 0.5
    refinedCenter = locationOfMax - upsampledCenter
    peakScore = upsampled[locationOfMax]

    return refinedCenter,peakScore