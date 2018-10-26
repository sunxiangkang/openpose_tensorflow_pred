# -*- coding: utf-8 -*-
# @Time    : 2018/9/30 15:47
# @Author  : sunxk
# @Email   : 944977981@qq.com
# @File    : findconnectedjoints.py

import numpy as np
from ol.openpose import config

def findConnectedJoints(jointListPerType,pafsReshaped):
    connectedLimbs=[]
    for index,limbType in enumerate(config.jointToLimb_heatmapRelationship):
        jointsSrc=jointListPerType[limbType[0]]
        jointsDst=jointListPerType[limbType[1]]
        if len(jointsSrc)==0 or len(jointsDst)==0:
            connectedLimbs.append([])
        else:
            candidateConnections=[]
            srcPafIndex=[config.limbToPafs[index][0]]*config.intermedPtsnum
            dstPafIndex=[config.limbToPafs[index][1]]*config.intermedPtsnum
            for i,jointSrc in enumerate(jointsSrc):
                for j,jointDst in enumerate(jointsDst):
                    scoreInteremedPts,scorePenalizingLongDIst=calcCorrelation\
                        (pafsReshaped,srcPafIndex,dstPafIndex,jointSrc,jointDst)
                    criterion1=(np.count_nonzero(scoreInteremedPts>config.connectJointsThres)>
                                0.8*config.intermedPtsnum)
                    criterion2=(scorePenalizingLongDIst>0)
                    if criterion1 and criterion2:
                        candidateConnections.append([i,j,scorePenalizingLongDIst,
                                                     scorePenalizingLongDIst+jointSrc[2]+jointDst[2]])
            limbs=pickupLimbs(candidateConnections,jointsSrc,jointsDst)
            connectedLimbs.append(limbs)

    return connectedLimbs


def calcCorrelation(pafsReshaped,srcPafIndex,dstPafIndex,jointSrc,jointDst):
    limbVec = jointDst[:2] - jointSrc[:2]
    limbDist = np.sqrt(np.sum(limbVec ** 2)) + 1e-8
    limbVec /= limbDist
    rowIndex= np.round(np.linspace(jointSrc[0], jointDst[0], num=config.intermedPtsnum)).astype(np.int)
    colIndex = np.round(np.linspace(jointSrc[1], jointDst[1], num=config.intermedPtsnum)).astype(np.int)
    intermedPaf = pafsReshaped[colIndex, rowIndex, [srcPafIndex,dstPafIndex]].T
    scoreInteremedPts = intermedPaf.dot(limbVec)
    scorePenalizingLongDist = scoreInteremedPts.mean() + min(0.5 * pafsReshaped.shape[0] /limbDist-1,0)

    return scoreInteremedPts,scorePenalizingLongDist

def pickupLimbs(candidateConnections,jointsSrc,jointsDst):
    candidateConnections = sorted(candidateConnections, key=lambda x: x[2], reverse=True)
    limbs=np.empty((0,5))
    maxLimbs=min(len(jointsDst),len(jointsSrc))
    for item in candidateConnections:
        i,j,score=item[:3]
        if i not in limbs[:,3] and j not in limbs[:,4]:
            limbs=np.vstack([limbs,[jointsSrc[i][3],jointsDst[j][3],score,i,j]])
            if len(limbs)>maxLimbs:
                break

    return limbs