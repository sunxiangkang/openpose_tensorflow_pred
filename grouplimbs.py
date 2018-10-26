# -*- coding: utf-8 -*-
# @Time    : 2018/10/7 15:31
# @Author  : sunxk
# @Email   : 944977981@qq.com
# @File    : grouplimbs.py

import numpy as np
from ol.openpose import config


def groupLimb(connectedLimbs,jointList):
    personJointAssoc=[]
    for limbType,joints in enumerate(config.jointToLimb_heatmapRelationship):
        jointSrc=joints[0];jointDst=joints[1]
        for limbInfo in connectedLimbs[limbType]:
            personAssocInd=[]
            for person,personLimbs in enumerate(personJointAssoc):
                if personLimbs[jointSrc]==limbInfo[0] or personLimbs[jointDst]==limbInfo[1]:
                    personAssocInd.append(person)
            if len(personAssocInd)==1:
                personLimbs=personJointAssoc[personAssocInd[0]]
                if personLimbs[jointDst]!=limbInfo[1]:
                    personLimbs[jointDst]=limbInfo[1]
                    personLimbs[-1]+=1
                    personLimbs[-2]+=jointList[limbInfo[1].astype(int),2]+limbInfo[2]
            elif len(personAssocInd)==2:
                personLimbs1=personJointAssoc[personAssocInd[0]]
                personLimbs2=personJointAssoc[personAssocInd[1]]
                memberShip=((personLimbs1>=0)&(personLimbs2>=0))[:-2]
                if not memberShip.any():
                    personLimbs1[:-2]+=personLimbs2[:-1]
                    personLimbs1[-2:]+=personLimbs2[-2:]
                    personLimbs1[-2]+=limbInfo[2]
                    personJointAssoc.pop(personAssocInd[1])
                else:
                    pass
            else:
                row=-1*np.ones(20)
                row[jointSrc]=limbInfo[0];row[jointDst]=limbInfo[1]
                row[-1]=2;row[-2]=sum(jointList[limbInfo[:2].astype(int),2])+limbInfo[2]
                personJointAssoc.append(row)

    personToDelete=[]
    for person,personLimbInfo in enumerate(personJointAssoc):
        if personLimbInfo[-1]<3 or personLimbInfo[-2]/personLimbInfo[-1]<0.2:
            personToDelete.append(person)
    for person in personToDelete[::-1]:
        personJointAssoc.pop(person)

    return np.array(personJointAssoc)