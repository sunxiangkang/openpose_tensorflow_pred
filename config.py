# -*- coding: utf-8 -*-
# @Time    : 2018/9/26 14:46
# @Author  : sunxk
# @Email   : 944977981@qq.com
# @File    : config.py


"""数据结构
NMS返回jointListPerType(三维)：
    [第一类关节：[
                  第一个：[x,y,score,唯一id]，
                  第二个：[x,y,score,唯一id]，
                  第n个： [x,y,score,唯一id]
                  ]
    第二类关节：[
                  第一个：[x,y,score,唯一id]，
                  第二个：[x,y,score,唯一id]，
                  第n个： [x,y,score,唯一id]
                  ],
    第n类关节：[
                ]
    ]

findConnectedJoints返回connectedLimbs(三维)：
    [第一类肢干：[
                  第一个肢干：[起始关节唯一id,结束关节唯一id,惩罚后连接肢干score,
                                              起始关节是这一类关节的第几个，结束关节是这一类关节的第一个]，
                  第二个肢干：[起始关节唯一id,结束关节唯一id,惩罚后连接肢干score,
                                              起始关节是这一类关节的第几个，结束关节是这一类关节的第一个]，
                  第n 个肢干：[起始关节唯一id,结束关节唯一id,惩罚后连接肢干score,
                                              起始关节是这一类关节的第几个，结束关节是这一类关节的第一个]
                 ]，
    第二类肢干：[
                  第一个肢干：[起始关节唯一id,结束关节唯一id,惩罚后连接肢干score,
                                              起始关节是这一类关节的第几个，结束关节是这一类关节的第一个]，
                  第二个肢干：[起始关节唯一id,结束关节唯一id,惩罚后连接肢干score,
                                              起始关节是这一类关节的第几个，结束关节是这一类关节的第一个]，
                  第n 个肢干：[起始关节唯一id,结束关节唯一id,惩罚后连接肢干score,
                                              起始关节是这一类关节的第几个，结束关节是这一类关节的第一个]
                 ]
    第n类肢干：[
                ]
    ]

groupLimbs中的jointList：
    [
        [x,y,score,关节唯一id，关节是jointPerList中第几类关节]，
        ......,
    ]

"""

import numpy as np

factor=8

isCeil=True

#scales=[0.5,1.0,1.5,2.0,2.5]
scales=[1.0]

modelPath='./model/modeltensor'

swapHeat = np.array((0, 1, 5, 6, 7, 2, 3, 4, 11, 12,
                      13, 8, 9, 10, 15, 14, 17, 16, 18))

swapPaf = np.array((6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 20, 21, 22, 23,
                     24, 25, 26, 27, 12, 13, 14, 15, 16, 17, 18, 19, 28,
                     29, 32, 33, 30, 31, 36, 37, 34, 35))

upsampFactor=1

refineCenter=True

gaussianFilt=False

numJoints=18

jointsThres=0.1

winSize=2

intermedPtsnum=10

#The order in this work:
#(0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
#5-'left_shoulder' 6-'left_elbow'	    7-'left_wrist'  8-'right_hip'
#9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee'
#13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear'
#17-'left_ear' )

jointToLimb_heatmapRelationship = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6],
                                   [6, 7], [1, 8], [8, 9], [9, 10],[1, 11],
                                   [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
                                   [0, 15], [15, 17],[2, 16], [5, 17]]

limbToPafs = [[12, 13], [20, 21], [14, 15], [16, 17],
              [22, 23],[24, 25], [0, 1], [2, 3], [4, 5],
              [6, 7], [8, 9], [10, 11], [28, 29],[30, 31],
              [34, 35], [32, 33], [36, 37], [18, 19], [26, 27]]

connectJointsThres=0.05

colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]