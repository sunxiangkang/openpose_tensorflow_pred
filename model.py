# -*- coding: utf-8 -*-
# @Time    : 2018/9/8 8:29
# @Author  : sunxk
# @Email   : 944977981@qq.com
# @File    : model.py

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import cv2
import config
import getoutputs

def ppad(input,padSize):
    padMat=np.array([[0,0],[padSize,padSize],[padSize,padSize],[0,0]])
    paddedInput=tf.pad(input,padMat,name='pad')
    return paddedInput

def vgg19(inputx,stddev=0.01):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.random_normal_initializer(stddev=stddev),
                        biases_initializer=tf.constant_initializer(0.0),
                        padding='VALID',
                        stride=1,
                        activation_fn=tf.nn.relu
                        ):
        net=inputx
        for i in range(2):
            net=ppad(net,1)
            net=slim.conv2d(net,64,[3,3],scope='conv1_{}'.format(i+1))
        net=slim.max_pool2d(net,[2,2],stride=2,scope='pool1_stage1')

        for i in range(2):
            net=ppad(net,1)
            net=slim.conv2d(net,128,[3,3],scope='conv2_{}'.format(i+1))
        net=slim.max_pool2d(net,[2,2],stride=2,scope='pool2_stage1')

        for i in range(4):
            net=ppad(net,1)
            net=slim.conv2d(net,256,[3,3],scope='conv3_{}'.format(i+1))
        net=slim.max_pool2d(net,[2,2],stride=2,scope='pool3_stage1')

        for i in range(2):
            net=ppad(net,1)
            net=slim.conv2d(net,512,[3,3],scope='conv4_{}'.format(i+1))
        net=ppad(net,1)
        net=slim.conv2d(net,256,[3,3],scope='conv4_3_CPM')
        net=ppad(net,1)
        net=slim.conv2d(net,128,[3,3],scope='conv4_4_CPM')

    return net

def makeStage(net,stddev=0.1):
    savedForLoss=[]
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.random_normal_initializer(stddev=stddev),
                        biases_initializer=tf.constant_initializer(0.0),
                        padding='VALID',
                        stride=1,
                        activation_fn=tf.nn.relu
                        ):
        netO=net
        net_1=net;net_2=net
        #stage1_L1
        for i in range(3):
            net_1=ppad(net_1,1)
            net_1=slim.conv2d(net_1,128,[3,3],scope='conv5_{}_CPM_L1'.format(i+1))
        net_1=slim.conv2d(net_1,512,[1,1],scope='conv5_4_CPM_L1')
        net_1=slim.conv2d(net_1,38,[1,1],activation_fn=None,scope='conv5_5_CPM_L1')
        #stage1_L2
        for i in range(3):
            net_2=ppad(net_2,1)
            net_2=slim.conv2d(net_2,128,[3,3],scope='conv5_{}_CPM_L2'.format(i+1))
        net_2=slim.conv2d(net_2,512,[1,1],scope='conv5_4_CPM_L2')
        net_2=slim.conv2d(net_2,19,[1,1],activation_fn=None,scope='conv5_5_CPM_L2')

        savedForLoss.append(net_1)
        savedForLoss.append(net_2)
        net=tf.concat([net_1,net_2,netO],axis=-1)

        #stage2-6
        for s in range(2,7):
            net_1 = net;net_2 = net
            #L1
            for i in range(5):
                net_1=ppad(net_1,3)
                net_1=slim.conv2d(net_1,128,[7,7],scope='Mconv{}_stage{}_L1'.format(i+1,s))
            net_1=slim.conv2d(net_1,128,[1,1],scope='Mconv6_stage{}_L1'.format(s))
            net_1=slim.conv2d(net_1,38,[1,1],activation_fn=None,scope='Mconv7_stage{}_L1'.format(s))
            #L2
            for i in range(5):
                net_2=ppad(net_2,3)
                net_2=slim.conv2d(net_2,128,[7,7],scope='Mconv{}_stage{}_L2'.format(i+1,s))
            net_2=slim.conv2d(net_2,128,[1,1],scope='Mconv6_stage{}_L2'.format(s))
            net_2=slim.conv2d(net_2,19,[1,1],activation_fn=None,scope='Mconv7_stage{}_L2'.format(s))

            savedForLoss.append(net_1)
            savedForLoss.append(net_2)
            net=tf.concat([net_1,net_2,netO],axis=-1)

    return (net_1,net_2),savedForLoss

if __name__=="__main__":
    imgPath = './ski.jpg'
    img = cv2.imread(imgPath)
    scales = [x * 368. / float(img.shape[0]) for x in config.scales]
    batchImgs = getoutputs.getInputBatch(img, scales)
    inputHolder=tf.placeholder(tf.float32,shape=[None,None,None,3])
    net = vgg19(inputHolder)
    (net1,net2),_=makeStage(net)

    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(config.modelPath)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    #res = sess.run(net, feed_dict={inputHolder: batchImgs})
    res1,res2=sess.run([net1,net2],feed_dict={inputHolder:batchImgs})
    print(res2[0,:,0,0])
    #print(res[0,:,0,0])
