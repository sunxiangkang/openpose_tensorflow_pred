# -*- coding: utf-8 -*-
# @Time    : 2018/10/25 8:49
# @Author  : sunxk
# @Email   : 944977981@qq.com
# @File    : restoremodel.py

import tensorflow as tf
import model
import config
import cv2
import getoutputs

def restoreModel():
    inputHolder=tf.placeholder(tf.float32,shape=[None,None,None,3])
    net=model.vgg19(inputHolder)
    (net1,net2),_=model.makeStage(net)

    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(config.modelPath)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    return inputHolder,net1,net2,sess


if __name__=='__main__':
    imgPath = './ski.jpg'
    img = cv2.imread(imgPath)
    scales = [x * 368. / float(img.shape[0]) for x in config.scales]
    batchImgs = getoutputs.getInputBatch(img, scales)
    inputHolder,net1,net2,sess=restoreModel()
    res1,res2=sess.run([net1,net2],feed_dict={inputHolder:batchImgs})
    print(res2[0,:,0,0])