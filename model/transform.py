# -*- coding: utf-8 -*-
# @Time    : 2018/9/25 8:42
# @Author  : sunxk
# @Email   : 944977981@qq.com
# @File    : transform.py

#tesorflow框架下卷积层的权重shape是[kernel_height, kernel_width, input_channels, output_channels]
# (相反，反卷积层的权重shape是[kernel_height, kernel_width, output_channels, input_channels])

#在caffe框架中，卷积层权重参数shape是[output_channels, input_channels, kernel_height, kernel_width]
# (相反地，反卷积层权重参数是[input_channels, output_channels, kernel_height, kernel_width])


import caffe
import tensorflow as tf
from ol.openpose import model
import os

prototxtpath='./modelcaffe/pose_deploy_linevec.prototxt'
caffeModelpath='./modelcaffe/pose_iter_440000.caffemodel'
tensorModelPath='./modeltensor'


def main(tensorModelPath=tensorModelPath):
    net = caffe.Net(prototxtpath, caffeModelpath, caffe.TEST)
    weights = net.params
    
    sess=tf.Session()
    inputHolder = tf.placeholder(tf.float32, shape=(None, 512, 512, 3))
    net = model.vgg19(inputHolder)
    (net1, net2), savedForLoss = model.makeStage(net)
    init=tf.global_variables_initializer()
    sess.run(init)
    saver=tf.train.Saver()

    for var in tf.global_variables():
        scopeName, variableName = var.name.split('/')
        with tf.variable_scope(scopeName, reuse=True):
            weight=tf.get_variable('weights')
            bias=tf.get_variable('biases')
            if weights[scopeName]:
                weightC=weights[scopeName][0].data.transpose(2,3,1,0)
                biasC=weights[scopeName][1].data
                sess.run([tf.assign(weight,weightC),tf.assign(bias,biasC)])
            else:
                raise KeyError('Wrong key:{}'.format(scopeName))

    saver.save(sess,os.path.join(tensorModelPath,'model.ckpt'))

if __name__=='__main__':
    main()