import numpy as np
import tensorflow as tf
REGULARIZER_COF = 1e-8






def _instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset


def _fc_variable( weight_shape,name="fc"):
    with tf.variable_scope(name):
        # check weight_shape
        input_channels  = int(weight_shape[0])
        output_channels = int(weight_shape[1])
        weight_shape    = (input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)

        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer =regularizer)
        bias   = tf.get_variable("b", [weight_shape[1]],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv_variable( weight_shape,name="conv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels  = int(weight_shape[2])
        output_channels = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                regularizer=regularizer)
        bias   = tf.get_variable("b", [output_channels],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _deconv_variable( weight_shape,name="conv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        output_channels = int(weight_shape[2])
        input_channels  = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape    ,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                regularizer=regularizer)
        bias   = tf.get_variable("b", [input_channels], initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv2d( x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def _deconv2d( x, W, output_shape, stride=1):
    # x           : [nBatch, height, width, in_channels]
    # output_shape: [nBatch, height, width, out_channels]
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,stride,stride,1], padding = "SAME",data_format="NHWC")

def _conv_layer(x, input_layer, output_layer, stride, filter_size=3, name="conv", isTraining=True):
    conv_w, conv_b = _conv_variable([filter_size,filter_size,input_layer,output_layer],name="conv"+name)
    h = _conv2d(x,conv_w,stride=stride) + conv_b
    h = _instance_norm(h,name)
    h = tf.nn.leaky_relu(h)
    return h

def _deconv_layer(x,input_layer, output_layer, stride=2, filter_size=3, name="deconv", isTraining=True):
    bs, h, w, c = x.get_shape().as_list()
    deconv_w, deconv_b = _deconv_variable([filter_size,filter_size,input_layer,output_layer],name="deconv"+name )
    h = _deconv2d(x,deconv_w, output_shape=[bs,h*2,w*2,output_layer], stride=stride) + deconv_b
    h = _instance_norm(h,name)
    h = tf.nn.leaky_relu(h)
    return h

def buildGenerator(x,reuse=False,isTraining=True,nBatch=64,ksize=4,resBlock=9,name="generator"):

    with tf.variable_scope(name, reuse=reuse) as scope:
        if reuse: scope.reuse_variables()

        conv_w, conv_b = _conv_variable([7,7,3,64],name="conv1-e_g")# gai
        h = _conv2d(x,conv_w,stride=1) + conv_b
        h = tf.nn.leaky_relu(h)

        h = _conv_layer(h, 64, 64, 2, ksize, "2-1e_g")
        h = _conv_layer(h, 64, 128, 2, ksize,"1-2e_g")

        #h = _conv_layer(h, 128, 128, 1, ksize, "3-1e_g")
        h = _conv_layer(h, 128, 256, 2, ksize, "2-2e_g")

        for i in range(resBlock):
            conv_w, conv_b = _conv_variable([ksize,ksize,256,256],name="res%s-1" % i)
            nn = _conv2d(h,conv_w,stride=1) + conv_b
            nn = _instance_norm(h,name="Norm%s-1_g" %i)
            nn = tf.nn.leaky_relu(nn)
            conv_w, conv_b = _conv_variable([ksize,ksize,256,256],name="res%s-2" % i)
            nn = _conv2d(nn,conv_w,stride=1) + conv_b
            nn = _instance_norm(h,name="Norm%s-2_g" %i)

            nn = tf.math.add(h,nn, name="resadd%s" % i)
            h = nn


        h = _deconv_layer(h, 256, 128, 2, ksize, "2-2d_g")
        #h = _conv_layer(h, 128, 128, 1, ksize, "2-1_g")

        h = _deconv_layer(h, 128, 64, 2, ksize, "1-2d_g")
        h = _deconv_layer(h, 64, 64, 2, ksize, "1-1d_g")

        #h = tf.math.add(tmp,h, name="add1")
        conv_w, conv_b = _conv_variable([7,7,64,3],name="convo_g" )#gai
        h = _conv2d(h,conv_w,stride=1) + conv_b
        y = tf.nn.tanh(h)

    return y

def buildDiscriminator(y,reuse=False,isTraining=False,nBatch=16,ksize=4,name="discriminator"):
    with tf.variable_scope(name) as scope:
        if reuse: scope.reuse_variables()

        h =  y

        # conv1
        h = _conv_layer(h, 3, 64, 2, ksize, "1-1_d")#gai

        # conv2
        h = _conv_layer(h, 64, 128, 2, ksize, "2-1_d")

        # conv3
        h = _conv_layer(h, 128, 256, 2, ksize, "3-1_d")

        # conv4
        h = _conv_layer(h, 256, 512, 1, ksize, "4-1_d")

        # conv4
        conv_w, conv_b = _conv_variable([ksize,ksize,512,1],name="conv5_d")
        h = _conv2d(h,conv_w, stride=1) + conv_b

    return h
