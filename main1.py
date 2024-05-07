import os,sys,shutil
import tensorflow  as tf
import scipy.misc as misc
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from model1 import *
from btgen import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
A_DIR = "./cityscapes/train/A"
B_DIR = "./cityscapes/train/B"
valA_DIR ="./cityscapes/val/A"
valB_DIR = "./cityscapes/val/B"
SAVE_DIR = "model2"
SVIM_DIR = "samples2"


def tileImage(imgs):
    d = int(math.sqrt(imgs.shape[0]-1))+1
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.zeros((h*d,w*d,3),dtype=np.float32)
    for idx,img in enumerate(imgs):
        idx_y = int(idx/d)
        idx_x = idx-idx_y*d
        r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
    return r

def foloderLength(folder):
    dir = folder
    paths = os.listdir(dir)
    return len(paths)

def printParam(scope):
    total_parameters = 0
    for variable in tf.trainable_variables(scope=scope):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("{} has {} parameters".format(scope, total_parameters))

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(SVIM_DIR):
        os.makedirs(SVIM_DIR)
    img_size = 256
    bs = 4
    lmd = 100

    # loading images on training
    imgenA = BatchGenerator(img_size=img_size, imgdir=A_DIR)
    imgenB = BatchGenerator(img_size=img_size, imgdir=B_DIR)
    vlgenA = BatchGenerator(img_size=img_size, imgdir=valA_DIR, aug=False)
    vlgenB = BatchGenerator(img_size=img_size, imgdir=valB_DIR, aug=False)

    lenA = foloderLength(A_DIR)
    lenB = foloderLength(B_DIR)
    vlenA = foloderLength(valA_DIR)
    vlenB = foloderLength(valB_DIR)

    #print(lenA,lenB,vlenA,vlenB)

    # sample images
    id = np.random.choice(range(lenA),bs)
    _A  = imgenA.getBatch(bs,id)#[0]
    id = np.random.choice(range(lenB),bs)
    _B  = imgenB.getBatch(bs,id)#[0]
    _Z = np.concatenate([_A,_B],axis=1)
    _Z = (_Z + 1)*127.5
    cv2.imwrite("input.png",_Z)

    #build models
    start = time.time()

    a = tf.placeholder(tf.float32, [bs, img_size, img_size, 3])


    b = tf.placeholder(tf.float32, [bs, img_size, img_size, 3])


    lr = tf.placeholder(tf.float32)

    a2b = buildGenerator(a, reuse=False, nBatch=bs, name="genA2B")

    b2a = buildGenerator(b, reuse=False, nBatch=bs, name="genB2A")

    a2b2a = buildGenerator(a2b, reuse=True, nBatch=bs, name="genB2A")

    b2a2b = buildGenerator(b2a, reuse=True, nBatch=bs, name="genA2B")



    dis_a_fake = buildDiscriminator(b2a, nBatch=bs, reuse=False, name="disA")
    dis_a_real = buildDiscriminator(a, nBatch=bs, reuse=True, name="disA")
    dis_b_fake = buildDiscriminator(a2b, nBatch=bs, reuse=False, name="disB")
    dis_b_real = buildDiscriminator(b, nBatch=bs, reuse=True, name="disB")

    # d_loss
    a_loss_real = tf.reduce_mean((dis_a_real-tf.ones_like (dis_a_real))**2)
    a_loss_fake = tf.reduce_mean((dis_a_fake-tf.zeros_like (dis_a_fake))**2)
    b_loss_real = tf.reduce_mean((dis_b_real-tf.ones_like (dis_b_real))**2)
    b_loss_fake = tf.reduce_mean((dis_b_fake-tf.zeros_like (dis_b_fake))**2)


    #g_loss
    b2a_loss    = tf.reduce_mean((dis_a_fake-tf.ones_like (dis_a_fake))**2)
    a2b_loss    = tf.reduce_mean((dis_b_fake-tf.ones_like (dis_b_fake))**2)

    wd_a2b = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="genA2B")
    wd_b2a = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="genB2A")
    wd_disA = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="disA")
    wd_disB = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="disB")

    wd_a2b = tf.reduce_sum(wd_a2b)
    wd_b2a = tf.reduce_sum(wd_b2a)
    wd_disA = tf.reduce_sum(wd_disA)
    wd_disB = tf.reduce_sum(wd_disB)

    a2b_cycle_loss = tf.reduce_mean(tf.abs(a - a2b2a)) 
    b2a_cycle_loss = tf.reduce_mean(tf.abs(b - b2a2b))

    cycle_loss = (a2b_cycle_loss + b2a_cycle_loss)/2

    genA2B_loss = a2b_loss + lmd * cycle_loss + wd_a2b
    genB2A_loss = b2a_loss + lmd * cycle_loss + wd_b2a
    disA_loss = a_loss_real + a_loss_fake + wd_disA
    disB_loss = b_loss_real + b_loss_fake + wd_disB

    genA2B_opt = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(genA2B_loss,
                    var_list=[x for x in tf.trainable_variables() if "genA2B" in x.name])
    genB2A_opt = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(genB2A_loss,
                    var_list=[x for x in tf.trainable_variables() if "genB2A" in x.name])
    disA_opt = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(disA_loss,
                    var_list=[x for x in tf.trainable_variables() if "disA" in x.name])
    disB_opt = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(disB_loss,
                    var_list=[x for x in tf.trainable_variables() if "disB" in x.name])



    printParam(scope="genA2B")
    printParam(scope="genB2A")
    printParam(scope="disA")
    printParam(scope="disB")

    print("%.3f sec took building model"%(time.time()-start))

    start = time.time()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7))

    sess =tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    summary = tf.summary.merge_all()

    ckpt = tf.train.get_checkpoint_state('model2')

    if ckpt: 
        last_model = ckpt.model_checkpoint_path 
        print ("load " + last_model)
        saver.restore(sess, last_model) 
        print("succeed restore model")
    else:
        print("models were not found")
        init = tf.global_variables_initializer()
        sess.run(init)

    print("%.3f sec took initializing"%(time.time()-start))

    a2b_hist = []
    b2a_hist = []
    da_hist = []
    db_hist = []

    start = time.time()

    trans_lr = 2e-4

    for i in range(100001):
        # loading images on training
        id = np.random.choice(range(lenA),bs)
        batch_images_a  = imgenA.getBatch(bs,id)
        id = np.random.choice(range(lenB),bs)
        batch_images_b  = imgenB.getBatch(bs,id)

        tmp, tmp, dis_lossA, dis_lossB = \
            sess.run([disA_opt,disB_opt,disA_loss,disB_loss], feed_dict={
            a: batch_images_a,
            b: batch_images_b,
            lr: trans_lr
        })

        tmp, tmp, gen_lossA2B, gen_lossB2A = \
            sess.run([genA2B_opt,genB2A_opt,genA2B_loss,genB2A_loss], feed_dict={
            a: batch_images_a,
            b: batch_images_b,
            lr: trans_lr
        })

        trans_lr = trans_lr * 0.99998



        print("in step %s, disA_loss = %.3f, disB_loss = %.3f, genA2B_loss = %.3f, genB2A_loss = %.3f"
            %(i,dis_lossA,dis_lossB, gen_lossA2B, gen_lossB2A))
        da_hist.append(dis_lossA)
        db_hist.append(dis_lossB)
        a2b_hist.append(gen_lossA2B)
        b2a_hist.append(gen_lossB2A)


        if i %100 ==0:

            id = np.random.choice(range(vlenA),1)
            batch_images_a  = vlgenA.getBatch(bs,id)
            id = np.random.choice(range(vlenB),1)
            batch_images_b  = vlgenB.getBatch(bs,id)
            _A2B = sess.run(a2b,feed_dict={
                a: batch_images_a})#[:1]
            _A2B2A = sess.run(b2a,feed_dict={
                b:_A2B})#[:1]
            _B2A = sess.run(b2a,feed_dict={
                b: batch_images_b})#[:1]
            _B2A2B = sess.run(a2b,feed_dict={
                a:_B2A})#[:1]

            _A2B = np.concatenate([batch_images_a[0],_A2B[0],_A2B2A[0]],axis=1)
            _B2A = np.concatenate([batch_images_b[0],_B2A[0],_B2A2B[0]],axis=1)

            _Z = np.concatenate([_A2B,_B2A],axis=0)
            _Z = ( _Z + 1) * 127.5

            cv2.imwrite("%s/%s.png"%(SVIM_DIR, i),_Z)
            fig = plt.figure(figsize=(8,6), dpi=128)
            ax = fig.add_subplot(111)
            plt.title("Loss")
            plt.grid(which="both")
            plt.yscale("log")
            ax.plot(a2b_hist,label="a2b_loss", linewidth = 0.25)
            ax.plot(b2a_hist,label="b2a_loss", linewidth = 0.25)
            ax.plot(a2b_hist,label="a2b_loss", linewidth = 0.25)
            ax.plot(b2a_hist,label="b2a_loss", linewidth = 0.25)
            plt.xlabel('step', fontsize = 16)
            plt.ylabel('loss', fontsize = 16)
            plt.legend(loc='upper left')
            plt.savefig("histGAN1.png")
            plt.close()

            print("%.3f sec took 100steps" %(time.time()-start))
            start = time.time()

        if i%500==0 and i!=0:
            saver.save(sess,os.path.join(SAVE_DIR,"model.ckpt"),i)
    sess.close()

if __name__ == '__main__':
    main()
