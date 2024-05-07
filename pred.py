import os,sys,shutil
import tensorflow as tf

import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from model import *
DATASET_DIR = "data"
VAL_DIR ="test"
MODEL_DIR = "model"
OUT_DIR_A2B = "outA2B"
OUT_DIR_B2A = "outB2A"
def main(folder=["testA","testB"]):

    img_size = 256
    if not os.path.exists(OUT_DIR_A2B):
        os.makedirs(OUT_DIR_A2B)
    if not os.path.exists(OUT_DIR_B2A):
        os.makedirs(OUT_DIR_B2A)
    folderA2B = folder[0]
    filesA2B = os.listdir(folderA2B)
    folderB2A = folder[1]
    filesB2A = os.listdir(folderB2A)

    start = time.time()

    ar = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
    b = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
    a2b = buildGenerator(a, reuse=False, name="genA2B")
    b2a = buildGenerator(b, reuse=False, name="genB2A")

    sess = tf.Session()
    saver = tf.train.Saver()


    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    if ckpt: # checkpointがある場合
        #last_model = ckpt.all_model_checkpoint_paths[3]
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        print ("load " + last_model)

        saver.restore(sess, last_model) # 変数データの読み込み
        print("succeed restore model")

    else:
        print("checkpoints were not found.")
        print("saved model must exist in {}".format(MODEL_DIR))
        return

    print("%.4e sec took initializing"%(time.time()-start))

    start = time.time()
    #
    print("{} has {} files".format(folder[0], len(filesA2B)))
    for i in range(len(filesA2B)):

        img_path = "{}/{}".format(folderA2B,filesA2B[i])
        img = cv2.imread(img_path)
        img = (img-127.5)/127.5
        h,w = img.shape[:2]

        input = cv2.resize(img,(img_size,img_size))
        input= input.reshape(1,img_size,img_size,3)

        out = sess.run(a2b,feed_dict={a:input})
        out = out.reshape(img_size,img_size,3)
        out = cv2.resize(out,(w,h))
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        denorm_o = (out + 1) * 127.5
        cv2.imwrite(OUT_DIR_A2B+os.sep+'predicted' + image_name + '.png', denorm_o)
    print("{} has {} files".format(folder[1], len(filesB2A)))
    for i in range(len(filesB2A)):

        img_path = "{}/{}".format(folderB2A,filesB2A[i])
        img = cv2.imread(img_path)
        img = (img-127.5)/127.5
        h,w = img.shape[:2]

        input = cv2.resize(img,(img_size,img_size))
        input= input.reshape(1,img_size,img_size,3)

        out = sess.run(b2a,feed_dict={b:input})
        out = out.reshape(img_size,img_size,3)
        out = cv2.resize(out,(w,h))
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        denorm_o = (out + 1) * 127.5
        cv2.imwrite(OUT_DIR_B2A+os.sep+'predicted' + image_name + '.png', denorm_o)

    print("%.4e sec took for predicting" %(time.time()-start))

if __name__ == '__main__':
    folder = []
    folder.append("testA")
    folder.append("testB")
    try:
        folder[0] = sys.argv[1]
        folder[1] = sys.argv[2]
    except:
        pass
    print("folderA = {} folderB = {} ".format(folder[0],folder[1]))
    main(folder)
