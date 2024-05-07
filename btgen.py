import glob
import cv2
import numpy as np


class BatchGenerator:
    def __init__(self, img_size, imgdir, aug=True):
        self.folderPath = imgdir
        self.imagePath = glob.glob(self.folderPath+"/*")
        self.aug = aug
        #self.orgSize = (218,173)
        self.imgSize = (img_size,img_size)
        assert self.imgSize[0]==self.imgSize[1]

    def augment(self, img1):
        #軸反転
        if np.random.random() >0.5:
            img1 = cv2.flip(img1,1)

        #軸移動
        rand = (np.random.random()-0.5)/20
        y,x = img1.shape[:2]
        x_rate = x*(np.random.random()-0.5)/20
        y_rate = y*(np.random.random()-0.5)/20
        M = np.float32([[1,0,x_rate],[0,1,y_rate]])
        img1 = cv2.warpAffine(img1,M,(x,y),127)

        #回転
        rand = (np.random.random()-0.5)*5
        M = cv2.getRotationMatrix2D((x/2,y/2),rand,1)

        img1 = cv2.warpAffine(img1,M,(x,y))

        return img1

    def getBatch(self,nBatch,id,ocp=0.25):
        x   = np.zeros( (nBatch,self.imgSize[0],self.imgSize[1],3), dtype=np.float32)

        for i,j in enumerate(id):

            input = cv2.imread(self.imagePath[j])
            input = cv2.resize(input,self.imgSize)
            if self.aug:
                input = self.augment(input)
            x[i,:,:,:] = (input - 127.5) / 127.5

        return x
