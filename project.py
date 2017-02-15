import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier

import cv2

def prepareTrainData(data):
    close_kernel = np.ones((5, 5), np.uint8)
    for i in range(0, len(data)):
        number = data[i].reshape(28, 28)
        th = cv2.inRange(number, 150, 255)
        closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_kernel)
        labeled = label(closing)
        regions = regionprops(labeled)
        if(len(regions) > 1):
            max_width = 0
            max_height = 0
            for region in regions:
                t_bbox = region.bbox
                t_width = t_bbox[3] - t_bbox[1]
                t_height = t_bbox[2] - t_bbox[0]
                if(max_width < t_width and max_height < t_height):
                    max_height = t_height
                    max_width = t_width
                    bbox = t_bbox
        else:
            bbox = regions[0].bbox
        img = np.zeros((28, 28))
        x = 0
        for row in range(bbox[0], bbox[2]):
            y = 0
            for col in range(bbox[1], bbox[3]):
                img[x, y] = number[row, col]
                y += 1
            x += 1
        data[i] = img.reshape(1, 784)

def jednacinaPrave(x, y):
    return x*(y2-y1) + y*(x1-x2) + (x2*y1-x1*y2)

def getNumberImage(bbox, img):
    min_row = bbox[0]
    min_col = bbox[1]
    width = bbox[3] - min_col
    height = bbox[2] - min_row
    img_number = np.zeros((28, 28))
    for x in range(0, height):
        for y in range(0, width):
            img_number[x, y] = img[min_row+x-1, min_col+y-1]
    return img_number

def presekSaPravom(bbox, height, width):
    #if(x1 > bbox[3] and x2 > bbox[3]):#Linija je desno od pravouganika
    #    return False
    #if(x1 < bbox[1] and x2 < bbox[1]):#Linija je levo od pravouganika
    #    return False
    #if(y1 > bbox[0]+height/2 and y2 > bbox[0]+height/2):#Linija je iznad pravouganika
    #    return False
    #if(y1 < bbox[2] and y2 < bbox[2]):#Linija je ispod pravouganika
    #    return False
    if(bbox[2]+4 < y2 or bbox[3]+4 < x1 or bbox[1] > x2):
        return False
    recEnd = jednacinaPrave(bbox[3]+1, bbox[2]+1)
    if(recEnd <= 0): #Ukoliko je kraj regiona ispod linije racunati kao da je ceo pravouganik presao liniju
        return False
    TL = jednacinaPrave(bbox[1], bbox[0])
    TR = jednacinaPrave(bbox[3]+4, bbox[0])
    BL = jednacinaPrave(bbox[1], bbox[2]+4)
    BR = jednacinaPrave(bbox[3]+4, bbox[2]+4)
    if(TL > 0 and TR > 0 and BL > 0 and BR > 0):
        return False
    elif(TL < 0 and TR < 0 and BL < 0 and BR < 0):
        return False
    else:
        return True

def addNumber(lista, broj, width, height, region, bbox):
    x=bbox[1]
    y=bbox[0]
    for tup in lista:
        if(tup[0] == broj and tup[1] < x+5  and tup[2] < y+5 and tup[3] == width):
            lista.remove(tup)
            lista.append((broj, x, y, width))
            return False
    lista.append((broj, x, y, width))

DIR = 'D:\\MARKO FAKULTET\\Cetvrta godina\\Soft Computing\\Projekat'
mnistFile = 'mnistPrepared'

mnist = fetch_mldata('MNIST original')
if(os.path.exists(os.path.join(DIR, mnistFile)+'.npy')):
    train = np.load(os.path.join(DIR, mnistFile)+'.npy')
else:    
    train = mnist.data
    prepareTrainData(train)
    np.save(os.path.join(DIR, mnistFile), train)
train_labels = mnist.target
knn = KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(train, train_labels)

eros_kernel = np.ones((2, 2), np.uint8)#Kernel koriscen prilikom erozije (priprema za skeleton operaciju)
close_kernel = np.ones((4, 4), np.uint8)
DIR = DIR+'\\Videos'
video_names = [os.path.join(DIR, name) for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
for vid_num in range(0, len(video_names)):
    print 'Putanja do videa: ' + video_names[vid_num]
    cap = cv2.VideoCapture(video_names[vid_num])
    frameNum = 0
    lista_brojeva=[]
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(frameNum%2 != 0):
            frameNum += 1
            continue
        if(ret == False):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if(frameNum == 0):
            line_th = cv2.inRange(gray, 4, 55)
            erosion = cv2.erode(line_th, eros_kernel, iterations=1)
            skeleton = skeletonize(erosion/255.0)
            cv_skeleton = img_as_ubyte(skeleton)
            lines = cv2.HoughLinesP(cv_skeleton, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            x1, y1, x2, y2 = lines[0][0]
        th = cv2.inRange(gray, 163, 255)
        closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_kernel)
        gray_labeled = label(closing)
        regions = regionprops(gray_labeled)
        for region in regions:
            bbox = region.bbox
            height = bbox[2]-bbox[0]
            width = bbox[3]-bbox[1]
            if(height <= 10):
                continue
            if(presekSaPravom(bbox, height, width) == False):
                continue
            img_number = getNumberImage(bbox, gray)
            num = int(knn.predict(img_number.reshape(1, 784)))
            #cv2.line(gray,(x1,y1),(x2,y2),(255,255,255),1)
            #plt.imshow(gray, 'gray')
            #plt.show()
            if(addNumber(lista_brojeva, num, width, height, region, bbox) == False):
                continue
            #print 'U frejmu '+ str(frameNum)+ '. prepoznat broj '+str(num)
        frameNum += 1
    suma=0
    for tup in lista_brojeva:
        suma += tup[0]
    print 'Suma: '+str(suma)+'\n'
cap.release()
cv2.destroyAllWindows()