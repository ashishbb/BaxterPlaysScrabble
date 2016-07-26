import os
import numpy as np
import scipy
import scipy.ndimage
from scipy import misc
import math

def load_dir_data(directory, channels, character_to_index, ch74_list):
    def load_image(path):
        return scipy.ndimage.imread(path, mode="RGB")
        if channels == 1:
            return scipy.ndimage.imread(path, mode="F") # grayscale
        elif channels == 3:
            return scipy.ndimage.imread(path, mode="RGB") # color
        else:
            raise Exception("Unknown number of image channels: "+channels)

    # X_train = np.zeros((1, 3, 32, 32), dtype="uint8")
    # y_train = np.zeros((1,), dtype="uint8")
    #print len(os.walk(directory))
    for d, _, files in os.walk(directory):
        print(d)
        for f in files:
            if f.endswith(".jpg") or f.endswith(".png"):
                # img = load_image(d+"/"+f) - 1
                print(d[18:21])
                character = ch74_list[int(d[18:21]) - 1]
                index = int(d[18:21]) - 1
                # img = scipy.misc.imresize(img, (32, 32))
                img = np.transpose(scipy.misc.imresize(load_image(d+"/"+f), (32, 32)),
                        (2, 0, 1)).astype('float32')
                img = img / 255.0
                #print img.shape
                yield img, int(index)

def load_data(directory):
    ch74kASCII = range(ord('A'), ord('Z') + 1)
        #range(ord('0'), ord('9') + 1) + \
        #range(ord('A'), ord('Z') + 1) + \
        # range(ord('a'), ord('z') + 1)
    ch74k_character_list = [chr(i) for i in ch74kASCII]
    data_directories = [f for f in os.listdir(directory) 
                            if os.path.isdir(directory+"/"+f)]
    
    character_list = [unichr(int(d[7:9])) for d in data_directories]
    character_to_index = {c:i for i, c in enumerate(character_list)}
    X_train = np.zeros((0, 3, 32, 32), dtype="uint8")
    y_train = np.zeros((0,), dtype="uint8")
    X_test = np.zeros((0, 3, 32, 32), dtype="uint8")
    y_test = np.zeros((0,), dtype="uint8")


    for d in os.listdir(directory):
    	print(d)
        if not os.path.isdir(directory+"/"+d): 
            continue
        #print d
        data = load_dir_data(directory+"/"+d, 
                3, character_to_index, ch74k_character_list)
        images, labels = zip(*data)
        split = int(math.floor(len(labels)*.9))
        X_addition_train = np.array(images[0:split-1], dtype='float32').reshape(
            [split-1, 3, 32, 32])
        X_addition_test = np.array(images[split:len(labels)], dtype='float32').reshape(
            [len(labels)-split, 3, 32, 32])
        X_train =  np.vstack((X_train,X_addition_train))
        X_test = np.vstack((X_test,X_addition_test))
        y_train =  np.hstack((y_train,np.array(labels[0:split-1], dtype='int32')))
        y_test =  np.hstack((y_test,np.array(labels[split:len(labels)], dtype='int32')))

    y_train = y_train[:,None]
    y_test = y_test[:,None]
    return (X_train, y_train), (X_test, y_test)

load_data("char74kdata")