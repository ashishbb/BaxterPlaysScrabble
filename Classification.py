import numpy as np
import scipy.misc
from keras.models import model_from_json
from matplotlib import pyplot as plt
import os
import subprocess
import time

class CNN_Model:
   '''Creates an instance of Convolution Neutral Network classification model'''
   def __init__(self):
      self.num_to_abc = {0: 'A', 1:'B', 2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M' \
                        ,13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
      self.model = self.load_model('char74k_architecture.json', 'char74k_weights.h5')

   def load_model(self, model_def_fname, model_weight_fname):
      model = model_from_json(open(model_def_fname).read())
      model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
      model.load_weights(model_weight_fname)
      return model

   def scale_img(self, i):
      scaled_img = np.transpose(scipy.misc.imresize(i, (32, 32)), 
                  (2, 0, 1)).astype('float32')
      scaled_img = np.array(scaled_img) / 255
      return scaled_img.reshape(1,3,32,32)

   def classify(self,img):
      # load image and rescale it
      scaled_img = self.scale_img(img)
      prediction = self.model.predict_classes(scaled_img)[0]
      return self.num_to_abc[prediction]
