import numpy as np
import scipy.misc
from keras.models import model_from_json
from matplotlib import pyplot as plt
import os
import subprocess
import time

class Classify:
   '''Creates an instance of Convolution Neutral Network classification model'''
   def __init__(self):
      self.num_to_abc = {0: 'A', 1:'B', 2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M' \
                        ,13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
      self.model = self.load_model('char74k_architecture1.json', 'char74k_weights1.h5')
      self.columns = 16
      self.rows = 16
      # keep track of board spaces that have already been classified. 
      # Keys of 0 have not been classified. Keys of 1 have been classified.
      self.classified_spaces = np.zeros((15,15))

   def load_model(self, model_def_fname, model_weight_fname):
      model = model_from_json(open(model_def_fname).read())
      model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
      model.load_weights(model_weight_fname)
      return model

   def load_and_scale_imgs(self, imgs_array):
      img = [np.transpose(scipy.misc.imresize(i, (32, 32)), 
         (2, 0, 1)).astype('float32') for i in imgs_array]
      img = np.array(img) / 255
      return img

   def classify(self):
      board = ['letter1.jpg', 'letter2.jpg', 'letter3.jpg', 'letter4.jpg', 'letter5.jpg', 'letter6.jpg','letter7.jpg','letter8.jpg','letter9.jpg','letter10.jpg','letter11.jpg','letter12.jpg']
      imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)),
                        (2, 0, 1)).astype('float32')
           for img_name in board]
      board_matrix = np.array(imgs)/255

      # load board matrix (15x15) and classify each letter placed on the board
      board_matrix = load_and_scale_imgs(board)

      for i in range(self.rows):
         for j in range(self.columns):
            if (self.classified_spaces[i][j] != 0):
               continue
            space = board_matrix[i][j]
            if (space != None):           
               prediction = self.model.predict_classes(space)
               board_matrix[i][j] = self.num_to_abc(prediction)
               self.classified_spaces[i][j] = 1
   
      return predict

classif = Classify()
start = time.time()
predict = classif.classify()
print predict
elap = time.time() - start
print elap
