import numpy as np
import scipy.misc
from keras.models import model_from_json
from matplotlib import pyplot as plt
import os
import subprocess
 
def load_and_scale_imgs(imgs_array):
   # img_names = ['happy-dog.jpg','subaru.jpg', 'standing-cat.jpg', 'vette.jpg','dog-face.jpg']
   #img_names = ['m.jpg']
   #A,'K.jpg', 'O.jpg', 'C.jpg', '2.jpg', '6.jpg','M.jpg'
   #print scipy.misc.imread(img_names[0]).shape
   # imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)),
   #                      (2, 0, 1)).astype('float32')
   #         for img_name in img_names]

   imgs = [np.transpose(scipy.misc.imresize(img, (32, 32)),
                         (2, 0, 1)).astype('float32')
            for img in imgs_array]
   #print a.shape
   # print np.array(imgs)/255

   imgs = np.array(imgs) / 255
   print img.shape
   # view_eg = img[3:4].reshape(32,32,3)
   # print(view_eg.shape)
   # plt.imshow(view_eg.tolist(), interpolation="nearest")
   # plt.show()
   return imgs

def load_model(model_def_fname, model_weight_fname):
   model = model_from_json(open(model_def_fname).read())
   model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
   model.load_weights(model_weight_fname)
   return model

def classify(imgs_array):
   dict = {0: 'A', 1:'B', 2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M' \
   ,13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
   imgs = load_and_scale_imgs(imgs_array)
   model = load_model('char74k_architecture.json', 'char74k_weights.h5')
   predictions = model.predict_classes(imgs)
   predict = [dict[i] for i in predictions]
   print(predictions)  
   #print(predict)
   return predict


def handle_blanks(board):
   classify_arr = []
   first_blank = 0
   for i in range(0,226):
      if (space == None):
         non_spaces = board[first_blank:]


# if __name__ == '__main__':
#    classify(imgs_array)

