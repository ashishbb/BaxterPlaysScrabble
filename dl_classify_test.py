import numpy as np
import scipy.misc
from keras.models import model_from_json
from matplotlib import pyplot as plt
import os
import subprocess
 
def load_and_scale_imgs():
   # img_names = ['happy-dog.jpg','subaru.jpg', 'standing-cat.jpg', 'vette.jpg','dog-face.jpg']

   img_names = ['letter1.jpg', 'letter2.jpg', 'letter3.jpg', 'letter4.jpg', 'letter5.jpg', 'letter6.jpg','letter7.jpg','letter8.jpg','letter9.jpg','letter10.jpg','letter11.jpg','letter12.jpg']
   print scipy.misc.imread(img_names[0]).shape
   imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)),
                        (2, 0, 1)).astype('float32')
           for img_name in img_names]



   # imgs = [np.transpose(scipy.misc.imresize(img, (32, 32)),
   #                       (2, 0, 1)).astype('float32')
   #          for img_name in imgs_array]

   img = np.array(imgs) / 255
   return img

def load_model(model_def_fname, model_weight_fname):
   model = model_from_json(open(model_def_fname).read())
   model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
   model.load_weights(model_weight_fname)
   return model

def classify():
   dict = {0: 'A', 1:'B', 2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M' \
   ,13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
   imgs = load_and_scale_imgs()
   model = load_model('char74k_architecture1.json', 'char74k_weights1.h5')
   print (model == None)




   predictions = model.predict_classes(imgs)
   predict = [dict[i] for i in predictions]
   print(predictions) 
   print(predict) 
   return predict

if __name__ == '__main__':
   classify()

