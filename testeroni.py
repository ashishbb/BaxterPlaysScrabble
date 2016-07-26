import cv2


def get_one():
  cam2 = cv2.VideoCapture(0)
  _, im = cam2.read()
  return im

def get_two():
  cam1 = cv2.VideoCapture(1)
  _, im = cam1.read()
  return im

cv2.imshow('1',get_one())
cv2.waitKey(1000)
cv2.imshow('2',get_two())
cv2.waitKey(1000)