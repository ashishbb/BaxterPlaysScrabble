#!/usr/bin/env python

import sys
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import math
#import boardreader
#import boardtester
import scrabblecv
import collections
from std_msgs.msg import String

bridge = None
globalim = None
globalflag = False
samples = None
responses = None
globalstring = ''
clearflag = False
rack = ''

def boardtrainer(newrack):
    #initialize node
    rospy.init_node('scrabblerack')
    rospy.Subscriber("cameras/left_hand_camera/image",Image,imagecallback)
    global bridge
    bridge = CvBridge()
    global samples, responses
    samples =  np.empty((0,400),np.float32)
    responses = []
    # raw_input("nextsample")
    # boardtester.testimage(globalim)
    #while not rospy.is_shutdown():
    global responses, samples, rack
    # while not rospy.is_shutdown():
    global globalstring
    globalstring = ''
    rospy.Rate(.05).sleep()
    mostcommon = None
    mostcommon = collections.Counter(globalstring).most_common(20)
    print mostcommon
    racknum = 0
    newrack = ''
    for idx, string in enumerate(mostcommon):
        if (65 <= ord(string[0]) <= 90 or ord(string[0]) == 48) and racknum < 7:
            if (ord(string[0]) == 48):
                mostcommon = (string[0],'O')
            newrack =newrack + string[0]
            racknum = racknum + 1
            if string[1] > 140 and string[1] < 220 and racknum < 7:
                newrack = rack + string[0]
                racknum = racknum + 1
            if string[1] > 220 and racknum < 6:
                newrack = newrack + string[0]
                racknum = racknum + 1
                newrack = newrack + string[0]
                racknum = racknum + 1
    global clearflag
    clearflag = True
    global globalstring
    globalstring = ''
    oldrack = rack
    rack = ''
    return newrack

    # raw_input("begin")
    # for i in range (1,50):
    #     im = globalim
    #     if boardreader.getlen(im) == 4:
    #         newsamps, newresps = boardreader.gettileimages(im)
    #         responses.append(newresps)
    #         samples = np.append(samples,newsamps,0)
    #     else:
    #         print("Green not recognized")
    # np.savetxt('src/baxterscrabble/scripts/generalsamples.data',samples)
    # np.savetxt('src/baxterscrabble/scripts/generalresponses.data',responses)

def imagecallback(data):
    try:
        global globalim, globalstring, clearflag
        globalim = bridge.imgmsg_to_cv2(data, "bgr8")
        cv2.imshow("Result", globalim)
        globalstring = globalstring + scrabblecv.getrack(globalim)
        if clearflag:
            globalstring = ''
            clearflag = False
        cv2.waitKey(3)
        # if not globalflag and not globalim == None:
        #     global globalflag
        #     boardreader.gettileimages(globalim,samples,responses)
        #     globalflag = True
    except CvBridgeError as e:
        print(e)

def get_rack():
    # while rack == '':
    #     x = 1
    return rack

if __name__ == "__main__":
    try:
        boardtrainer()
    except rospy.ROSInterruptException:
        pass