#!/usr/bin/env python

#import sys
#import rospy
import numpy as np
import math
#import baxter_interface
import time
import scipy.misc

from board import Board
from dictionary import Dictionary
from bag import generate_rack, get_full_bag

import cv2
import speech
import time
import dictionary
import direction
import board
import solution
import scrabblerack
from Classification import CNN_Model

DICTIONARY_FILENAME = "dictionary"

def main():

    # Load the dictionary.
    dictionary = Dictionary.load(DICTIONARY_FILENAME)
    board = Board()

    # Keep track of the winning solution at each round.
    winners = []

    # List of letters we can still pick from.
    bag = get_full_bag()

    # Rack starts out empty. Keep track of current and last rack state.
    rack = ""
    old_rack = ""
    count = 0

    # Keep track of current and last board state,
    update_board = None
    old_board = None

    # Baxter's score
    my_score = 0

    #Create classifier
    classify = CNN_Model()

    # Create video feeds
    # cam = cv2.VideoCapture(1)
    # print cam.isOpened()

    # Keep playing until we're out of tiles or solutions.
    while count < 8:
        count+=1
        # Fill up our rack.
        print "Bag: %s" % "".join(bag)
        old_rack = rack

        # Updates rack with current rack from video feed.
        # cam1 = cv2.VideoCapture(1)
        # print cam1.isOpened()
        cam1 = 1
        rack = get_rack(classify,cam1)
        # cam1.release()
        cv2.destroyAllWindows()

        # Get a list of possible solutions. These aren't all necessarily legal.
        solutions = board.generate_solutions(rack, dictionary)

        solution = board.find_best_solution(solutions, dictionary)

        if solution:
            print "Winner: %s" % solution

            # Play the winning solution.
            board.create_board()
            print("I suggest you play the word:"+solution.word)
            #speech.speak(solution)
        else:
            # Should put letters back in bag.
            break
        print board
        
        # Wait for "Enter" press, signifying the player has completed his/her turn.
        #wait = raw_input("Press enter when finished with move")

        # Get word that was just played on the board by fetching the new board state.
        # cam1.release()
        # del cam1
        # cam1 = cv2.VideoCapture(0)
        # while not cam1.isOpened():
        #     cam1.release()
        #     del cam1
        #     cam1 = cv2.VideoCapture(0)

        update_board = get_board(classify,cam1)
        # cam1.release()
        
        move,letter_placed_on_board = board.get_played_word(update_board,old_board,dictionary)
        print ("The word"+move+"was just played")

        if (move == solution.word):
            print("Player listened to Baxter")
        else:
            print("defied Baxter")




        print "Baxter's Score: %d" % my_score

        generate_rack(rack,old_rack,bag)

        for char in letter_placed_on_board:
            rack = rack.replace(char,"")

    print "Baxter's Score: %d" % my_score
    print "Baxter's Words:"
    for rack, winner in winners:
        print "    %s: %s" % (rack, winner)

def get_board(cl,camera):

    camera = cv2.VideoCapture(0)
    print camera.isOpened()

    _,im = camera.read()
    cv2.imshow('initial',im)
    cv2.waitKey(50)
    points = get_green_box_points(camera)

    while not (len(points) == 4):
        points, img = get_green_box_points(camera)
        cv2.imshow('not 4 green boxes',img)
        cv2.waitKey(100)

    camera.release()

    xmin = 0
    for pt in points:
        if pt.item(0) > 250:
            if pt.item(1) < 250:
                tr = pt
            if pt.item(1) > 250:
                br = pt
        if pt.item(0) < 250:
            if pt.item(1) < 250:
                tl= pt
            if pt.item(1) > 250:
                bl= pt

    rect = np.zeros((4, 2), dtype = "float32")
    rect[0] = tl
    rect[1] = tr
    rect[2] = br
    rect[3] = bl

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
     
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(im, M, (maxWidth, maxHeight))
    mask = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations = 1)
    kernel = np.ones((2,2),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations = 1)
    mask = cv2.inRange(mask, 0, 65)
    mask = 255 - mask

    maxHeight, maxWidth, channels = warp.shape

    tilesizex = maxWidth/15.0
    tilesizey = maxHeight/15.0

    origin = np.array([0,0])
    across = np.array([(maxWidth)/15.0,0.0])
    down = np.array([0.0,(maxHeight)/15.0])
    crops = []
    k = .006

    board =[]
    for i in range(0,15):
        for j in range(0,15):
            clone = warp.copy()
            cv2.imshow('clone',clone)
            pos = origin + i*down + j*across
            new_im = clone[round(pos.item(1)):round(pos.item(1)+tilesizey),round(pos.item(0)):round(pos.item(0)+tilesizex)]
            crops.append(new_im)
            new_im2 = cv2.resize(new_im,(32,32), interpolation = cv2.INTER_LINEAR)
            black = 0
            not_black = 0
            bwimg = cv2.cvtColor(new_im2,cv2.COLOR_BGR2GRAY)
            lower_blue = np.array([0,0,0], dtype=np.uint8)
            upper_blue = np.array([115,115,115], dtype=np.uint8)
            bwmask = cv2.inRange(new_im2, lower_blue, upper_blue)
            bwmask = 255 - bwmask
            thresh = .22
            for y in range(32):
                for x in range(32):
                    pixel = bwmask[y][x]
                    d = pixel
                    if d == 0:
                        black = black + 1
                    else:
                        not_black = not_black +1
            blackpercent = float(black)/(32.0*32.0)
            _, contours, hier = cv2.findContours(bwmask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            newwhitemask = np.ones((32,32),np.float32)
            newwhitemask = newwhitemask * 255
            x=0
            y=0
            h=32
            w=32
            biggest = 0
            for cnt in contours:
                if 35<cv2.contourArea(cnt)<475:
                    if cv2.contourArea(cnt) > biggest:
                        biggest = cv2.contourArea(cnt)
                        (x,y,w,h) = cv2.boundingRect(cnt)
                    cv2.drawContours(new_im2,[cnt],0,(0,255,0),2)
            newwhitemask[y:y+h,x:x+w] = bwmask[y:y+h,x:x+w]
            kernel = np.ones((1,2),np.uint8)
            newwhitemask = cv2.morphologyEx(newwhitemask, cv2.MORPH_OPEN, kernel,iterations = 1) 
            extra = newwhitemask.copy()
            croppedmask = newwhitemask[y:y+h,x:x+w]
            croppedmask = cv2.resize(croppedmask,(64,64), interpolation = cv2.INTER_LINEAR)
            cv2.imshow('clone2',croppedmask) 
            if blackpercent < .4 and blackpercent > .07:
                if not_colored(new_im2):
                    save = cv2.cvtColor(croppedmask,cv2.COLOR_GRAY2RGB)
                    save = scipy.misc.toimage(save, cmin=0.0, cmax=1.0)
                    classified = cl.classify(save)
                    board.append(classified)
                else:
                    board.append(' ')
            else:
                board.append(' ')
            cv2.waitKey(40)
            #name = 'letter'+ str(nameint)+'.jpg'
            #scipy.misc.toimage(save, cmin=0.0, cmax=1.0).save(name)

                #cv2.waitKey(500)
    for i in range(15):
        print board[i*15:i*15+15]

def not_colored(im):
    lower_red = np.array([0,0,80], dtype=np.uint8)
    upper_red = np.array([60,60,200], dtype=np.uint8)
    bwmask = cv2.inRange(im, lower_red, upper_red)
    bwmask = 255 - bwmask
    #thresh = .22
    black = 0
    for y in range(32):
        for x in range(32):
            pixel = bwmask[y][x]
            d = pixel
            if d == 0:
                black = black + 1
    blackpercent = float(black)/(32.0*32.0)
    if blackpercent > .1:
        return False
    lower_blue = np.array([110,90,15], dtype=np.uint8)
    upper_blue = np.array([165,120,85], dtype=np.uint8)
    bwmask = cv2.inRange(im, lower_blue, upper_blue)
    bwmask = 255 - bwmask
    #thresh = .22
    black = 0
    for y in range(32):
        for x in range(32):
            pixel = bwmask[y][x]
            d = pixel
            if d == 0:
                black = black + 1
    blackpercent = float(black)/(32.0*32.0)
    if blackpercent > .1:
        return False
    return True

def get_green_box_points(camera):
    _,im = camera.read()

    ###############SETTINGS FOR GREEN BOXES################
    lower_blue = np.array([40,125,0], dtype=np.uint8)
    #upper_blue = np.array([115,205,110], dtype=np.uint8) #night
    upper_blue = np.array([150,240,115], dtype=np.uint8) #day

    #masking and morphological transformations to find green boxes
    mask = cv2.inRange(im, lower_blue, upper_blue)
    kernel = np.ones((2,2),np.uint8)
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations = 2)
    kernel = np.ones((3,3),np.uint8)
    morph2 = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel,iterations = 3)
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(morph2,kernel,iterations = 1)
    invmask = 255 - dilation

    #blob detecting for green boxes
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True;
    params.minArea = 150;
    params.maxArea = 1700;
    detector = cv2.SimpleBlobDetector_create(params)

    #get keypoints and store as nparray
    keypoints = detector.detect(invmask)
    points = []
    for kp in keypoints:
        points.append(np.array([kp.pt[0],kp.pt[1]]))

    im = cv2.drawKeypoints(im, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return (points , im)

def get_rack(cl,cam2):

    cam2 = cv2.VideoCapture(1)
    print cam2.isOpened()

    _, im3 = cam2.read()
    points = get_orange_box_points(cam2)

    while not (len(points) == 4):
        points, im3 = get_orange_box_points(cam2)
        #cv2.imshow('not 4 green boxes',im3)
        cv2.waitKey(80)

    cam2.release()
    xmid = 0.0
    ymid = 0.0
    for pt in points:
        xmid += pt.item(0)
        ymid += pt.item(1)
    xmid /= 4
    ymid /= 4
    for pt in points:
        if pt.item(0) > xmid:
            if pt.item(1) < ymid:
                tr = pt
            if pt.item(1) > ymid:
                br = pt
        if pt.item(0) < xmid:
            if pt.item(1) < ymid:
                tl= pt
            if pt.item(1) > ymid:
                bl= pt

    rect = np.zeros((4, 2), dtype = "float32")
    rect[0] = tl
    rect[1] = tr
    rect[2] = br
    rect[3] = bl

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
     
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(im3, M, (maxWidth, maxHeight))
    mask = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations = 1)
    kernel = np.ones((2,2),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations = 1)
    mask = cv2.inRange(mask, 0, 65)
    mask = 255 - mask

    maxHeight, maxWidth, channels = warp.shape

    tilesizex = maxWidth/8
    tilesizey = maxHeight

    origin = np.array([0,0])
    across = np.array([(maxWidth)/8,0.0])

    rack = ''
    for i in range(0,7):
        clone = warp.copy()
        pos = origin + i*across
        new_im = clone[round(pos.item(1))+8:round(pos.item(1)+tilesizey)-2,round(pos.item(0)):round(pos.item(0)+tilesizex)+4]
        new_im3 = cv2.resize(new_im,(32,32), interpolation = cv2.INTER_LINEAR)
        lower_blue = np.array([0,0,0], dtype=np.uint8)
        upper_blue = np.array([105,105,105], dtype=np.uint8)
        bwmask = cv2.inRange(new_im3, lower_blue, upper_blue)
        bwmask = 255 - bwmask
        _, contours, hier = cv2.findContours(bwmask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        newwhitemask = np.ones((32,32),np.float32)
        newwhitemask = newwhitemask * 255
        x=0
        y=0
        h=32
        w=32
        biggest = 0
        for cnt in contours:
            if 35<cv2.contourArea(cnt)<475:
                if cv2.contourArea(cnt) > biggest:
                    biggest = cv2.contourArea(cnt)
                    (x,y,w,h) = cv2.boundingRect(cnt)
                cv2.drawContours(new_im3,[cnt],0,(0,255,0),2)
        newwhitemask[y:y+h,x:x+w] = bwmask[y:y+h,x:x+w]
        kernel = np.ones((1,2),np.uint8)
        newwhitemask = cv2.morphologyEx(newwhitemask, cv2.MORPH_OPEN, kernel,iterations = 1) 
        extra = newwhitemask.copy()
        croppedmask = newwhitemask[y:y+h,x:x+w]
        croppedmask = cv2.resize(croppedmask,(64,64), interpolation = cv2.INTER_LINEAR)
        #cv2.imshow('tilesave',croppedmask)
        save = cv2.cvtColor(croppedmask,cv2.COLOR_GRAY2RGB)
        save = scipy.misc.toimage(save, cmin=0.0, cmax=1.0)
        classified = cl.classify(save)
        print classified
        rack = rack + classified
        cv2.waitKey(40)
    cam2.release()
    return rack

def get_orange_box_points(cam2):
    _,im = cam2.read()

    ###############SETTINGS FOR GREEN BOXES################
    lower_blue = np.array([40,125,0], dtype=np.uint8)
    upper_blue = np.array([125,205,110], dtype=np.uint8) #night
    #upper_blue = np.array([150,240,115], dtype=np.uint8) #day

    #masking and morphological transformations to find green boxes
    mask = cv2.inRange(im, lower_blue, upper_blue)
    kernel = np.ones((2,2),np.uint8)
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations = 2)
    kernel = np.ones((3,3),np.uint8)
    morph2 = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel,iterations = 3)
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(morph2,kernel,iterations = 1)
    invmask = 255 - dilation

    #blob detecting for green boxes
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True;
    params.minArea = 300;
    params.maxArea = 1700;
    detector = cv2.SimpleBlobDetector_create(params)

    #get keypoints and store as nparray
    keypoints = detector.detect(invmask)
    points = []
    for kp in keypoints:
        points.append(np.array([kp.pt[0],kp.pt[1]]))

    im = cv2.drawKeypoints(im, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return (points , im)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass