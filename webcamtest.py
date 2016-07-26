import cv2
import numpy as np
c2 = cv2.VideoCapture(0)
import scipy.misc
from matplotlib import pyplot

def gettileimages(im):

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    cv2.imshow("hsv", im)

    lower_blue = np.array([40,125,0], dtype=np.uint8)
    #upper_blue = np.array([115,205,110], dtype=np.uint8) #night
    upper_blue = np.array([150,240,115], dtype=np.uint8) #day

    mask = cv2.inRange(im, lower_blue, upper_blue)

    cv2.imshow("Result1", mask)

    kernel = np.ones((2,2),np.uint8)
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations = 2)
    kernel = np.ones((3,3),np.uint8)
    morph2 = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel,iterations = 3)
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(morph2,kernel,iterations = 1)
    invmask = 255 - dilation

    cv2.imshow("Result2", invmask)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0;
    params.maxThreshold = 255;

    params.filterByConvexity = False;
    # params.minConvexity = 0.7;

    params.filterByInertia = False;
    # params.minInertiaRatio = 0.7;

    params.filterByCircularity = False;
    params.minCircularity = .65;

    params.filterByArea = True;
    params.minArea = 250;
    params.maxArea = 1700;
     
    # Set up the detector
    detector = cv2.SimpleBlobDetector_create(params)

    #detect block blobs
    keypoints = detector.detect(invmask)
    points = []
    for kp in keypoints:
        points.append(np.array([kp.pt[0],kp.pt[1]]))

    im = cv2.drawKeypoints(im, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('appebd',im)

    if len(points) < 4:
        return

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

    # pos = (points[0]+points[1]+points[2]+points[3])*.25
    # clone = im.copy()
    # new_im = clone[round(pos.item(1)):round(pos.item(1)+tilesize),round(pos.item(0)):round(pos.item(0)+tilesize)]

    rect = np.zeros((4, 2), dtype = "float32")
    # rect = np.hstack((tl, tr, br, bl))
    rect[0] = tl
    rect[1] = tr
    rect[2] = br
    rect[3] = bl

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
     
    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
     
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
     
    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
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

    cv2.waitKey(1000)
    origin = np.array([0,0])
    across = np.array([(maxWidth)/15.0,0.0])
    down = np.array([0.0,(maxHeight)/15.0])
    crops = []
    k = .006

    nameint = 0
    testint = 19
    for i in range(0,15):
        for j in range(0,15):
            clone = warp.copy()
            cv2.imshow("warp",clone)
            pos = origin + i*down + j*across
            new_im = clone[round(pos.item(1)):round(pos.item(1)+tilesizey),round(pos.item(0)):round(pos.item(0)+tilesizex)]
            crops.append(new_im)
            new_im2 = cv2.resize(new_im,(32,32), interpolation = cv2.INTER_LINEAR)
            black = 0
            not_black = 0
            bwimg = cv2.cvtColor(new_im2,cv2.COLOR_BGR2GRAY)
            lower_blue = np.array([0,0,0], dtype=np.uint8)
            upper_blue = np.array([110,110,110], dtype=np.uint8)
            bwmask = cv2.inRange(new_im2, lower_blue, upper_blue)
            bwmask = 255 - bwmask
            # bwmask = cv2.inRange(bwimg, 100,255)
            # kernel = np.ones((2,2),np.uint8)
            # bwmask = cv2.morphologyEx(bwmask, cv2.MORPH_CLOSE, kernel,iterations = 1)
            # kernel = np.ones((2,2),np.uint8)
            # bwmask = cv2.morphologyEx(bwmask, cv2.MORPH_OPEN, kernel,iterations = 1)
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
            # kernel = np.ones((2,5),np.uint8)
            # croppedmask2 = cv2.morphologyEx(croppedmask, cv2.MORPH_OPEN, kernel,iterations = 2)        
            # if (blackpercent < thresh and blackpercent > 0):
            # cv2.imshow("tile2",newwhitemask)
            start = 15
            if (i*15+j > start) and (i*15+j < start+13):
                save = cv2.cvtColor(croppedmask,cv2.COLOR_GRAY2RGB)
                cv2.imshow("TILE",croppedmask)
                nameint += 1
                #name = ('lettersdata/Sample%03d/' %nameint) + 'aletter'+ str(nameint)+ '-test' + str(testint) + '.jpg'
                classified = dl_classify.classify(save)
                print classified
                name = 'letter'+ str(nameint)+'.jpg'
                #scipy.misc.toimage(save, cmin=0.0, cmax=1.0).save(name)

                cv2.waitKey(100)
    cv2.waitKey(5000)
                

    # keys = [k for k in range(97,110)]
    # keys.append(32)
    # samples =  np.empty((0,400))
    # responses = []

    # for i in range(78,125):
    #     letter = crops[i].copy()
    #     ret,thresh = cv2.threshold(letter,127,255,0)
    #     contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #     max_index = len(contours) - 1
    #     while  (cv2.contourArea(contours[max_index]) < 100 or cv2.contourArea(contours[max_index]) > tilesize**2) and max_index != 0:
    #         max_index = max_index - 1
    #     cnt=contours[max_index]
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     cv2.rectangle(crops[i],(x,y),(x+w,y+h),(0,255,0),2)
    #     roi = letter[y:y+h,x:x+w]
    #     roismall = cv2.resize(roi,(20,20))
    #     cv2.imshow("TILE", roismall)
    #     key = 32
    #     if i < 86:
    #         key = i+19
    #     if 93 <= i < 101:
    #         key = i+12
    #     if 108 <= i < 116:
    #         key = i+5
    #     if 123 <= i < 131:
    #         key = i-2
    #     if key in keys:
    #         responses.append(key)
    #         sample = roismall.reshape((1,400))
    #         samples = np.append(samples,sample,0)

    # responses = np.array(responses,np.float32)
    # responses = responses.reshape((responses.size,1))

    # samples = np.float32(samples)
    # responses = np.float32(responses)

    # return samples, responses

while(1):
    _,f2 = c2.read()
    cv2.imshow('Logitech Cam',f2)
    gettileimages(f2)
    if cv2.waitKey(5)==27:
        break
cv2.destroyAllWindows()