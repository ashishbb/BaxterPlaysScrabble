import cv2
import numpy as np
import scipy.misc
import Classify

def get_rack(cl,cam2):

    cam2 = cv2.VideoCapture(1)
    print cam2.isOpened()

    points = get_orange_box_points(cam2)
    while not (len(points) == 4):
        points, im3 = get_orange_box_points(cam2)
        # cv2.imshow('not 4 green boxes',im3)
        # cv2.waitKey(8)

    cam2.release()
    cv2.destroyAllWindows()
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

    origin = np.array([12,0])
    across = np.array([(maxWidth)/8,0.0])

    rack = ''
    for i in range(0,7):
        clone = warp.copy()
        #cv2.imshow('tilesase',clone)
        #cv2.waitKey(90)
        pos = origin + i*across
        new_im = clone[round(pos.item(1)):round(pos.item(1)+tilesizey),round(pos.item(0))-12-2*i:round(pos.item(0)+tilesizex-8)]
        new_im3 = cv2.resize(new_im,(32,32), interpolation = cv2.INTER_LINEAR)
        #cv2.imshow('tilesave',new_im3)
        #cv2.waitKey(900)
        lower_blue = np.array([0,0,0], dtype=np.uint8)
        upper_blue = np.array([95,95,95], dtype=np.uint8)
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
        #cv2.imshow('tilesave',save)
        save = scipy.misc.toimage(save, cmin=0.0, cmax=1.0)

        classified = cl.classify(save)
        print classified
        rack = rack + classified
        cv2.waitKey(2)
    cam2.release()
    return rack

def get_orange_box_points(cam2):
    _,im = cam2.read()

    ###############SETTINGS FOR GREEN BOXES################
    # lower_blue = np.array([25,125,0], dtype=np.uint8)
    # #upper_blue = np.array([125,205,110], dtype=np.uint8) #night
    # upper_blue = np.array([138,240,113], dtype=np.uint8) #day
    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    # lower_blue = np.array([0,125,125], dtype=np.uint8) #night
    lower_blue = np.array([0,105,125], dtype=np.uint8)
    upper_blue = np.array([100,255,255], dtype=np.uint8) #day


    #masking and morphological transformations to find green boxes
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(im,im,mask = mask)


    lower_blue = np.array([25,125,0], dtype=np.uint8)
    upper_blue = np.array([185,250,130], dtype=np.uint8) #day


    #masking and morphological transformations to find green boxes
    mask = cv2.inRange(res, lower_blue, upper_blue)
    kernel = np.ones((2,2),np.uint8)
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations = 5)
    kernel = np.ones((3,3),np.uint8)
    morph2 = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel,iterations = 3)
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(morph2,kernel,iterations = 1)
    invmask = 255 - dilation
    # cv2.imshow('inv',invmask)
    # cv2.waitKey(20)

    #blob detecting for green boxes
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True;
    params.minArea = 300;
    params.maxArea = 2500;
    detector = cv2.SimpleBlobDetector_create(params)

    #get keypoints and store as nparray
    keypoints = detector.detect(invmask)
    points = []
    for kp in keypoints:
        points.append(np.array([kp.pt[0],kp.pt[1]]))

    im = cv2.drawKeypoints(im, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return (points , im)

# cl = Classify.Classify()
# print 'YOURE RUNNING IT FROM THE IMPORT'
# get_rack(cl,1)
