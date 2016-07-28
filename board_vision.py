import cv2
import numpy as np
import scipy.misc
import Classify

def get_board(cl,camera):

    ## find the 4 colored boxes around the rack, keep running until 4 boxes found
    camera = cv2.VideoCapture(0)
    print camera.isOpened()

    _,img = camera.read()
    # cv2.imshow('initial',img)
    points = get_green_box_points(camera)

    while not (len(points) == 4):
        points, img = get_green_box_points(camera)
        # for testing
        # cv2.imshow('not 4 green boxes',img)
        # cv2.waitKey(5)

    ## map the 4 colored points to the 4 corners of a rectangle
    camera.release()
    cv2.destroyAllWindows()

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
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
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

    ## for each of the 225 positions in the rack, cut out the part of the image with the letter inside it, apply morphological
    ## transformations, then call the character recognition code to determine the letter in the area
    board =[]
    for i in range(0,15):
        for j in range(0,15):
            clone = warp.copy()
            #cv2.imshow('clone',clone)
            #cv2.waitKey(20)
            pos = origin + i*down + j*across
            new_im = clone[round(pos.item(1)):round(pos.item(1)+tilesizey),round(pos.item(0)):round(pos.item(0)+tilesizex)]
            crops.append(new_im)
            new_im2 = cv2.resize(new_im,(32,32), interpolation = cv2.INTER_LINEAR)
            black = 0
            not_black = 0
            bwimg = cv2.cvtColor(new_im2,cv2.COLOR_BGR2GRAY)
            lower_blue = np.array([0,0,0], dtype=np.uint8)
            upper_blue = np.array([83,83,83], dtype=np.uint8)
            bwmask = cv2.inRange(new_im2, lower_blue, upper_blue)
            bwmask = 255 - bwmask
            #cv2.imshow('clone3',bwmask)
            #cv2.waitKey(20)
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
                if 40<cv2.contourArea(cnt)<475:
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
            # cv2.imshow('clone2',new_im2)
            # hsv = cv2.cvtColor(new_im2,cv2.COLOR_BGR2HSV)
            # cv2.imshow('clone7',hsv)
            # cv2.waitKey(5)
            if blackpercent < .4 and blackpercent > .04:
                if not_colored(new_im2):
                    save = cv2.cvtColor(croppedmask,cv2.COLOR_GRAY2RGB)
                    save = scipy.misc.toimage(save, cmin=0.0, cmax=1.0)
                    classified = cl.classify(save)
                    board.append(classified)
                else:
                    board.append(None)
            else:
                board.append(None)
            #name = 'letter'+ str(nameint)+'.jpg'
            #scipy.misc.toimage(save, cmin=0.0, cmax=1.0).save(name)
            cv2.waitKey(2)
    for i in range(15):
        print board[i*15:i*15+15]
    cv2.destroyAllWindows()
    return board

## check to see if the space is colored (if we can see the color of a square, it isn't covered by a tile, and therefore should be blank)
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
    if blackpercent > .05:
        print 'too red'
        return False
    lower_blue = np.array([110,90,15], dtype=np.uint8)
    upper_blue = np.array([165,125,8], dtype=np.uint8)
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
        print 'too blue'
        return False
    # night
    # lower_blue = np.array([120,115,145], dtype=np.uint8)
    # upper_blue = np.array([155,145,165], dtype=np.uint8)
    lower_blue = np.array([120,115,145], dtype=np.uint8)
    upper_blue = np.array([150,130,165], dtype=np.uint8)
    bwmask = cv2.inRange(im, lower_blue, upper_blue)
    bwmask = 255 - bwmask
    # cv2.imshow('clone2',bwmask)
    # cv2.waitKey(1000) 
    #thresh = .22
    black = 0
    for y in range(32):
        for x in range(32):
            pixel = bwmask[y][x]
            d = pixel
            if d == 0:
                black = black + 1
    blackpercent = float(black)/(32.0*32.0)
    if blackpercent > .04:
        print 'too pink'
        return False
    return True

def get_green_box_points(camera):
    _,im = camera.read()

    ###############SETTINGS FOR GREEN BOXES################
    # lower_blue = np.array([40,105,70], dtype=np.uint8)
    # upper_blue = np.array([140,205,170], dtype=np.uint8) #today
    
    lower_blue = np.array([10,105,0], dtype=np.uint8) #worked last night
    upper_blue = np.array([95,205,75], dtype=np.uint8) #worked last night

    # upper_blue = np.array([75,245,105], dtype=np.uint8) #day
    # upper_blue = np.array([95,205,77], dtype=np.uint8) #night
    cl = Classify.Classify()

    #masking and morphological transformations to find green boxes
    mask = cv2.inRange(im, lower_blue, upper_blue)
    #cv2.imshow('clone5',mask)
    # kernel = np.ones((2,2),np.uint8)
    # morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations = 2)
    kernel = np.ones((3,3),np.uint8)
    morph2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations = 5)
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(morph2,kernel,iterations = 1)
    invmask = 255 - dilation
    #cv2.imshow('clone1',invmask)

    #blob detecting for green boxes
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True;
    params.minArea = 175;
    params.maxArea = 1700;
    detector = cv2.SimpleBlobDetector_create(params)

    #get keypoints and store as nparray
    keypoints = detector.detect(invmask)
    points = []
    for kp in keypoints:
        points.append(np.array([kp.pt[0],kp.pt[1]]))

    im = cv2.drawKeypoints(im, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return (points , im)

# for testing
# cl = Classify.Classify()
# print 'YOURE RUNNING IT FROM THE IMPORT'
# get_board(cl,1)