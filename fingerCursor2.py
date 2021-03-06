#!/usr/bin/env python3

# w/ openCV3, version 3.3.0

import numpy as np
import cv2
from collections import deque
import imutils
from sklearn.metrics import pairwise

def dist(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5

def count(thresholded, segmented):
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * maximum_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # take out the circular region of interest which has 
    # the palm and the fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    (_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initalize the finger count
    count = 0

    # loop through the contours found
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count


def fingerCursor(device):
    cap = cv2.VideoCapture(device)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # print(cap_height,cap_width)

    ## gesture matching initialization
    gesture2 = cv2.imread('gesture2.png')
    gesture2 = cv2.cvtColor(gesture2, cv2.COLOR_BGR2GRAY)
    _, gesture2 , _ = cv2.findContours(gesture2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    ## skin color segmentation mask
    skin_min = np.array([0, 40, 50],np.uint8)  # HSV mask
    skin_max = np.array([50, 250, 255],np.uint8) # HSV mask

    ## trajectory drawing initialization
    topmost_last = (200,100)    # initial position of finger cursor
    traj = np.array([], np.uint16)
    traj = np.append( traj, topmost_last)
    dist_pts = 0
    dist_records = [dist_pts]

    ## finger cursor position low_pass filter
    low_filter_size = 5
    low_filter = deque([topmost_last,topmost_last,topmost_last,topmost_last,topmost_last],low_filter_size )  # filter size is 5

    ## gesture_index low_pass filter
    gesture_filter_size = 10
    gesture_matching_filter = deque([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], gesture_filter_size )
    gesture_index_thres = 0

    ## color definition
    orange = (0,97,255)
    blue = (255,0,0)
    green = (0,255,0)
    black = (0,0,0)

    ## background segmentation
    # fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    # some kernels
    kernel_size = 5
    kernel1 = np.ones((kernel_size,kernel_size),np.float32)/kernel_size/kernel_size
    kernel2 = np.ones((10,10), np.uint8)/100
    s = 0
    screenshot = 0
    last_screenshot = 0
    while(cap.isOpened()):
        ## Capture frame-by-frame
        ret, frame_raw = cap.read()
        while not ret:
            ret,frame_raw = cap.read()
        frame_raw = cv2.flip(frame_raw,1)
        frame = frame_raw[:round(cap_height),:round(cap_width)]    # ROI of the image
        #cv2.imshow('raw_frame',frame)
        ## Color seperation and noise cancellation at HSV color space

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, skin_min, skin_max)
        res = cv2.bitwise_and(hsv, hsv, mask= mask)
        res = cv2.erode(res, kernel1, iterations=1)
        res = cv2.dilate(res, kernel1, iterations=1)
        # res = cv2.filter2D(res,-1,kernel2)    # hacky
        # cv2.imshow('hey2',res)

        ## Canny edge detection at Gray space.
        rgb = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        img = 255 * np.ones((round(cap_height),round(cap_width),3), np.uint8)
        #cv2.imshow('rgb_2',rgb)
        # cv2.imshow('rgb',rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('gray',gray)
        gray = cv2.filter2D(gray,-1,kernel2)    # hacky
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        #cv2.imshow('gray',gray)
        gray = cv2.threshold(gray,30, 255,cv2.THRESH_BINARY)[1]
        #gray = cv2.dilate(gray, kernel2, iterations=1)
        cv2.imshow('gray',gray)

        ## Canny edge detection at Gray space.
        # canny = cv2.Canny(gray, 300, 600)
        # cv2.imshow('canny',canny)
        # canny = cv2.erode(canny, kernel1, iterations=1)
        # canny = cv2.dilate(canny, kernel1, iterations=1)

        ## Background segmentation using motion detection (Optional)
        # fgmask = fgbg.apply(canny)

        ## main function: find finger cursor position & draw trajectory
        im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    # find all contours in the image
        if len(contours) !=0:
            c = max(contours, key = cv2.contourArea)  # find biggest contour in the image
            if cv2.contourArea(c) > 1000:
                topmost = tuple(c[c[:,:,1].argmin()][0])  # consider the topmost point of the biggest contour as cursor
                # gesture_index = cv2.matchShapes(c,gesture2[0],2,0.0)
                # obtain gesture matching index using gesture matching low_pass filter
                gesture_index = count(gray,c)
                #if gesture_index==1:
                    #gesture_index = 1
                #else:
                    #gesture_index = 0
                gesture_matching_filter.append(gesture_index)
                sum_gesture1 = 0
                sum_gesture2 = 0
                sum_gesture3 = 0
                for i in gesture_matching_filter:
                        if i==1:
                            sum_gesture1 += 1
                        elif i==2:
                            sum_gesture2 += 1
                        elif i==3:
                            sum_gesture3 += 1
                if sum_gesture1>=sum_gesture2 and sum_gesture1>=sum_gesture3:
                    gesture_index = 1
                elif sum_gesture2>=sum_gesture1 and sum_gesture2>=sum_gesture3:
                    gesture_index = 2
                elif sum_gesture3>=sum_gesture1 and sum_gesture3>=sum_gesture2:
                    gesture_index = 3
                print(gesture_index)
                screenshot = 0
                dist_pts = dist(topmost,topmost_last)  # calculate the distance of last cursor position and current cursor position
                if dist_pts < 150:  # filter big position change of cursor
                    try:
                        cv2.drawContours(rgb, [c], 0 , (0, 255, 0),5)
                        low_filter.append(topmost)
                        sum_x = 0
                        sum_y = 0
                        for i in low_filter:
                            sum_x += i[0]
                            sum_y += i[1]
                        topmost = (sum_x//low_filter_size, sum_y//low_filter_size)
                        if gesture_index == 2:
                            traj = np.array([], np.uint16)
                            traj = np.append( traj, topmost_last)
                            dist_pts = 0
                            dist_records = [dist_pts]
                            pass 
                        elif gesture_index == 3:
                            screenshot = 1
                            print('ss')#cv2.imwrite('screenshot' + str(s) + '.jpg', rgb)
                        else:
                            #gesture_index > 0:
                            traj = np.append( traj, topmost)
                            dist_records.append(dist_pts)
                            #traj = np.reshape( traj,(-1,1,2) )
                            #print(traj.shape)
                            # cv2.polylines(frame,[traj],0,orange,10)
                        topmost_last = topmost  # update cursor position
                    except:
                        print('error')
                        pass
                ## If move too fast, erase all trajectories
                elif dist_pts > 200:
                    traj = np.array(topmost_last, np.uint16)

        # traj = np.reshape( traj,(-1,1,2) )
        # print(traj.shape)
        # cv2.polylines(frame,[traj],0,orange,10)
        # print(traj)
        for i in range(1, len(dist_records)):
            # try:
            thickness = int(-0.072 * dist_records[i] + 13)
            cv2.line(frame, (traj[i*2-2],traj[i*2-1]), (traj[i*2],traj[i*2+1]), blue , thickness)
            cv2.line(rgb, (traj[i*2-2],traj[i*2-1]), (traj[i*2],traj[i*2+1]), blue , thickness)
            cv2.line(img, (traj[i*2-2],traj[i*2-1]), (traj[i*2],traj[i*2+1]), black , thickness)
            # except:
                # print(i)
        cv2.circle(frame, topmost_last, 10, blue , 3)
        cv2.circle(rgb, topmost_last, 10, blue , 3)
        if screenshot==1 and last_screenshot==0:
            s+=1
            cv2.imwrite('screenshot' + str(s) + '.jpg', img)
            print('screenshot')
            last_screenshot = 1
        elif screenshot==0:
            last_screenshot = 0
        ## Display the resulting frame
        # cv2.imshow('canny', canny)
        cv2.imshow('img', img)
        #cv2.imshow('rgb', rgb)
        # frame_raw = cv2.resize(frame_raw, (1024,768))
        cv2.imshow('frame', frame_raw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ## When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    device = 0    # if device = 0, use the built-in computer camera
    fingerCursor(device)
