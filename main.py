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

    # find the area of the circle
    area = (np.pi * radius * radius)

    # find the area of the segmented hand (contour)
    carea = cv2.contourArea(segmented)

    # if the hand is in a fist, return count of fingers as 0
    if carea/area >= 0.9:
        return 0

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

    ## skin color segmentation mask
    skin_min = np.array([0, 40, 50],np.uint8)  # HSV mask
    skin_max = np.array([50, 250, 255],np.uint8) # HSV mask

    ## trajectory drawing initialization
    #  an array that will hold the points of trajectory 
    traj = [deque(maxlen = 1024)]

    ## gesture_index low_pass filter 
    #  this increases accuracy for gesture recognition
    gesture_filter_size = 10
    gesture_matching_filter = deque([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], gesture_filter_size )
    gesture_index_thres = 0

    ## color definition
    orange = (0,97,255)
    blue = (255,0,0)
    green = (0,255,0)
    black = (0,0,0)

    ## some kernels
    #  used for morphological operations
    kernel_size = 5
    kernel1 = np.ones((kernel_size,kernel_size),np.float32)/kernel_size/kernel_size
    kernel2 = np.ones((10,10), np.uint8)/100

    ## index to keep track of screenshots
    s_index = 0
    screenshot = 0

    ## variable to avoid repetitive screenshots of the same image
    last_screenshot = 0

    ## This index will be used to mark position of pointer in traj
    index = 0

    ## keep looping till camera is open
    while(cap.isOpened()):
        ## Capture frame-by-frame
        ret, frame_raw = cap.read()
        while not ret:
            ret,frame_raw = cap.read()
 
        ## Flipping the frame to see same side of yours 
        frame_raw = cv2.flip(frame_raw,1)

        frame = frame_raw[:round(cap_height),:round(cap_width)]    # ROI of the image
  
        ## Color seperation and noise cancellation at HSV color space
        #  Converting to HSV color space 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #  Making a mask for skin color ranges 
        mask = cv2.inRange(hsv, skin_min, skin_max)

        #  Bitwise AND between the HSV frame and mask to segment hand 
        res = cv2.bitwise_and(hsv, hsv, mask= mask)

        # Erosion and Dilation for noise cancellation
        res = cv2.erode(res, kernel1, iterations=1)
        res = cv2.dilate(res, kernel1, iterations=1)

        ## defining frame img to show trajectory on white background
        img = 255 * np.ones((round(cap_height),round(cap_width),3), np.uint8)
        
        ## Converting to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        ## 2D Convulation for image filtering
        gray = cv2.filter2D(gray,-1,kernel2)  

        ## Gaussian Blur for image smoothing
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
       
        ## threshold image into black and white pixels
        gray = cv2.threshold(gray,30, 255,cv2.THRESH_BINARY)[1]
        cv2.imshow('gray',gray)


        ## main function: find finger cursor position & draw trajectory

        # find all contours in the image
        im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  
  
        # if any contour is found
        if len(contours) !=0:
            # find biggest contour in the image
            c = max(contours, key = cv2.contourArea)  

            if cv2.contourArea(c) > 1000:
                # consider the topmost point of the biggest contour as cursor
                topmost = tuple(c[c[:,:,1].argmin()][0])  
               
                # obtain gesture index(count of fingers) using count
                gesture_index = count(gray,c)
                
                # use low pass filter to increase accuracy
                gesture_matching_filter.append(gesture_index)
                
                # one fingered gesture in low pass filter
                sum_gesture1 = 0

                # two fingered gesture in low pass filter
                sum_gesture2 = 0

                # five fingered gesture in low pass filter
                sum_gesture3 = 0

                # fist gesture in low pass filter
                sum_gesture4 = 0
   
                for i in gesture_matching_filter:
                        if i==0:
                            sum_gesture0 += 1
                        elif i==1:
                            sum_gesture1 += 1
                        elif i==2:
                            sum_gesture2 += 1
                        elif i==3:
                            sum_gesture3 += 1
 
                ## take maximum of all these sums as final gesture_index
                if sum_gesture1>=sum_gesture2 and sum_gesture1>=sum_gesture3 and sum_gesture1>=sum_gesture4:
                    gesture_index = 1
                elif sum_gesture2>=sum_gesture1 and sum_gesture2>=sum_gesture3 and sum_gesture2>=sum_gesture4:
                    gesture_index = 2
                elif sum_gesture3>=sum_gesture1 and sum_gesture3>=sum_gesture2 and sum_gesture3>=sum_gesture4:
                    gesture_index = 3
                else:
                    gesture_index = 0
                
                # initialize screenshot variable to 0 for no screenshot
                screenshot = 0
                
                if gesture_index >= 0:  # filter big position change of cursor
                    try:
                        ## if two fingered gesture then clear frame
                        if gesture_index == 2:
                            traj = [deque(maxlen = 512)]
                            index = 0
                            pass 

                        ## if three fingered gesture then take screenshot
                        elif gesture_index == 3:
                            screenshot = 1

                        ## if one fingered gesture then store trajectory point and draw
                        elif gesture_index == 1:
                            traj[index].appendleft(topmost) 
                            
                        ## if fist gesture then do nothing and append the next deque
                        elif gesture_index == 0:
                            traj.append(deque(maxlen = 512))
                            index += 1
                    except:
                        print('error')
                        pass

        ## draw lines on frame from stored trajectory points 
        for i in range(len(traj)):
            for j in range(1, len(traj[i])):
                thickness = 10
                if traj[i][j-1] is None or traj[i][j] is None:
                    continue
                cv2.line(frame, tuple(traj[i][j-1]), tuple(traj[i][j]), blue , thickness)
                cv2.line(img, tuple(traj[i][j-1]), tuple(traj[i][j]), blue , thickness)
           
        ## if screenshot gesture and no repetition then save screenshot as jpg image
        if screenshot==1 and last_screenshot==0:
            s+=1
            cv2.imwrite('screenshot' + str(s) + '.jpg', img)
            last_screenshot = 1
        elif screenshot==0:
            last_screenshot = 0

        ## Display the resulting frames
        cv2.imshow('img', img)
        cv2.imshow('frame', frame_raw)

        ## if 'q' key is pressed then terminate program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ## When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    device = 0    # if device = 0, use the built-in computer camera
    fingerCursor(device)
