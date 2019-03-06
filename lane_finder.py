#!/usr/bin/env python

###
# Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# $    Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# $    Apply a distortion correction to raw images.
#     Use color transforms, gradients, etc., to create a thresholded binary image.
#     Apply a perspective transform to rectify binary image ("birds-eye view").
#     Detect lane pixels and fit to find the lane boundary.
#     Determine the curvature of the lane and vehicle position with respect to center.
#     Warp the detected lane boundaries back onto the original image.
#     Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
###

import pickle
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import cv2
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib qt

def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')
nx = 9
ny = 6

# Step through the list and search for chessboard corners
#def corners_unwarp(nx, ny) : #(img, nx, ny) : #, mtx, dist):
for fname in images:
    img = cv2.imread(fname)
    
#    undist = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x = gray.shape[1]
    y = gray.shape[0]

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

    # If found, add object points, image points
    if ret == True:
        offset = 100
        img_size = (x,y)

        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)

        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                    [img_size[0]-offset, img_size[1]-offset],
                                    [offset, img_size[1]-offset]])
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        #print(img.shape)
        #print(gray.shape)
        undist = cal_undistort(gray, objpoints, imgpoints)

        # e) use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(undist, M, img_size)
    
    plot = False
    # Ploting both images Original and Binary
    if(plot):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
        ax1.set_title('original')
        ax1.imshow(img)
        ax2.set_title('gray')
        ax2.imshow(gray, cmap='gray')  
        ax3.set_title('undist')
        ax3.imshow(undist)
        plt.show()

#     Use color transforms, gradients, etc., to create a thresholded binary image.
def thresholding(img, thresh=(20,100), sobel_kernel=3, s_thresh=(170,255)):
    ##### Threshold color channel
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    s_channel = hls[:,:,2]
    
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    scaled_s_binary = np.uint8(255*s_binary/np.max(s_binary))

    h_thresh = (15, 100)
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel > h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    undist = cal_undistort(gray, objpoints, imgpoints)
    
    plot = True
    # Ploting both images Original and Binary
    if(plot):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
        ax1.set_title('original')
        ax1.imshow(img[..., ::-1], cmap='gray')
        ax2.set_title('gray')
        ax2.imshow(gray, cmap='gray')  
        ax3.set_title('undist')
        ax3.imshow(undist, cmap='gray')
        plt.show()
    
    ### 1. start Absolute Sobel
    # get x gradient
    sobel_1 = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize = sobel_kernel)
    abs_sobel_x = np.absolute(sobel_1)
    scaled_sobel_x = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))
    binary_output_x = np.zeros_like(scaled_sobel_x)
    binary_output_x[(scaled_sobel_x >= 10) & (scaled_sobel_x <= 100)] = 255
    
    # get y gradient
    abs_sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_32F, 0, 1))#, ksize = sobel_kernel))
    scaled_sobel_y = np.uint8(255*abs_sobel_y/np.max(abs_sobel_y))
    binary_output_y = np.zeros_like(scaled_sobel_y)
    binary_output_y[(scaled_sobel_y >= 50) & (scaled_sobel_y <= 150)] = 255

    ### end Absolute Sobel
    
    ### 2. start Magnitude Sobel
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel)) # Take the derivative in x
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel)) # Take the derivative in y
    sobelx_sqd = abs_sobelx ** 2
    sobely_sqd = abs_sobely ** 2
    # 3) Calculate the magnitude 
    sobel_sqd = (sobelx_sqd + sobely_sqd) ** 0.5
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobelxy = np.uint8(255*sobel_sqd/np.max(sobel_sqd))
    sxy_binary = np.zeros_like(scaled_sobelxy)
    sxy_binary[(scaled_sobelxy >= thresh[0]) & (scaled_sobelxy <= thresh[1])] = 1
    scaled_sxy_binary = np.uint8(255*sxy_binary/np.max(sxy_binary))

    ### end Magnitude Sobel

    ### 3. start Direction Sobel
    dir_sobel_x = cv2.Sobel(scaled_s_binary, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
    dir_sobel_y = cv2.Sobel(scaled_s_binary, cv2.CV_32F, 0, 1, ksize=sobel_kernel)
    abs_grad = np.arctan2(np.absolute(dir_sobel_y), np.absolute(dir_sobel_x))
    dir_binary = np.zeros_like(abs_grad)
    dir_binary[(abs_grad >= 0.6) & (abs_grad <= 1.4)] = 1

    ### end Direction Sobel

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxy_binary)
    combined_binary[
            ((binary_output_x   == 255) & (binary_output_y   == 255)) |  #1.2.3_4_
 #           ((binary_output_x   == 255) & (scaled_sxy_binary >  1 )) |  #1.2_3.4_
#            ((binary_output_x   == 255) & (dir_binary        == 1)) |  #1.2_3_4.
#            ((binary_output_y   == 255) & (scaled_sxy_binary >  1)) |  #1_2.3.4_
#            ((binary_output_y   == 255) & (dir_binary        == 1)) |  #1_2.3_4.
            ((scaled_sxy_binary >  1  ) & (dir_binary        == 1))   #1_2_3.4.

            ] = 1
#    combined_binary[(s_binary == 1) | (scaled_h_binary == 1) |  (scaled_sxy_binary == 1)] = 1
    scaled_combined_binary = np.uint8(255*combined_binary/np.max(combined_binary))

#    while(1):
#    cv2.imshow("img", img)
#    cv2.imshow("s_channel", s_channel)
#    cv2.imshow("gray", gray)
#    cv2.imshow("sobel_1", sobel_1)
#    cv2.imshow("abs_sobel_x", abs_sobel_x)
#    cv2.imshow("scaled_sobel_x", scaled_sobel_x)
#    cv2.imshow("binary_output_x", binary_output_x)
#    cv2.imshow("binary_output_y", binary_output_y)
#    cv2.imshow("scaled_sxy_binary", scaled_sxy_binary)
#    cv2.imshow("dir_binary", dir_binary)
#    cv2.imshow("scaled_combined_binary", scaled_combined_binary)
#    k = cv2.waitKey(33)
##    if k==27:    # Esc key to stop
#        break

    plot = False
    # Ploting both images Original and Binary
    if(plot):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
        ax1.set_title('binary_output_x')
        ax1.imshow(binary_output_x, cmap='gray')
        ax2.set_title('scaled_sobel_x')
        ax2.imshow(scaled_sobel_x)  
        ax3.set_title('binary_output_x')
        ax3.imshow(binary_output_x)
        plt.show()

    return(scaled_combined_binary)

#     Apply a perspective transform to rectify binary image ("birds-eye view").
def calculate_M(image, x, y) :
    persp = np.float32([ [580,450], [180,680], [1130,680], [740,450]])
    birdseye = np.float32([ [0,0], [0,y], [x,y], [x,0]])
    global_M = cv2.getPerspectiveTransform(persp, birdseye)
    return global_M

def corner_unwarp(image, M, image_size) :
    return cv2.warpPerspective(image, M, (image_size), flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)

def corner_warp(image, M, warp_size) :
    return cv2.warpPerspective(image, M, (warp_size), flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)

#     Detect lane pixels and fit to find the lane boundary.
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 15
    # Set the width of the windows +/- margin
    margin = 180
    # Set minimum number of pixels found to recenter window
    minpix = 40

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img = find_lane_pixels(binary_warped)

    left_fit = np.zeros(leftx.size, dtype=np.float32)
    right_fit = np.zeros(rightx.size, dtype=np.float32)
    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    #try:
    left_fit = np.polyfit(lefty, leftx, 2)
#        try:
    right_fit = np.polyfit(righty, rightx, 2)
            
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
#        except TypeError:
#            print("Couldn't fit a right function")
#    except TypeError:
#        print("Couldn't fit a left function")

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fitx, left_lane_inds, right_fitx, right_lane_inds, ploty

#     Determine the curvature of the lane and vehicle position with respect to center.
def compute_curvature(left_fit, right_fit, leftx, rightx, ploty): #, left_fitx, right_fitx): #, leftx, lefty, rightx, righty):
 
    # Define conversions in x and y from pixels space to meters
    m_per_y_pix = 30.0/720 # meters per pixel in y dimension
    m_per_x_pix = 3.7/700 # meters per pixel in x dimension
 
    y_eval = np.max(ploty)
 
    left_fit_cr = np.polyfit((ploty*m_per_y_pix), (leftx*m_per_x_pix), 2, 2)
    right_fit_cr = np.polyfit(ploty*m_per_y_pix, rightx*m_per_x_pix, 2)

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
 
    return (left_curverad + right_curverad) / 2

def compute_offset(left_fit, right_fit, leftx, rightx, ploty):
    m_per_y_pix = 30.0/720 # meters per pixel in y dimension
    m_per_x_pix = 3.7/700 # meters per pixel in x dimension
    
    centre = abs(640.0 - ((rightx[-1]+leftx[-1])/2))
#    print(rightx)
#    print(rightx[-1])
    return centre * m_per_x_pix
#    shift_left = np.polyval()

#     Warp the detected lane boundaries back onto the original image.
def fill_poly(img, left, right):
    both_lines = np.concatenate((left, np.flipud(right)), axis=0)
    im = np.zeros((img.shape[::1][0],img.shape[::1][1],3), dtype=np.uint8)
    cv2.fillPoly(im, [both_lines.astype(np.int32)],(0,255,0))
    return im


def add_poly_to_image(image1, image2) :
    return image1 + image2

def process_image(img):
    #cv2.imshow("original image", img)
    #print(img.shape[::-1])
    image_size = (img.shape[::-1][1], img.shape[::-1][2])
    #print(image_size)
    M = calculate_M(img, image_size[0], image_size[1])
    
#    top_down_1 = corner_unwarp(img, M, image_size)
#    thresholding_1 = thresholding(top_down_1, thresh=(20,90), sobel_kernel=3)
#    out_img, left_fitx, left_lane_inds, right_fitx, right_lane_inds, ploty = fit_polynomial(thresholding_1)
#    warp_size = thresholding_1.shape[::-1]

    result = thresholding(img, thresh=(20,90), sobel_kernel=3)
    top_down = corner_unwarp(result, M, image_size)
    warp_size = top_down.shape[::-1]
    out_img, left_fitx, left_lane_inds, right_fitx, right_lane_inds, ploty = fit_polynomial(top_down)

    curverad = compute_curvature(left_fitx, right_fitx, np.array(left_fitx), right_fitx, ploty )
    centre = compute_offset(left_fitx, right_fitx, np.array(left_fitx), right_fitx, ploty )
        
    left_points = np.vstack((np.array(left_fitx), ploty)).T
    left_points = left_points.reshape((-1,1,2))

    right_points = np.vstack((np.array(right_fitx), ploty)).T
    right_points = right_points.reshape((-1,1,2))
    
    warp_orig = corner_unwarp(img, M, image_size)
    with_poly = fill_poly(warp_orig, left_points, right_points)
    rewarped_image = corner_warp(with_poly, M, warp_size)
    
    output_image = np.copy(rewarped_image)
    cv2.addWeighted(rewarped_image, 0.5,img, 0.5, 0.0, output_image)
    cv2.putText(output_image, "Road curvature: {:6.2f}m".format(curverad), (420, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
                thickness=5, color=(255, 255, 255))
    if centre > 640 :
        cv2.putText(output_image, "Offset right of centre: {:6.2f}m".format(centre), (420, 100), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
                thickness=5, color=(255, 255, 255))
    elif centre < 640 :
        cv2.putText(output_image, "Offset left of centre: {:6.2f}m".format(centre), (420, 100), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
                thickness=5, color=(255, 255, 255))
    else :
        cv2.putText(output_image, "Vehicle in centre!", (420, 100), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
                thickness=5, color=(255, 255, 255))
    
    while(0):
        cv2.imshow("original image", img)
#        cv2.imshow("top_down_1", top_down_1)
#        cv2.imshow("thresholding_1", thresholding_1)
        cv2.imshow("threshold_orig", result)
        cv2.imshow("out_img", out_img)
        cv2.imshow("warp_orig", warp_orig)
        cv2.imshow("with_poly", with_poly)
        cv2.imshow("output image", output_image)
        k = cv2.waitKey(33)
        if k==27:    # Esc key to stop
            break
    cv2.destroyAllWindows()
    return output_image

def process_video(img):
    img = img[..., ::-1]
    value = process_image(img)
    return value[..., ::-1]


#cv2.destroyAllWindows()
inputs = "test_images"
outputs = "output_images"


### process the images
for image_file in os.listdir(inputs) :
    img = cv2.imread(os.path.join(inputs, image_file))
    output_image = process_image(img)

#    cv2.imshow("lane_curve_w_poly", output_image)
    #print(os.path.join(outputs, "output_" + image_file))
    cv2.imwrite( os.path.join(outputs, "output_" + image_file), output_image );
    #img = lanes_detected.process(img, True, show_period = 1, blocking=False)

#### process the videos
#from moviepy.editor import VideoFileClip
#input_videos = ['project_video.mp4', 'harder_challenge_video.mp4', 'challenge_video.mp4', ]
#output_path = "output_videos"

#for file in input_videos :
#    video = VideoFileClip(file)#.subclip(24,25)
#    outclip = video.fl_image(process_video)
#    outclip.write_videofile(os.path.join(output_path, "output_" + file), audio=False)
