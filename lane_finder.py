#!/usr/bin/env python

###
# Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
#     Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
#     Apply a distortion correction to raw images.
#     Use color transforms, gradients, etc., to create a thresholded binary image.
#     Apply a perspective transform to rectify binary image ("birds-eye view").
#     Detect lane pixels and fit to find the lane boundary.
#     Determine the curvature of the lane and vehicle position with respect to center.
#     Warp the detected lane boundaries back onto the original image.
#     Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
###

import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#%matplotlib qt

def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    #undist = np.copy(img)  # Delete this line
    return undist

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
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
        cv2.imshow('img',img)

        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                     [img_size[0]-offset, img_size[1]-offset],
                                     [offset, img_size[1]-offset]])
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)

        undist = cal_undistort(gray, objpoints, imgpoints)

        # e) use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(undist, M, img_size)

        cv2.imshow('warped', warped)

        cv2.waitKey(500)

#    return warped, M



cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.destroyAllWindows()

# Compute distortion coefficients
