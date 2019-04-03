#!/usr/bin/python

import cv2
import dlib
import numpy
import numpy as np
import sys
import os
import random

PREDICTOR_PATH = "dlib_align_model.dat"
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
eyes_bown_index = range(17,27) + range(36,48)
# Points used to line up the images.
# ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                            #    RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
MY_OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
]
# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

# initial the alignment model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass
class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[int(p.x), int(p.y)] for p in predictor(im, rects[0]).parts()])


# warped the particular regions, eyes and eyebrow for this program 
def extract_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)
    for group in MY_OVERLAY_POINTS:
        extract_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return im
    
def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))





imgPrefix = 'test_images/'
# read the name-list of images with open eyes 
openFile  = open('open_list.txt')
openList = openFile.readlines()
# read the name-list of images with close eyes
closeFile = open('close_list.txt')
closeList = closeFile.readlines()


for openeyeImageName in openList:
    if os.path.isfile(imgPrefix + openeyeImageName.split('\n')[0]) == False:
        continue
    openeyeImg = cv2.imread(imgPrefix + openeyeImageName.split('\n')[0])
    openLandmarks = get_landmarks(openeyeImg)

    for closeeyeImageName in closeList:
        if os.path.isfile(imgPrefix + closeeyeImageName.split('\n')[0]) == False:
            continue
        closeeyeImg = cv2.imread(imgPrefix + closeeyeImageName.split('\n')[0])
        closeLandmarks = get_landmarks(closeeyeImg)

        # now start to swap eyes
        M = transformation_from_points(openLandmarks[ALIGN_POINTS],
                               closeLandmarks[ALIGN_POINTS])

        
        # generate the images with closed eye
        mask = get_face_mask(closeeyeImg, closeLandmarks)
        warped_mask = warp_im(mask, M, openeyeImg.shape)
        combined_mask = np.max([get_face_mask(openeyeImg, openLandmarks), warped_mask],
                          axis=0)
        warped_im2 = warp_im(closeeyeImg, M, openeyeImg.shape)
        warped_corrected_im2 = correct_colours(openeyeImg, warped_im2, openLandmarks)
        output_im = openeyeImg * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
        # rescale value of pixels to avoid overflow when convert it to unit-8
        min_value = np.min(output_im)
        max_value = np.max(output_im)
        output_im = (output_im - min_value) * 255.0 / (max_value-min_value)
        # print(outout_im)
        print(np.max(output_im))
        print(np.min(output_im))
        output_im = output_im.astype(np.uint8)


        # get the landmarks of transformed images 
        reverse_M = transformation_from_points(closeLandmarks[ALIGN_POINTS], openLandmarks[ALIGN_POINTS])
        temp = np.ones((closeLandmarks.shape[0], 3))
        temp[:, 0:2] = closeLandmarks
        transformed_close_landmarks = reverse_M[:2, :] * temp.T
        transformed_close_landmarks = transformed_close_landmarks.T
        openLandmarks[eyes_bown_index, :]  = transformed_close_landmarks[eyes_bown_index, :]
        
        # visualize it
        for i in range(openLandmarks.shape[0]):
            cv2.circle(output_im, (openLandmarks[i,0], openLandmarks[i,1]), 1, (0,255,0), 1)
        cv2.imshow('output_im', output_im)
        key = cv2.waitKey(-1)
        if key == 27:
            cv2.destroyAllWindows()
            exit()



