import math

import numpy as np
import cv2
import imutils

def locate_sudoku(img):
    # get the gray img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply a bilateral filter to smooth out contrast
    # it is also possible to use a Gaussian Filter here
    # @params: d, sigma color, sigma Space
    bfilter_img = cv2.bilateralFilter(img_gray, 5, 35, 35)

    # find edges using Canny
    # @params: first and second Threshold
    edges = cv2.Canny(bfilter_img, 35, 150)

    # find the vertices of two contour.
    vertices = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(vertices)

    # sort contours based on the Area of a contour from big to small
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # find the actual board in the img
    board = None
    for contour in sorted_contours:
        approximated_poly = cv2.approxPolyDP(contour, 100, True)
        if len(approximated_poly) == 4:
            board = approximated_poly
            break
        if math.fabs(cv2.contourArea(approximated_poly)) > 1000:
            rect = cv2.minAreaRect(approximated_poly)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            board = box
            break

    if board is not None:
        test_img = cv2.drawContours(img.copy(), [board], -1, (0, 255, 0), 3)
        cv2.imshow("TestImage", test_img)


img = cv2.imread('sudoku3.jpg')
locate_sudoku(img)
cv2.waitKey()
