import numpy as np
import cv2
import imutils

def locate_sudoku(img):
    # get the gray img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply a bilateral filter to smooth out contrast
    # it is also possible to use a Gaussian Filter here
    # @params: d, sigma color, sigma Space
    bfilter_img = cv2.bilateralFilter(img_gray, 9, 35, 35)

    # find edges using Canny
    # @params: first and second Threshold
    edges = cv2.Canny(bfilter_img, 50, 200)

    # find the vertices of two contour.
    vertices = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(vertices)

    test_img = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
    cv2.imshow("TestImage", test_img)


img = cv2.imread('sudoku2.jpg')
locate_sudoku(img)
cv2.waitKey()
