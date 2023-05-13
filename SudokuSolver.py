import math

import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt


def show_image(img):
    cv2.imshow('image', img)  # Display the image
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()  # Close all windows


def plot_many_images(images, titles, rows=1, columns=2):
    for i, image in enumerate(images):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image, 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # Hide tick marks
    plt.show()


def display_points(in_img, points, color=(0, 0, 255)):
    img = in_img.copy()
    for point in points:
        img = cv2.drawContours(img, [point], -1, color, 3)
    show_image(img)
    return img


def display_rects(in_img, contours, color=255):
    img = in_img.copy()
    for rect in contours:
        img = cv2.drawContours(img, [rect], -1, color, 3)
    show_image(img)
    return img


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
            box = np.array(box, dtype=np.int32)
            board = box
            break
    return board


# Projekt the board into a 900 * 900 window
def get_perspektiveProjection(img, board, height=900, width=900):
    board_vertecies = np.float32([board[0], board[1], board[2], board[3]])
    box_vertecies = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

    projektMatrix = cv2.getPerspectiveTransform(board_vertecies, box_vertecies)
    return cv2.warpPerspective(img, projektMatrix, (height, width))


def main():
    img = cv2.imread('sudoku2.jpg')
    board = locate_sudoku(img)
    if board is not None:
        display_points(img, board)
        display_rects(img, [board])


if __name__ == '__main__':
    main()
